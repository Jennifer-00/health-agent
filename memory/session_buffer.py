import json
import os
import logging

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
SESSION_TTL = int(os.getenv("SESSION_TTL_SECONDS", "3600"))

# memory_search 冷却参数
# 冷却轮数与 MAX_HISTORY_MESSAGES 保持一致，改一个地方即可
MEM_SEARCH_COOLDOWN_TURNS = int(os.getenv("MAX_HISTORY_MESSAGES", "10")) // 2
MEM_SEARCH_COOLDOWN_TTL   = int(os.getenv("MEM_SEARCH_COOLDOWN_TTL_SECONDS", "1800"))

# 模块级连接池，避免每次请求新建连接
_pool = aioredis.ConnectionPool.from_url(
    REDIS_URL,
    decode_responses=True,
    socket_timeout=5,
    socket_connect_timeout=5,
    max_connections=10,
)

# Redis 不可用时的内存降级缓存
_fallback: dict[str, list] = {}
_consult_fallback: dict[str, dict] = {}


class SessionBuffer:
    """会话级短期记忆，使用 Redis 存储并设置 TTL。Redis 不可用时降级为内存缓存。"""

    def __init__(self, session_id: str):
        self.key = f"session:{session_id}"

    def _client(self):
        return aioredis.Redis(connection_pool=_pool)

    async def get(self) -> list[dict]:
        try:
            r = self._client()
            raw = await r.get(self.key)
            return json.loads(raw) if raw else []
        except Exception:
            return _fallback.get(self.key, [])

    async def set(self, messages: list[dict]) -> None:
        try:
            r = self._client()
            await r.setex(self.key, SESSION_TTL, json.dumps(messages))
        except Exception:
            _fallback[self.key] = messages

    async def append(self, message: dict) -> None:
        history = await self.get()
        history.append(message)
        await self.set(history)

    async def clear(self) -> None:
        try:
            r = self._client()
            await r.delete(self.key)
        except Exception:
            _fallback.pop(self.key, None)

    # ── 问诊状态 ──────────────────────────────────────────────────────────────

    @property
    def _consult_key(self) -> str:
        return f"consult:{self.key}"

    async def get_consult(self) -> dict:
        """返回问诊状态 {"active": bool, "history": [...]}"""
        try:
            r = self._client()
            raw = await r.get(self._consult_key)
            return json.loads(raw) if raw else {"active": False, "history": []}
        except Exception:
            return _consult_fallback.get(self._consult_key, {"active": False, "history": []})

    async def set_consult(self, state: dict) -> None:
        try:
            r = self._client()
            await r.setex(self._consult_key, SESSION_TTL, json.dumps(state))
        except Exception:
            _consult_fallback[self._consult_key] = state

    async def clear_consult(self) -> None:
        try:
            r = self._client()
            await r.delete(self._consult_key)
        except Exception:
            _consult_fallback.pop(self._consult_key, None)

    # ── memory_search 冷却计数器 ──────────────────────────────────────────────
    # 调用 memory_search 后设置倒计时，后续几轮无需重复检索。
    # 值含义：剩余可跳过轮数（0 = 冷却结束，需重新检索）

    @property
    def _mem_search_cooldown_key(self) -> str:
        return f"mem_search_cd:{self.key}"

    async def get_mem_search_cooldown(self) -> int:
        """返回剩余冷却轮数，0 表示冷却结束。"""
        try:
            r = self._client()
            val = await r.get(self._mem_search_cooldown_key)
            return int(val) if val else 0
        except Exception:
            return 0

    async def set_mem_search_cooldown(self) -> None:
        """重置冷却计数器。轮数和时间 TTL 均从环境变量读取。"""
        try:
            r = self._client()
            await r.setex(
                self._mem_search_cooldown_key,
                MEM_SEARCH_COOLDOWN_TTL,
                str(MEM_SEARCH_COOLDOWN_TURNS),
            )
        except Exception:
            pass

    # ── 记忆预热缓存（user 级别，跨 session 共享）────────────────────────────────
    # 页面加载时触发 warmup，提前拉取全量记忆存入此缓存。
    # memory_search 优先读取，命中则跳过 Mem0 / Zep 网络调用，消除冷启动延迟。
    # TTL = 10 分钟，过期后自动回落到全量检索。

    @staticmethod
    def _prefetch_key(user_id: str) -> str:
        return f"mem_prefetch:{user_id}"

    @staticmethod
    async def get_mem_prefetch(user_id: str) -> list[dict] | None:
        """返回预热缓存的记忆列表；缓存不存在时返回 None。"""
        key = SessionBuffer._prefetch_key(user_id)
        try:
            r = aioredis.Redis(connection_pool=_pool)
            raw = await r.get(key)
            return json.loads(raw) if raw else None
        except Exception:
            return None

    @staticmethod
    async def set_mem_prefetch(user_id: str, memories: list[dict]) -> None:
        """写入预热缓存，TTL 10 分钟。"""
        key = SessionBuffer._prefetch_key(user_id)
        try:
            r = aioredis.Redis(connection_pool=_pool)
            await r.setex(key, 600, json.dumps(memories, ensure_ascii=False))
        except Exception:
            pass

    async def decrement_mem_search_cooldown(self) -> int:
        """每轮调用一次，计数器减 1，返回减后剩余值。"""
        try:
            r = self._client()
            new_val = await r.decr(self._mem_search_cooldown_key)
            if new_val <= 0:
                await r.delete(self._mem_search_cooldown_key)
                return 0
            return new_val
        except Exception:
            return 0
