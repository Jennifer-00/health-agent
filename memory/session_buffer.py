import json
import os
import logging

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


def _m_inc_degrade(operation: str):
    try:
        from agent.metrics import redis_degradation
        redis_degradation.labels(operation=operation).inc()
    except Exception:
        pass


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
        except Exception as exc:
            logger.warning("[session_buffer] Redis get failed, using fallback key=%s: %s", self.key, exc)
            _m_inc_degrade("session_load")
            return _fallback.get(self.key, [])

    async def set(self, messages: list[dict]) -> None:
        try:
            r = self._client()
            await r.setex(self.key, SESSION_TTL, json.dumps(messages))
        except Exception as exc:
            logger.warning("[session_buffer] Redis set failed, using fallback key=%s: %s", self.key, exc)
            _m_inc_degrade("session_save")
            _fallback[self.key] = messages

    async def append(self, message: dict) -> None:
        history = await self.get()
        history.append(message)
        await self.set(history)

    async def clear(self) -> None:
        try:
            r = self._client()
            await r.delete(self.key)
        except Exception as exc:
            logger.warning("[session_buffer] Redis delete failed, clearing fallback key=%s: %s", self.key, exc)
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
        except Exception as exc:
            logger.warning("[session_buffer] Redis get_consult failed, using fallback key=%s: %s", self._consult_key, exc)
            _m_inc_degrade("consult_load")
            return _consult_fallback.get(self._consult_key, {"active": False, "history": []})

    async def set_consult(self, state: dict) -> None:
        try:
            r = self._client()
            await r.setex(self._consult_key, SESSION_TTL, json.dumps(state))
        except Exception as exc:
            logger.warning("[session_buffer] Redis set_consult failed, using fallback key=%s: %s", self._consult_key, exc)
            _m_inc_degrade("consult_save")
            _consult_fallback[self._consult_key] = state

    async def clear_consult(self) -> None:
        try:
            r = self._client()
            await r.delete(self._consult_key)
        except Exception as exc:
            logger.warning("[session_buffer] Redis clear_consult failed key=%s: %s", self._consult_key, exc)
            _consult_fallback.pop(self._consult_key, None)

    # ── memory_search 冷却计数器 ──────────────────────────────────────────────

    @property
    def _mem_search_cooldown_key(self) -> str:
        return f"mem_search_cd:{self.key}"

    async def get_mem_search_cooldown(self) -> int:
        """返回剩余冷却轮数，0 表示冷却结束。"""
        try:
            r = self._client()
            val = await r.get(self._mem_search_cooldown_key)
            return int(val) if val else 0
        except Exception as exc:
            logger.warning("[session_buffer] Redis get_cooldown failed key=%s: %s", self._mem_search_cooldown_key, exc)
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
        except Exception as exc:
            logger.warning("[session_buffer] Redis set_cooldown failed key=%s: %s", self._mem_search_cooldown_key, exc)

    # ── 记忆预热缓存（user 级别，跨 session 共享）────────────────────────────────

    @staticmethod
    def _prefetch_key(user_id: str) -> str:
        return f"mem_prefetch:{user_id}"

    @staticmethod
    async def get_mem_prefetch(user_id: str) -> list[dict] | None:
        """Mem0 超时降级缓存（写入由外部在 add 后调用）；不存在时返回 None。"""
        key = SessionBuffer._prefetch_key(user_id)
        try:
            r = aioredis.Redis(connection_pool=_pool)
            raw = await r.get(key)
            return json.loads(raw) if raw else None
        except Exception as exc:
            logger.warning("[session_buffer] Redis get_prefetch failed user=%s: %s", user_id, exc)
            return None

    async def decrement_mem_search_cooldown(self) -> int:
        """每轮调用一次，计数器减 1，返回减后剩余值。"""
        try:
            r = self._client()
            new_val = await r.decr(self._mem_search_cooldown_key)
            if new_val <= 0:
                await r.delete(self._mem_search_cooldown_key)
                return 0
            return new_val
        except Exception as exc:
            logger.warning("[session_buffer] Redis decrement_cooldown failed key=%s: %s", self._mem_search_cooldown_key, exc)
            return 0

    # ── Mem0 待写队列（session 级别）──────────────────────────────────────────────

    @property
    def _mem_pending_key(self) -> str:
        return f"mem_pending:{self.key}"

    async def append_mem_pending(self, user_content: str, assistant_content: str) -> int:
        """追加一轮对话到待写队列，返回当前队列长度（即已累积轮数）。"""
        item = json.dumps({"user": user_content, "assistant": assistant_content}, ensure_ascii=False)
        try:
            r = self._client()
            count = await r.rpush(self._mem_pending_key, item)
            await r.expire(self._mem_pending_key, SESSION_TTL)
            return count
        except Exception as exc:
            logger.warning("[session_buffer] append_mem_pending failed key=%s: %s", self._mem_pending_key, exc)
            return 0

    # ── 健康档案缓存（user 级别，Neo4j 查询结果）────────────────────────────────

    _PROFILE_CACHE_TTL = int(os.getenv("PROFILE_CACHE_TTL", "1800"))  # 默认 30 分钟

    @staticmethod
    async def get_profile_cache(user_id: str) -> list[dict] | None:
        key = f"profile:{user_id}"
        try:
            r = aioredis.Redis(connection_pool=_pool)
            raw = await r.get(key)
            return json.loads(raw) if raw else None
        except Exception as exc:
            logger.warning("[session_buffer] get_profile_cache failed user=%s: %s", user_id, exc)
            return None

    @staticmethod
    async def set_profile_cache(user_id: str, rows: list[dict]) -> None:
        key = f"profile:{user_id}"
        try:
            r = aioredis.Redis(connection_pool=_pool)
            await r.setex(key, SessionBuffer._PROFILE_CACHE_TTL,
                          json.dumps(rows, ensure_ascii=False))
        except Exception as exc:
            logger.warning("[session_buffer] set_profile_cache failed user=%s: %s", user_id, exc)

    @staticmethod
    async def invalidate_profile_cache(user_id: str) -> None:
        key = f"profile:{user_id}"
        try:
            r = aioredis.Redis(connection_pool=_pool)
            await r.delete(key)
        except Exception as exc:
            logger.warning("[session_buffer] invalidate_profile_cache failed user=%s: %s", user_id, exc)

    async def drain_mem_pending(self) -> list[dict]:
        """取出并清空待写队列，返回所有轮次的 {user, assistant} 列表。"""
        try:
            r = self._client()
            items = await r.lrange(self._mem_pending_key, 0, -1)
            if items:
                await r.delete(self._mem_pending_key)
            return [json.loads(i) for i in items] if items else []
        except Exception as exc:
            logger.warning("[session_buffer] drain_mem_pending failed key=%s: %s", self._mem_pending_key, exc)
            return []
