"""
mem0_client.py — OSS Memory 封装（本地运行，无需 Mem0 云 API Key）

LLM:         DeepSeek（OpenAI 兼容接口，提取事实）
Embedder:    BAAI/bge-m3（与 RAG 保持一致，1024 维 dense）
Vector DB:   Qdrant（复用现有实例，collection = health_memories）
"""
import asyncio
import logging
import os

import httpx
from mem0 import Memory

logger = logging.getLogger(__name__)

_mem: Memory | None = None

# mem0 内部提取 prompt — 只负责情景类，档案类由独立的 extract_profile_facts() 处理
_MEM0_EXTRACTION_PROMPT = """\
You are a health information extractor for a personal health assistant.
Extract episodic health facts from user messages in a conversation.

# CRITICAL RULES
- Extract ONLY from the USER's messages. Ignore assistant and system messages entirely.
- Preserve relative time expressions as-is — do not convert to absolute dates.
- Detect the user's language and record facts in that same language.
- Ignore: greetings, questions, speculation ("好像","可能","不确定"), hypotheticals.
- Do NOT extract diagnoses, medications, allergies, or medical history — those are stored separately.
- If nothing relevant is found, return {"facts": []}

# CATEGORIES
- [症状] — current/recent symptom with context (onset, triggers, severity)
- [检查] — lab/exam result; must include numeric value and date if mentioned
- [生活] — sleep, diet, exercise (only when user explicitly raises them)

# OUTPUT FORMAT (json object)
{"facts": ["[症状] ...", "[检查] ...", ...]}

# EXAMPLES

User: 最近三天眼睛很酸，盯屏幕更严重。
Output: {"facts": ["[症状] 最近三天眼睛酸涩，长时间盯屏幕后加重"]}

User: 上周查了血糖，空腹6.8mmol/L，医生说偏高。
Output: {"facts": ["[检查] 上周空腹血糖6.8mmol/L，医生告知偏高"]}

User: 对青霉素过敏，高血压三年了，一直吃氨氯地平5mg。
Output: {"facts": []}

User: 好的谢谢
Output: {"facts": []}
"""

# 档案类提取 prompt — 供 extract_profile_facts() 调用，仅提 5 类结构化事实写 Neo4j
_PROFILE_EXTRACTION_PROMPT = """\
You are a health profile extractor for a personal health assistant.
Extract confirmed structural health facts from user messages in a conversation.

# CRITICAL RULES
- Extract ONLY from the USER's messages. Ignore assistant and system messages entirely.
- Preserve relative time expressions as-is — do not convert to absolute dates.
- Detect the user's language and record facts in that same language.
- Do NOT extract symptoms, lab results, or lifestyle — those are stored separately.
- If nothing profile-relevant is found, return {"facts": []}

# CATEGORIES
Return each fact as {"category": "...", "name": "...", "detail": "..."}

- 诊断  — confirmed diagnosis the USER has; name = condition name
- 用药  — medication USER is taking; name = drug name, detail = dose/frequency
- 过敏  — USER's allergy; name = substance, detail = reaction type
- 病史  — USER's past condition/surgery; name = condition/procedure name
- 家族史 — blood relative's condition; name = condition, detail = which relative ("母亲患有" etc.)

# OUTPUT FORMAT (json object)
{"facts": [{"category": "诊断", "name": "高血压", "detail": "三年"}, ...]}

# EXAMPLES

User: 对青霉素过敏，之前用过出了皮疹。高血压三年了，一直吃氨氯地平5mg。
Output: {"facts": [{"category": "过敏", "name": "青霉素", "detail": "曾出现皮疹反应"}, {"category": "病史", "name": "高血压", "detail": "病史三年"}, {"category": "用药", "name": "氨氯地平", "detail": "5mg，长期服用"}]}

User: 我妈有2型糖尿病。最近三天眼睛很酸。
Output: {"facts": [{"category": "家族史", "name": "2型糖尿病", "detail": "母亲患有"}]}

User: 上周查了血糖，空腹6.8mmol/L。
Output: {"facts": []}

User: 好的谢谢
Output: {"facts": []}
"""


def _detect_device() -> str:
    device = os.getenv("EMBED_DEVICE")
    if device:
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _build_config(device: str | None = None) -> dict:
    if device is None:
        device = _detect_device()
    logger.info("[mem0] embedding device=%s", device)

    # low_cpu_mem_usage=False 阻止 accelerate 创建 meta 张量，
    # 保证权重先实例化在 CPU 上，SentenceTransformer 再 .to(device) 时不会报
    # "Cannot copy out of meta tensor" 错误。
    embedder_model_kwargs: dict = {
        "device": device,
        "model_kwargs": {"low_cpu_mem_usage": False},
    }

    return {
        "llm": {
            "provider": "openai",
            "config": {
                "model": os.getenv("MEM0_LLM_MODEL", "deepseek-chat"),
                "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
                "openai_base_url": "https://api.deepseek.com",
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": "BAAI/bge-m3",
                "embedding_dims": 1024,
                "model_kwargs": embedder_model_kwargs,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "url": os.getenv("QDRANT_URL", "http://localhost:6333"),
                "collection_name": os.getenv("MEM0_COLLECTION", "health_memories"),
                "embedding_model_dims": 1024,
            },
        },
        "reranker": {
            "provider": "huggingface",
            "config": {
                "model": "BAAI/bge-reranker-v2-m3",
                "top_n": 5,
            },
        },
        "custom_fact_extraction_prompt": _MEM0_EXTRACTION_PROMPT,
    }


def _get_client() -> Memory:
    global _mem
    if _mem is None:
        # 先尝试配置的设备（通常是 cuda），meta tensor 问题时自动降级 cpu
        devices_to_try = [_detect_device()]
        if devices_to_try[0] != "cpu":
            devices_to_try.append("cpu")

        last_exc: Exception | None = None
        for device in devices_to_try:
            try:
                _mem = Memory.from_config(_build_config(device))
                from memory.ranked_vector_store import RankedVectorStore
                _mem.vector_store = RankedVectorStore(_mem.vector_store)
                logger.info("[mem0] OSS Memory client ready (device=%s)", device)
                break
            except Exception as exc:
                last_exc = exc
                err_str = str(exc)
                is_meta = "meta tensor" in err_str.lower() or "Cannot copy out of meta" in err_str
                is_cuda = "cuda" in err_str.lower() or "CUDA" in err_str
                if (is_meta or is_cuda) and device != "cpu":
                    logger.warning("[mem0] %s load failed (%s), retrying with cpu",
                                   device, "meta tensor" if is_meta else "CUDA error")
                else:
                    logger.warning("[mem0] client init failed: %s", exc)
                    break

        if _mem is None:
            raise RuntimeError("Mem0 client unavailable") from last_exc
    return _mem


async def list_users() -> list[dict]:
    """OSS 版本无原生 users() 接口，返回空列表；启动预热和 alert monitor 会跳过批量遍历。"""
    return []


# 每个 user_id 独占一把写锁，防止并发 add() 造成 Qdrant 去重失效产生重复条目
_write_locks: dict[str, asyncio.Lock] = {}


def _get_write_lock(user_id: str) -> asyncio.Lock:
    if user_id not in _write_locks:
        _write_locks[user_id] = asyncio.Lock()
    return _write_locks[user_id]


class Mem0Client:
    """封装 Memory OSS add / search / update / get_all / delete，绑定 user_id。
    所有方法通过 asyncio.to_thread 将同步调用转为异步。
    """

    def __init__(self, user_id: str):
        self.user_id = user_id

    async def add(self, content: str, metadata: dict | None = None) -> dict:
        def _add():
            return _get_client().add(
                content,
                user_id=self.user_id,
                metadata=metadata or {},
            )
        # 同一用户的写操作串行执行，避免两个并发 add() 都查不到对方的写入而重复插入
        async with _get_write_lock(self.user_id):
            try:
                result = await asyncio.to_thread(_add)
                return result or {}
            except Exception as exc:
                logger.error("[mem0] add failed user=%r: %s", self.user_id, exc, exc_info=True)
                raise

    async def search(self, query: str, limit: int = 5) -> list[dict]:
        import time as _t
        def _search():
            t0 = _t.monotonic()
            result = _get_client().search(query, user_id=self.user_id, limit=limit, rerank=False)
            logger.info("[mem0_search] total=%.0fms user=%r", (_t.monotonic() - t0) * 1000, self.user_id)
            return result
        _timeout = float(os.getenv("MEM0_SEARCH_TIMEOUT", "30"))
        try:
            raw = await asyncio.wait_for(asyncio.to_thread(_search), timeout=_timeout)

            if isinstance(raw, dict):
                results = raw.get("results", [])
            else:
                results = raw if isinstance(raw, list) else []

            memories = [m for m in results if m]

            if len(memories) > 1:
                reranker = _get_client().reranker
                if reranker:
                    memories = reranker.rerank(query, memories, top_k=limit)

            return memories
        except Exception as exc:
            logger.error("[mem0] search failed user=%r query=%r: %s", self.user_id, query, exc, exc_info=True)
            return []

    async def get_all(self) -> list[dict]:
        def _get_all():
            return _get_client().get_all(user_id=self.user_id)
        try:
            raw = await asyncio.to_thread(_get_all)
            raw = raw if isinstance(raw, list) else raw.get("results", [])
            return [m for m in raw if m]
        except Exception as exc:
            logger.error("[mem0] get_all failed user=%r: %s", self.user_id, exc, exc_info=True)
            return []

    async def update(self, memory_id: str, content: str) -> dict:
        def _update():
            return _get_client().update(memory_id, content)
        try:
            return await asyncio.to_thread(_update) or {}
        except Exception as exc:
            logger.error("[mem0] update failed user=%r memory_id=%r: %s", self.user_id, memory_id, exc, exc_info=True)
            raise

    async def delete(self, memory_id: str) -> None:
        def _delete():
            _get_client().delete(memory_id)
        try:
            await asyncio.to_thread(_delete)
        except Exception as exc:
            logger.error("[mem0] delete failed user=%r memory_id=%r: %s", self.user_id, memory_id, exc, exc_info=True)
            raise


# ── 档案类 fact 提取（Neo4j 路）────────────────────────────────────────────────

import json as _json
import re as _re


def _run_profile_extraction(text: str) -> list[dict]:
    """同步调用 DeepSeek，仅提取档案类 5 个 category，返回结构化列表。"""
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url="https://api.deepseek.com",
        )
        resp = client.chat.completions.create(
            model=os.getenv("MEM0_LLM_MODEL", "deepseek-chat"),
            max_tokens=800,
            messages=[{
                "role": "user",
                "content": f"{_PROFILE_EXTRACTION_PROMPT}\n\nConversation:\n{text}",
            }],
        )
        raw = resp.choices[0].message.content.strip()
        match = _re.search(r"\{.*\}", raw, _re.DOTALL)
        if not match:
            return []
        return _json.loads(match.group()).get("facts", [])
    except Exception as exc:
        logger.warning("[mem0] extract_profile_facts failed: %s", exc)
        return []


async def extract_profile_facts(text: str) -> list[dict]:
    """提取档案类 facts（诊断/用药/过敏/病史/家族史），供 harness 写 Neo4j。"""
    return await asyncio.to_thread(_run_profile_extraction, text)
