"""
RankedVectorStore — 替换 Mem0 OSS 内部 vector_store，在三个位置注入逻辑：

  insert() / update()  →  LLM 打重要性分，写入 payload["importance_score"]
  search()             →  复合重排：similarity * W + recency_decay * W + importance * W

这样 importance 由 LLM 在事实写入时一次性打分，search 直接读取，无额外开销。
"""
import math
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ── 复合分权重（可通过环境变量覆盖）──────────────────────────────────────────────
W_SIM  = float(os.getenv("RANK_W_SIM",  "0.6"))
W_REC  = float(os.getenv("RANK_W_REC",  "0.25"))
W_IMP  = float(os.getenv("RANK_W_IMP",  "0.15"))

# 时间半衰期（天）：N 天后 recency 权重衰减到 0.5
HALF_LIFE_DAYS = float(os.getenv("RANK_HALF_LIFE_DAYS", "30"))
_LAMBDA = math.log(2) / HALF_LIFE_DAYS

# 超采样倍数：fetch N 倍候选再重排
OVERSAMPLE = int(os.getenv("RANK_OVERSAMPLE", "4"))

# LLM 重要性打分 prompt
_IMPORTANCE_PROMPT = """\
Rate the medical importance of this health fact with a single decimal number from 0.0 to 1.0.

Scale:
0.9-1.0  Critical   — confirmed diagnosis, allergy, medication with dosage, lab result with value
0.5-0.8  Moderate   — symptom with detail, lifestyle factor with health impact, health goal
0.1-0.4  Low        — vague complaint, general preference, uncertain or hypothetical

Fact: {fact}

Reply with only the number, nothing else."""


def _llm_importance(text: str) -> float:
    """同步调用 DeepSeek 对一条 health fact 打重要性分（0.0–1.0）。
    运行在 asyncio.to_thread 内部，同步调用安全。
    失败时返回中性值 0.5。
    """
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url="https://api.deepseek.com",
        )
        resp = client.chat.completions.create(
            model=os.getenv("MEM0_LLM_MODEL", "deepseek-chat"),
            max_tokens=8,
            messages=[{"role": "user", "content": _IMPORTANCE_PROMPT.format(fact=text)}],
        )
        score = float(resp.choices[0].message.content.strip())
        return max(0.0, min(1.0, score))
    except Exception as exc:
        logger.warning("[ranked_vs] importance LLM failed, using 0.5: %s", exc)
        return 0.5


def _recency(payload: dict) -> float:
    raw = payload.get("updated_at") or payload.get("created_at")
    if not raw:
        return 0.5
    try:
        if isinstance(raw, (int, float)):
            dt = datetime.fromtimestamp(float(raw), tz=timezone.utc)
        else:
            dt = datetime.fromisoformat(str(raw))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        days = max((datetime.now(timezone.utc) - dt).total_seconds() / 86400, 0)
        return math.exp(-_LAMBDA * days)
    except Exception:
        return 0.5


def _composite(sim: float, payload: dict) -> float:
    importance = payload.get("importance_score", 0.5)
    return W_SIM * sim + W_REC * _recency(payload) + W_IMP * importance


class RankedVectorStore:
    """
    包装 Mem0 Qdrant vector store：
    - insert / update  →  LLM 打分后写入 payload["importance_score"]
    - search           →  复合重排
    - 其余方法          →  透传
    """

    def __init__(self, base_store):
        self._base = base_store

    # ── 写入拦截：insert（ADD 事件） ──────────────────────────────────────────────

    def insert(self, vectors: list, payloads: list = None, ids: list = None):
        if payloads:
            for payload in payloads:
                text = payload.get("data", "")
                if text:
                    payload["importance_score"] = _llm_importance(text)
                    logger.debug(
                        "[ranked_vs] insert importance=%.2f text=%r",
                        payload["importance_score"], text[:40],
                    )
        return self._base.insert(vectors=vectors, payloads=payloads, ids=ids)

    # ── 写入拦截：update（UPDATE 事件） ───────────────────────────────────────────

    def update(self, vector_id, vector=None, payload=None):
        if payload:
            text = payload.get("data", "")
            if text:
                payload["importance_score"] = _llm_importance(text)
                logger.debug(
                    "[ranked_vs] update importance=%.2f text=%r",
                    payload["importance_score"], text[:40],
                )
        return self._base.update(vector_id=vector_id, vector=vector, payload=payload)

    # ── 检索拦截：search（add() 内部相似记忆查找） ───────────────────────────────

    def search(self, query: str, vectors: list, limit: int = 5, filters: dict = None) -> list:
        candidates = self._base.search(
            query=query,
            vectors=vectors,
            limit=limit * OVERSAMPLE,
            filters=filters,
        )
        if not candidates:
            return candidates

        scored = [
            (_composite(getattr(p, "score", 0.0), getattr(p, "payload", {}) or {}), p)
            for p in candidates
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        result = [p for _, p in scored[:limit]]

        logger.debug("[ranked_vs] search candidates=%d → top%d", len(candidates), limit)
        return result

    # ── 其余方法透传 ──────────────────────────────────────────────────────────────

    def __getattr__(self, name: str):
        return getattr(self._base, name)
