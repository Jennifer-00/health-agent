"""
健康预警广播监控器。

定时执行流程：
  web_search 抓取预警 → Haiku 解析结构化列表 → 匹配用户 Mem0 档案 → Redis Pub/Sub 推送
"""
import hashlib
import json
import logging
import os

import redis.asyncio as aioredis

from memory.mem0_client import Mem0Client
from memory.pubsub import publish_alert

logger = logging.getLogger(__name__)

_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
_DEDUP_TTL  = 7 * 24 * 3600  # 同一条预警 7 天内不重复推送

_ALERT_QUERIES = [
    "药品召回 最新通告",
    "食品药品安全预警",
    "传染病疫情预警 最新",
    "健康风险提示 最新",
]


# ── 抓取与解析 ─────────────────────────────────────────────────────────────────

async def _fetch_alerts() -> list[dict]:
    """用 web_search 搜索预警信息，用 Haiku 解析成结构化列表。"""
    from openai import AsyncOpenAI
    from agent.tools import _web_search

    snippets: list[str] = []
    for q in _ALERT_QUERIES:
        text = await _web_search(q)
        if text and "未配置" not in text and "未找到" not in text:
            snippets.append(text)

    if not snippets:
        return []

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": (
                "从以下搜索结果中提取健康预警信息，返回 JSON 数组，每条格式：\n"
                '{"title": "标题", "summary": "一句话摘要", "keywords": ["药名或疾病名"]}\n'
                "只提取真实的召回/预警/疫情信息，无关广告忽略。若无则返回 []。\n\n"
                + "\n\n---\n\n".join(snippets)
            ),
        }],
    )

    raw = response.choices[0].message.content.strip()
    start, end = raw.find("["), raw.rfind("]") + 1
    if start == -1 or end == 0:
        return []
    try:
        return json.loads(raw[start:end])
    except Exception as exc:
        logger.warning("[alert_monitor] JSON 解析失败: %s", exc)
        return []


# ── 去重（Redis SET，TTL 7天）─────────────────────────────────────────────────

def _hash(alert: dict) -> str:
    return hashlib.md5(alert.get("title", "").encode()).hexdigest()[:12]


async def _already_sent(user_id: str, h: str) -> bool:
    r = aioredis.from_url(_REDIS_URL, decode_responses=True)
    try:
        return bool(await r.exists(f"alert_sent:{user_id}:{h}"))
    finally:
        await r.aclose()


async def _mark_sent(user_id: str, h: str) -> None:
    r = aioredis.from_url(_REDIS_URL, decode_responses=True)
    try:
        await r.setex(f"alert_sent:{user_id}:{h}", _DEDUP_TTL, "1")
    finally:
        await r.aclose()


# ── 相关性匹配 ─────────────────────────────────────────────────────────────────

async def _is_relevant(alert: dict, user_id: str) -> bool:
    """用预警关键词检索用户 Mem0 档案，命中则视为相关。"""
    keywords = alert.get("keywords", [])
    if not keywords:
        return False
    try:
        results = await Mem0Client(user_id=user_id).search(" ".join(keywords), limit=3)
        return len(results) > 0
    except Exception as exc:
        logger.warning("[alert_monitor] _is_relevant failed user=%r keywords=%r: %s", user_id, keywords, exc)
        return False


# ── 主入口 ────────────────────────────────────────────────────────────────────

async def run_alert_monitor() -> None:
    """APScheduler 定时任务入口：抓取预警 → 匹配用户 → 发布到 Redis Pub/Sub。"""
    logger.info("[alert_monitor] 开始检查健康预警")

    alerts = await _fetch_alerts()
    if not alerts:
        logger.info("[alert_monitor] 本次未发现新预警")
        return
    logger.info("[alert_monitor] 解析到 %d 条预警", len(alerts))

    try:
        from memory.mem0_client import list_users
        users = await list_users()
        if not users:
            logger.info("[alert_monitor] OSS 模式无用户列表，跳过主动推送")
            return
    except Exception as exc:
        logger.error("[alert_monitor] 获取用户列表失败: %s", exc)
        return

    for user in users:
        user_id = user.get("name", "")
        if not user_id or user_id.isdigit():
            continue
        for alert in alerts:
            h = _hash(alert)
            if await _already_sent(user_id, h):
                continue
            if await _is_relevant(alert, user_id):
                await publish_alert(user_id, {"type": "health_alert", **alert})
                await _mark_sent(user_id, h)
                logger.info("[alert_monitor] 推送 user=%r title=%r", user_id, alert.get("title"))
