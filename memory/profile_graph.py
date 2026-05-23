"""
ProfileGraph — 结构化用户健康档案（Neo4j）

独立于 mem0，只存放确认性事实：诊断、用药、过敏、病史。
读取：user_id 直接 fetch，不做语义搜索，每次请求全量注入 system prompt。
写入：harness 单次提取后路由过来，规则打分 → 直接写图 or SQLite pending 队列。
"""
import asyncio
import logging
import os
import time

from neo4j import AsyncGraphDatabase

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLD = 0.70
_ALLERGY_THRESHOLD = 0.85

# category → (relationship type, node label)
_CATEGORY_MAP = {
    "诊断":  ("HAS_CONDITION",      "Condition"),
    "用药":  ("TAKES",              "Medication"),
    "过敏":  ("ALLERGIC_TO",        "Allergen"),
    "病史":  ("HAS_HISTORY",        "MedHistory"),
    "家族史": ("HAS_FAMILY_HISTORY", "FamilyCondition"),
}

_REL_TO_LABEL = {
    "HAS_CONDITION":      "诊断",
    "TAKES":              "用药",
    "ALLERGIC_TO":        "过敏",
    "HAS_HISTORY":        "病史",
    "HAS_FAMILY_HISTORY": "家族史",
}

# ── 规则打分（替代 LLM confidence 打分）─────────────────────────────────────

_KW_DOCTOR = {"医生", "确诊", "诊断", "手术", "住院", "过敏史", "病史"}
_KW_STABLE = {"一直", "长期", "每日", "每天", "多年", "三年", "两年", "一年"}


def _score(category: str, name: str, detail: str) -> tuple[float, str]:
    text = f"{name} {detail}".lower()
    if any(kw in text for kw in _KW_DOCTOR):
        return 0.9, "doctor_confirmed"
    if category in ("病史", "家族史"):   # 历史性事实，自述即可信
        return 0.82, "self_reported"
    if any(kw in text for kw in _KW_STABLE):
        return 0.78, "self_reported"
    if category == "过敏":
        return 0.72, "self_reported"
    return 0.62, "self_reported"   # 低于 0.7，进 pending


# ── Neo4j driver singleton ────────────────────────────────────────────────────

_driver = None


def _get_driver():
    global _driver
    if _driver is None:
        _driver = AsyncGraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                os.getenv("NEO4J_USER", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "changeme"),
            ),
        )
    return _driver


async def init_schema() -> None:
    """幂等建约束，应用启动时调用一次。"""
    driver = _get_driver()
    try:
        async with driver.session() as session:
            await session.run(
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (u:ProfileUser) "
                    "REQUIRE u.user_id IS UNIQUE"
                )
            for label in ("Condition", "Medication", "Allergen", "MedHistory", "FamilyCondition"):
                await session.run(
                    f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) "
                    f"REQUIRE n.name IS UNIQUE"
                )
        logger.info("[profile_graph] Neo4j schema ready")
    except Exception as exc:
        logger.warning("[profile_graph] init_schema failed (Neo4j unavailable?): %s", exc)


# ── Graph write ───────────────────────────────────────────────────────────────

async def _merge_node(
    user_id: str,
    rel_type: str,
    node_label: str,
    name: str,
    confidence: float,
    source: str,
    detail: str = "",
) -> None:
    now = time.time()
    cypher = f"""
    MERGE (u:ProfileUser {{user_id: $uid}})
    MERGE (n:{node_label} {{name: $name}})
    MERGE (u)-[r:{rel_type}]->(n)
    ON CREATE SET r.confidence    = $conf,
                  r.source        = $src,
                  r.detail        = $detail,
                  r.mention_count = 1,
                  r.first_seen    = $now,
                  r.last_seen     = $now
    ON MATCH  SET r.confidence    = CASE WHEN $conf > r.confidence
                                         THEN $conf ELSE r.confidence END,
                  r.mention_count = r.mention_count + 1,
                  r.last_seen     = $now
    """
    try:
        async with _get_driver().session() as session:
            await session.run(
                cypher, uid=user_id, name=name, conf=confidence,
                src=source, detail=detail, now=now,
            )
        logger.debug("[profile_graph] merged user=%r rel=%s name=%r conf=%.2f",
                     user_id, rel_type, name, confidence)
        from memory.session_buffer import SessionBuffer
        await SessionBuffer.invalidate_profile_cache(user_id)
    except Exception as exc:
        logger.error("[profile_graph] merge failed user=%r name=%r: %s", user_id, name, exc)


async def upsert_from_extraction(user_id: str, facts: list[dict]) -> None:
    """接收 profile 类 facts 列表（已按 category 过滤）。
    facts 格式：[{"category": str, "name": str, "detail": str}]
    规则打分后：高置信写图，中置信存 pending_profile。
    """
    from memory import turn_store

    for fact in facts:
        category = fact.get("category", "")
        name = (fact.get("name") or "").strip()
        detail = (fact.get("detail") or "").strip()

        if not name or category not in _CATEGORY_MAP:
            continue

        confidence, source = _score(category, name, detail)
        rel_type, node_label = _CATEGORY_MAP[category]
        threshold = _ALLERGY_THRESHOLD if category == "过敏" else _DEFAULT_THRESHOLD

        if confidence >= threshold:
            await _merge_node(user_id, rel_type, node_label, name, confidence, source, detail)
        elif confidence >= 0.5:
            await turn_store.save_pending_profile(
                user_id, category, name, {}, confidence, source
            )


async def promote_pending(user_id: str, current_facts: list[dict]) -> None:
    """pending 里被再次提及的事实升格写图。"""
    from memory import turn_store

    current_names = {(f.get("name") or "").strip().lower() for f in current_facts}
    if not current_names:
        return

    pending = await turn_store.get_pending_profile(user_id)
    promote_ids: list[int] = []
    increment_ids: list[int] = []

    for p in pending:
        if p["name"].lower() not in current_names:
            continue
        if p["mention_count"] >= 1:   # 已存过一次，本次是第二次提及 → 升格
            rel_type, node_label = _CATEGORY_MAP.get(
                p["category"], ("HAS_CONDITION", "Condition")
            )
            await _merge_node(
                user_id, rel_type, node_label, p["name"],
                min(float(p["confidence"]) + 0.1, 1.0), p["source"],
            )
            promote_ids.append(p["id"])
        else:
            increment_ids.append(p["id"])

    if promote_ids:
        await turn_store.delete_pending_profiles(promote_ids)
    if increment_ids:
        await turn_store.increment_pending_mentions(increment_ids)


# ── Graph read ────────────────────────────────────────────────────────────────

async def get_profile_rows(user_id: str) -> list[dict]:
    """返回原始档案行，优先从 Redis 缓存读取，未命中则查 Neo4j 并回填缓存。"""
    from memory.session_buffer import SessionBuffer

    cached = await SessionBuffer.get_profile_cache(user_id)
    if cached is not None:
        logger.debug("[profile_graph] cache hit user=%r", user_id)
        return cached

    try:
        async with _get_driver().session() as session:
            result = await session.run(
                """
                MATCH (u:ProfileUser {user_id: $uid})-[r]->(n)
                WHERE r.confidence >= 0.7
                RETURN type(r) AS rel, n.name AS name, r.source AS src,
                       r.detail AS detail, r.confidence AS confidence
                ORDER BY r.confidence DESC
                """,
                uid=user_id,
            )
            rows = await result.data()
    except Exception as exc:
        logger.warning("[profile_graph] fetch failed user=%r: %s", user_id, exc)
        return []

    await SessionBuffer.set_profile_cache(user_id, rows)
    return rows


async def fetch_profile(user_id: str) -> str:
    """返回格式化档案文本用于注入 system prompt，无档案时返回空串。"""
    rows = await get_profile_rows(user_id)
    if not rows:
        return ""

    groups: dict[str, list[str]] = {}
    for row in rows:
        label = _REL_TO_LABEL.get(row["rel"], row["rel"])
        if row["rel"] == "HAS_FAMILY_HISTORY":
            detail_tag = f"（{row['detail']}）" if row.get("detail") else ""
            groups.setdefault(label, []).append(f"{row['name']}{detail_tag}")
        else:
            src_tag = "（医生确认）" if row["src"] == "doctor_confirmed" else ""
            groups.setdefault(label, []).append(f"{row['name']}{src_tag}")

    return "\n".join(f"{k}：{'、'.join(v)}" for k, v in groups.items())
