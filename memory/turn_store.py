"""
SQLite 兜底存储：每轮对话在 Redis pending 队列之外写入本地 DB，
防止进程崩溃 / 浏览器强杀导致 pending 队列丢失、症状信息永久消失。

恢复路径：下次 warmup 时读取 flushed=0 的行，批量补写 Mem0，再标记 flushed=1。
"""
import logging
import os
import time

import aiosqlite

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("TURN_STORE_PATH", "data/turns.db")

_DDL = """
CREATE TABLE IF NOT EXISTS pending_turns (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      TEXT    NOT NULL,
    session_id   TEXT    NOT NULL,
    user_content TEXT    NOT NULL,
    asst_content TEXT    NOT NULL,
    flushed      INTEGER NOT NULL DEFAULT 0,
    created_at   REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_user_flushed
    ON pending_turns (user_id, flushed);

CREATE TABLE IF NOT EXISTS pending_profile (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       TEXT    NOT NULL,
    category      TEXT    NOT NULL,
    name          TEXT    NOT NULL,
    attributes    TEXT    NOT NULL DEFAULT '{}',
    confidence    REAL    NOT NULL,
    source        TEXT    NOT NULL,
    mention_count INTEGER NOT NULL DEFAULT 1,
    created_at    REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pending_profile_user
    ON pending_profile (user_id, category, name);
"""


async def init_db() -> None:
    """建表（幂等），应用启动时调用一次。"""
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(_DDL)
        await db.commit()
    logger.info("[turn_store] DB ready: %s", DB_PATH)


async def save_turn(
    user_id: str,
    session_id: str,
    user_content: str,
    asst_content: str,
) -> None:
    """每轮对话结束后写入一行（flushed=0）。"""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "INSERT INTO pending_turns "
                "(user_id, session_id, user_content, asst_content, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (user_id, session_id, user_content, asst_content, time.time()),
            )
            await db.commit()
    except Exception as exc:
        logger.error("[turn_store] save_turn failed user=%r: %s", user_id, exc)


async def get_unflushed(user_id: str) -> list[dict]:
    """返回该用户所有未写入 Mem0 的轮次（跨 session），按时间升序。"""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT id, session_id, user_content, asst_content "
                "FROM pending_turns "
                "WHERE user_id = ? AND flushed = 0 "
                "ORDER BY created_at",
                (user_id,),
            ) as cur:
                rows = await cur.fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.error("[turn_store] get_unflushed failed user=%r: %s", user_id, exc)
        return []


async def mark_flushed_by_session(user_id: str, session_id: str) -> None:
    """将某 session 所有 pending 行标记为已写入 Mem0。"""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "UPDATE pending_turns SET flushed = 1 "
                "WHERE user_id = ? AND session_id = ? AND flushed = 0",
                (user_id, session_id),
            )
            await db.commit()
    except Exception as exc:
        logger.error("[turn_store] mark_flushed_by_session failed user=%r session=%r: %s", user_id, session_id, exc)


async def mark_flushed_by_ids(ids: list[int]) -> None:
    """按主键批量标记 flushed=1（用于跨 session 恢复写入后）。"""
    if not ids:
        return
    try:
        placeholders = ",".join("?" * len(ids))
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                f"UPDATE pending_turns SET flushed = 1 WHERE id IN ({placeholders})",
                ids,
            )
            await db.commit()
    except Exception as exc:
        logger.error("[turn_store] mark_flushed_by_ids failed ids=%r: %s", ids, exc)


# ── pending_profile CRUD ──────────────────────────────────────────────────────

async def save_pending_profile(
    user_id: str,
    category: str,
    name: str,
    attributes: dict,
    confidence: float,
    source: str,
) -> None:
    """保存中等置信度的结构化事实，等待二次提及后升格写图。"""
    import json as _json
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            # 若同名实体已存在则只更新 mention_count 和 confidence（取较大值）
            await db.execute(
                """
                INSERT INTO pending_profile
                    (user_id, category, name, attributes, confidence, source, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT DO NOTHING
                """,
                (user_id, category, name, _json.dumps(attributes, ensure_ascii=False),
                 confidence, source, time.time()),
            )
            await db.commit()
    except Exception as exc:
        logger.error("[turn_store] save_pending_profile failed user=%r name=%r: %s",
                     user_id, name, exc)


async def get_pending_profile(user_id: str) -> list[dict]:
    """返回该用户所有待升格的结构化事实。"""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT id, category, name, attributes, confidence, source, mention_count "
                "FROM pending_profile WHERE user_id = ? ORDER BY created_at",
                (user_id,),
            ) as cur:
                rows = await cur.fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.error("[turn_store] get_pending_profile failed user=%r: %s", user_id, exc)
        return []


async def delete_pending_profiles(ids: list[int]) -> None:
    """升格完成后删除对应 pending 行。"""
    if not ids:
        return
    try:
        placeholders = ",".join("?" * len(ids))
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                f"DELETE FROM pending_profile WHERE id IN ({placeholders})", ids
            )
            await db.commit()
    except Exception as exc:
        logger.error("[turn_store] delete_pending_profiles failed ids=%r: %s", ids, exc)


async def increment_pending_mentions(ids: list[int]) -> None:
    """mention_count +1，用于累计提及计数。"""
    if not ids:
        return
    try:
        placeholders = ",".join("?" * len(ids))
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                f"UPDATE pending_profile SET mention_count = mention_count + 1 "
                f"WHERE id IN ({placeholders})",
                ids,
            )
            await db.commit()
    except Exception as exc:
        logger.error("[turn_store] increment_pending_mentions failed ids=%r: %s", ids, exc)
