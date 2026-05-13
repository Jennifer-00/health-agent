import logging
import re
from fastapi import APIRouter, HTTPException, Query, Request
from typing import List

from backend.schemas.chat import MemoryItem, MemoryPatchRequest, MemoryImportRequest
from memory.mem0_client import Mem0Client

logger = logging.getLogger(__name__)

_TAG_RE = re.compile(r"^\[([^\]]+)\]\s*")

def _parse_memory(raw: str) -> tuple[str, str | None]:
    """把 '[症状] 头痛三天' 拆成 ('头痛三天', '症状')，无标签则 category=None。"""
    m = _TAG_RE.match(raw)
    if m:
        return raw[m.end():], m.group(1)
    return raw, None

router = APIRouter(prefix="/memory", tags=["memory"])


@router.get("", response_model=List[MemoryItem])
async def list_memories(request: Request):
    """返回用户全部 Mem0 健康记忆。"""
    user_id: str = request.state.user_id

    client = Mem0Client(user_id=user_id)
    memories = await client.get_all()

    items = []
    for m in memories:
        if not m or not m.get("memory"):
            continue
        content, category = _parse_memory(m["memory"])
        # metadata 里有明确分类时优先用，否则用提取 prompt 里的标签
        if not category:
            category = (m.get("metadata") or {}).get("category")
        items.append(MemoryItem(
            id=m["id"],
            content=content,
            category=category,
            record_date=None,
            source="mem0",
        ))
    return items


@router.delete("/{memory_id}")
async def delete_memory(memory_id: str, request: Request):
    """删除指定 Mem0 记忆条目。"""
    user_id: str = request.state.user_id
    client = Mem0Client(user_id=user_id)
    await client.delete(memory_id)
    return {"ok": True}


@router.post("/import")
async def import_memory(body: MemoryImportRequest, request: Request):
    """将一段文字写入记忆库（两路后台执行，立即返回）。"""
    import asyncio
    user_id: str = request.state.user_id
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text is empty")

    async def _run():
        from memory.mem0_client import extract_profile_facts
        from memory import profile_graph
        client = Mem0Client(user_id=user_id)
        mem0_res, profile_facts = await asyncio.gather(
            client.add(text),
            extract_profile_facts(text),
            return_exceptions=True,
        )
        if isinstance(profile_facts, list) and profile_facts:
            try:
                await profile_graph.promote_pending(user_id, profile_facts)
                await profile_graph.upsert_from_extraction(user_id, profile_facts)
            except Exception as exc:
                logger.warning("[import] profile_graph write failed user=%r: %s", user_id, exc)

    asyncio.create_task(_run())
    return {"ok": True}


@router.get("/profile")
async def get_profile(request: Request):
    """返回 Neo4j 结构化健康档案（Redis 缓存优先）。"""
    user_id: str = request.state.user_id
    try:
        from memory.profile_graph import get_profile_rows, _REL_TO_LABEL
        rows = await get_profile_rows(user_id)
        return [
            {
                "category": _REL_TO_LABEL.get(row["rel"], row["rel"]),
                "name": row["name"],
                "detail": row.get("detail") or "",
                "confidence": round(float(row["confidence"]), 2),
                "confirmed": row.get("src") == "doctor_confirmed",
            }
            for row in rows
        ]
    except Exception as exc:
        logger.warning("[memory/profile] fetch failed user=%r: %s", user_id, exc)
        return []


@router.patch("/{memory_id}")
async def patch_memory(memory_id: str, body: MemoryPatchRequest, request: Request):
    """手动修正某条 Mem0 记忆。"""
    user_id: str = request.state.user_id
    client = Mem0Client(user_id=user_id)

    if body.action == "delete":
        await client.delete(memory_id)
    elif body.action == "update":
        if not body.content:
            raise HTTPException(status_code=422, detail="update action requires content")
        await client.update(memory_id, body.content)
    else:
        raise HTTPException(status_code=422, detail=f"unknown action: {body.action!r}")

    return {"ok": True}
