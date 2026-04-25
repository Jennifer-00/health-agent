from fastapi import APIRouter, HTTPException, Query, Request
from typing import List

from backend.schemas.chat import MemoryItem, MemoryPatchRequest
from memory.mem0_client import Mem0Client

router = APIRouter(prefix="/memory", tags=["memory"])


@router.get("", response_model=List[MemoryItem])
async def list_memories(request: Request):
    """返回用户全部 Mem0 健康记忆。"""
    user_id: str = request.state.user_id

    client = Mem0Client(user_id=user_id)
    memories = await client.get_all()

    return [
        MemoryItem(
            id=m["id"],
            content=m["memory"],
            category=(m.get("metadata") or {}).get("category"),
            record_date=None,
            source="mem0",
        )
        for m in memories
        if m and m.get("memory")
    ]


@router.delete("/{memory_id}")
async def delete_memory(memory_id: str, request: Request):
    """删除指定 Mem0 记忆条目。"""
    user_id: str = request.state.user_id
    client = Mem0Client(user_id=user_id)
    await client.delete(memory_id)
    return {"ok": True}


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
