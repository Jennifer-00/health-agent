from fastapi import APIRouter, HTTPException, Query, Request
from typing import List

from backend.schemas.chat import MemoryItem, MemoryPatchRequest
from memory.mem0_client import Mem0Client
from agent.skills.summary_gen import summary_gen
from agent.skills.memory_consolidate import _consolidate_mem0

router = APIRouter(prefix="/memory", tags=["memory"])


@router.post("/consolidate")
async def consolidate_memories(request: Request):
    """手动触发记忆整理合并，去除 Mem0 中重复条目。"""
    user_id: str = request.state.user_id
    result = await _consolidate_mem0(user_id)
    return {"ok": True, "result": result}


@router.get("/summary")
async def get_summary(request: Request):
    """直接调用 summary_gen 逻辑，返回结构化健康摘要文本。"""
    user_id: str = request.state.user_id
    text = await summary_gen.ainvoke({"user_id": user_id})
    return {"user_id": user_id, "summary": text}


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
            category=m.get("metadata", {}).get("category"),
            record_date=None,
            source="mem0",
        )
        for m in memories
        if m.get("memory")
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
    elif body.action == "update" and body.content:
        await client.update(memory_id, body.content)

    return {"ok": True}
