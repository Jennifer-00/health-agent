"""
chat.py — Transport 层（HTTP/SSE）。

职责边界：
  - 解析 HTTP 请求、读写 session、返回 StreamingResponse
  - 不包含任何业务逻辑（triage / agent / critic / 记忆写入均在 harness 层）

AgentHarness 在模块加载时初始化一次，on_stop_hooks 注册 session 持久化。
"""
import asyncio
import json
import logging
import os

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from agent.harness import AgentHarness, _background_tasks, _flush_pending_to_mem0
from agent.intake_graph import build_intake_graph, history_to_lc
from agent.observability import new_trace_id
from backend.schemas.chat import ChatRequest
from memory.mem0_client import Mem0Client
from memory.session_buffer import SessionBuffer
from memory import turn_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


async def _warmup_task(user_id: str) -> None:
    """热启动：预取健康档案写入 Redis + 恢复未写入轮次。"""
    try:
        from memory import profile_graph

        # 健康档案预取：Neo4j → Redis，前端打开时直接命中缓存
        await profile_graph.get_profile_rows(user_id)
        logger.info("[warmup] profile cached user=%r", user_id)

        # 崩溃恢复：补写上次未 flush 的轮次
        unflushed = await turn_store.get_unflushed(user_id)
        if not unflushed:
            return

        combined = "\n\n".join(
            f"用户: {r['user_content']}\n助手: {r['asst_content']}" for r in unflushed
        )
        from memory.mem0_client import extract_profile_facts

        async def _write_profile():
            facts = await extract_profile_facts(combined)
            if facts:
                await profile_graph.promote_pending(user_id, facts)
                await profile_graph.upsert_from_extraction(user_id, facts)

        await asyncio.gather(
            Mem0Client(user_id=user_id).add(combined),
            _write_profile(),
            return_exceptions=True,
        )
        await turn_store.mark_flushed_by_ids([r["id"] for r in unflushed])
        logger.info("[warmup] recovered %d unflushed turns user=%r", len(unflushed), user_id)
    except Exception as exc:
        logger.error("[warmup] failed user=%r: %s", user_id, exc)


MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "10"))
CHUNK_SIZE = 24

# ── Harness 单例 + on_stop_hooks 注册 ────────────────────────────────────────
_harness = AgentHarness()


async def _session_hook(
    trace_id: str,
    user_content: str,
    assistant_text: str,
    user_id: str,
    session_id: str,
) -> None:
    """on_stop_hook：会话历史持久化到 Redis。"""
    session = SessionBuffer(session_id=session_id)
    history = _trim_history(await session.get())
    await session.set(
        _trim_history(history + [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": assistant_text},
        ])
    )


_harness.on_stop_hooks.append(_session_hook)


# ── Transport 工具函数 ────────────────────────────────────────────────────────

def _trim_history(messages: list[dict], max_messages: int = MAX_HISTORY_MESSAGES) -> list[dict]:
    return messages[-max_messages:]


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item if isinstance(item, str)
            else item.get("text", "") if isinstance(item, dict) and item.get("type") == "text"
            else ""
            for item in content
        )
    return str(content)


def _encode_sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _stream_text(text: str):
    for start in range(0, len(text), CHUNK_SIZE):
        yield _encode_sse({"type": "text", "delta": text[start:start + CHUNK_SIZE]})


# ── HTTP 端点 ─────────────────────────────────────────────────────────────────

@router.post("/warmup")
async def warmup_endpoint(request: Request) -> dict:
    """页面加载时调用，后台预拉取用户记忆写入 Redis，消除首条消息的冷启动延迟。"""
    user_id: str = request.state.user_id
    task = asyncio.create_task(_warmup_task(user_id))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return {"status": "warming up"}


class _EndSessionRequest(BaseModel):
    session_id: str


@router.post("/end")
async def end_session_endpoint(req: _EndSessionRequest, request: Request) -> dict:
    """关闭对话时调用，将剩余未写入的轮次批量写入 Mem0。fire-and-forget，立即返回。"""
    user_id: str = request.state.user_id
    session = SessionBuffer(session_id=req.session_id)
    trace_id = new_trace_id()

    async def _flush():
        try:
            await _flush_pending_to_mem0(session, req.session_id, user_id, trace_id, trigger="session_end")
        except Exception as exc:
            logger.error("[end_session] flush failed user=%r: %s", user_id, exc)

    task = asyncio.create_task(_flush())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return {"status": "ok"}


@router.post("")
async def chat_endpoint(req: ChatRequest, request: Request):
    """读取会话历史，委托给 AgentHarness，以 SSE 实时推送进度和回复。"""
    user_id: str = request.state.user_id
    session_id = req.session_id or user_id
    session = SessionBuffer(session_id=session_id)

    async def event_generator():
        is_new_consult = req.message.strip().lower() == "/consult"
        consult_state  = await session.get_consult()
        in_consult     = consult_state.get("active", False)

        if is_new_consult or in_consult:
            logger.info("[chat] consult mode user=%r session=%r new=%s", user_id, session_id, is_new_consult)
            yield _encode_sse({"type": "mode", "mode": "consult"})
            async for event in _handle_consult(
                is_new_consult, consult_state, req.message,
                user_id, session, session_id,
            ):
                yield event
            return

        logger.info("[chat] normal mode user=%r session=%r msg_len=%d", user_id, session_id, len(req.message))
        history = _trim_history(await session.get())
        async for chunk in await _harness.run(
            user_message=req.message,
            user_id=user_id,
            session_id=session_id,
            history=history,
        ):
            yield chunk

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


# ── 问诊流程 ──────────────────────────────────────────────────────────────────

async def _handle_consult(
    is_new_consult: bool,
    consult_state: dict,
    user_message: str,
    user_id: str,
    session: SessionBuffer,
    session_id: str,
):
    intake_graph = build_intake_graph()
    history: list[dict] = consult_state.get("history", [])

    if is_new_consult:
        history = [{"role": "user", "content": "开始问诊"}]
    else:
        history.append({"role": "user", "content": user_message})

    result = await intake_graph.ainvoke({
        "messages": history_to_lc(history),
        "done": False,
        "summary": "",
    })

    last_ai    = next((m for m in reversed(result["messages"]) if isinstance(m, AIMessage)), None)
    reply_text = _extract_text(last_ai.content) if last_ai else ""

    if result.get("done"):
        logger.info("[chat] consult done user=%r session=%r summary_len=%d", user_id, session_id, len(result.get("summary", "")))
        for chunk in _stream_text(reply_text):
            yield chunk
        await session.clear_consult()

        summary = result.get("summary", "")
        yield _encode_sse({"type": "mode",   "mode": "chat"})
        yield _encode_sse({"type": "status", "text": "正在分析症状…"})

        main_history = _trim_history(await session.get())
        async for chunk in await _harness.run(
            user_message=f"[问诊摘要] {summary}",
            user_id=user_id,
            session_id=session_id,
            history=main_history,
            original_user_content=summary,
        ):
            yield chunk
    else:
        logger.info("[chat] consult ongoing user=%r session=%r turns=%d", user_id, session_id, len(history))
        history.append({"role": "assistant", "content": reply_text})
        await session.set_consult({"active": True, "history": history})
        for chunk in _stream_text(reply_text):
            yield chunk
        yield _encode_sse({"type": "done"})
