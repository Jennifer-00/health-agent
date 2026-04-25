"""
chat.py — Transport 层（HTTP/SSE）。

职责边界：
  - 解析 HTTP 请求、读写 session、返回 StreamingResponse
  - 不包含任何业务逻辑（triage / agent / critic / 记忆写入均在 harness 层）

AgentHarness 在模块加载时初始化一次，on_stop_hooks 注册 session 持久化。
"""
import asyncio
import json
import os

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage

from agent.harness import AgentHarness, _background_tasks
from agent.intake_graph import build_intake_graph, history_to_lc
from backend.schemas.chat import ChatRequest
from memory.mem0_client import Mem0Client
from memory.session_buffer import SessionBuffer

router = APIRouter(prefix="/chat", tags=["chat"])


async def _warmup_task(user_id: str) -> None:
    """后台预热：拉取 Mem0 记忆写入 Redis prefetch 缓存，消除冷启动延迟。"""
    try:
        memories = await Mem0Client(user_id=user_id).search("健康 症状 用药 记录", limit=20)
        if memories:
            await SessionBuffer.set_mem_prefetch(user_id, memories)
            print(f"[warmup] done user={user_id!r} count={len(memories)}")
        else:
            print(f"[warmup] skipped user={user_id!r} (empty)")
    except Exception as exc:
        print(f"[warmup] failed user={user_id!r}: {exc}")


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
            yield _encode_sse({"type": "mode", "mode": "consult"})
            async for event in _handle_consult(
                is_new_consult, consult_state, req.message,
                user_id, session, session_id,
            ):
                yield event
            return

        history = _trim_history(await session.get())
        async for chunk in await _harness.run(
            user_message=req.message,
            user_id=user_id,
            session_id=session_id,
            history=history,
        ):
            yield chunk

    return StreamingResponse(event_generator(), media_type="text/event-stream")


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
        history.append({"role": "assistant", "content": reply_text})
        await session.set_consult({"active": True, "history": history})
        for chunk in _stream_text(reply_text):
            yield chunk
        yield _encode_sse({"type": "done"})
