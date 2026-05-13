"""
AgentHarness — Pipeline 编排层。

将 Triage → Main Agent → Critic 三层显式串联，
并提供 Hook 注册点供横切关注点（日志、记忆写入等）接入。

Pipeline 流程：
  用户消息
      │
      ▼
  [L1] Triage Gate      ← input guard，Haiku 快速判定，紧急则短路返回
      │ 通过
      ▼
  [L2] Main Agent Graph ← LangGraph Runtime，工具按需调用（search_memory / search_rag / web_search / generate_report）
      │ 生成回复
      ▼
  [L3] Critic Review    ← output guard，质检回复合规性
      │
      ▼
  on_stop_hooks         ← fire-and-forget：session 持久化、Mem0 写入 + 缓存刷新
      │
      ▼
  最终回复（AsyncGenerator[str, None] → SSE chunks）
"""
import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)

_background_tasks: set[asyncio.Task] = set()

import os

from agent.critic import soften_absolutes, grounding_check, update_badcase
from agent.graph import build_graph
from agent.intent import classify_intent
from agent.observability import log_llm_call, log_pipeline_event, new_trace_id
from agent.triage import triage
from memory.mem0_client import Mem0Client
from memory.session_buffer import SessionBuffer
from memory import turn_store

__all__ = ["AgentHarness", "_background_tasks", "_flush_pending_to_mem0"]

MEM_WRITE_EVERY_N_TURNS = int(os.getenv("MEM_WRITE_EVERY_N_TURNS", "5"))

# ── Hook 类型别名 ──────────────────────────────────────────────────────────────
PreLLMHook   = Callable[[str, str, str], None]
PostToolHook = Callable[[str, list[dict]], None]
OnStopHook   = Callable[[str, str, str, str, str], Awaitable[None]]


async def _compliance_hook(
    trace_id: str,
    user_content: str,
    assistant_text: str,
    user_id: str,
    session_id: str,
) -> None:
    """合规审查 fire-and-forget：不合规写入 critic_failures.jsonl，下次请求注入反例。"""
    await update_badcase(user_content, assistant_text)


async def _memory_write_hook(
    trace_id: str,
    user_content: str,
    assistant_text: str,
    user_id: str,
    session_id: str,
) -> None:
    """累积 N 轮后两路并行写入：档案类 → Neo4j，情景类 → Qdrant。"""
    try:
        session = SessionBuffer(session_id=session_id)
        turn_count = await session.append_mem_pending(user_content, assistant_text)
        await turn_store.save_turn(user_id, session_id, user_content, assistant_text)
        log_pipeline_event(trace_id, "memory_pending_queued", user_id=user_id, turn=turn_count)

        if turn_count % MEM_WRITE_EVERY_N_TURNS == 0:
            await _flush_pending_to_mem0(session, session_id, user_id, trace_id, trigger="n_turns")
    except Exception as exc:
        logger.error("[memory_write_hook] failed user=%r: %s", user_id, exc)



async def _flush_pending_to_mem0(
    session: SessionBuffer,
    session_id: str,
    user_id: str,
    trace_id: str,
    trigger: str,
) -> None:
    """会话结束时调用：两路并行写入剩余未刷轮次。
    - mem0 路：情景类 facts → Qdrant
    - Neo4j 路：档案类 facts → Neo4j（兜底，补充每轮写入可能遗漏的信息）
    """
    turns = await session.drain_mem_pending()
    if not turns:
        return
    combined = "\n\n".join(
        f"用户: {t['user']}\n助手: {t['assistant']}" for t in turns
    )

    client = Mem0Client(user_id=user_id)
    from memory.mem0_client import extract_profile_facts
    from memory import profile_graph

    mem0_result, profile_facts = await asyncio.gather(
        client.add(combined),
        extract_profile_facts(combined),
        return_exceptions=True,
    )

    if isinstance(profile_facts, list) and profile_facts:
        try:
            await profile_graph.promote_pending(user_id, profile_facts)
            await profile_graph.upsert_from_extraction(user_id, profile_facts)
        except Exception as exc:
            logger.warning("[harness] profile_graph write failed user=%r: %s", user_id, exc)

    await turn_store.mark_flushed_by_session(user_id, session_id)
    log_pipeline_event(trace_id, "memory_write_done", user_id=user_id, trigger=trigger, turns=len(turns))


def _make_task(coro, label: str) -> asyncio.Task:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)

    def _on_done(t: asyncio.Task) -> None:
        _background_tasks.discard(t)
        if t.cancelled():
            logger.warning("[%s] task 被取消，未完成", label)
            return
        exc = t.exception()
        if exc:
            logger.error("[%s] 执行失败", label, exc_info=exc)

    task.add_done_callback(_on_done)
    return task


class AgentHarness:
    def __init__(self) -> None:
        self.pre_llm_hooks:   list[PreLLMHook]   = []
        self.post_tool_hooks: list[PostToolHook]  = []
        self.on_stop_hooks:   list[OnStopHook]    = [_memory_write_hook, _compliance_hook]
        self._graph = None

    def _get_graph(self):
        if self._graph is None:
            self._graph = build_graph()
        return self._graph

    async def run(
        self,
        user_message: str,
        user_id: str,
        session_id: str,
        history: list[dict],
        original_user_content: str | None = None,
    ) -> AsyncGenerator[str, None]:
        return self._run_pipeline(
            user_message, user_id, session_id, history, original_user_content
        )

    async def _run_pipeline(
        self,
        user_message: str,
        user_id: str,
        session_id: str,
        history: list[dict],
        original_user_content: str | None,
    ) -> AsyncGenerator[str, None]:
        import time as _time
        t_start = _time.monotonic()
        trace_id = new_trace_id()
        log_pipeline_event(trace_id, "pipeline_start", user_id=user_id, session_id=session_id)

        # ── L1: Triage 与 L2: Agent 并发启动 ─────────────────────────────────
        # Triage 作为后台 task，Agent 同时开始推理；
        # Agent 产生的 SSE chunk 先缓冲，等 Triage 完成后再决定是推送还是丢弃。
        log_pipeline_event(trace_id, "triage_start")
        triage_task  = asyncio.create_task(triage(user_message))
        intent_task  = asyncio.create_task(classify_intent(user_message))

        # pre_llm_hooks 在 triage 结果确认前运行（均为快速同步操作）
        for hook in self.pre_llm_hooks:
            try:
                hook(trace_id, user_message, user_id)
            except Exception as exc:
                log_pipeline_event(trace_id, "hook_error", hook="pre_llm", error=str(exc))

        # intent 比 triage 快，先 await 拿到结果注入 graph state
        intent = await intent_task
        log_pipeline_event(trace_id, "intent_done", intent=intent)

        log_pipeline_event(trace_id, "agent_start")
        graph = self._get_graph()
        conversation = history + [{"role": "user", "content": user_message}]
        assistant_text = ""
        tool_events_from_graph: list[dict] = []
        _last_usage: dict = {}
        _web_sources: list[tuple[str, str]] = []

        _first_token    = True
        _triage_ok      = False   # triage 完成且非紧急
        _pre_buf: list[str] = []  # triage 完成前积累的 SSE chunk
        _tool_contents: dict = {}  # {tool_name: result_text} for grounding check

        async def _resolve_triage() -> tuple[bool, str]:
            """等待 triage 结果；紧急时直接 yield 并返回 False，否则 True。"""
            nonlocal _triage_ok
            is_emergency, emergency_response = await triage_task
            log_pipeline_event(trace_id, "triage_done", emergency=is_emergency)
            if is_emergency:
                return False, emergency_response
            _triage_ok = True
            return True, ""

        async for event in graph.astream_events(
            {"messages": conversation, "user_id": user_id, "trace_id": trace_id,
             "intent": intent, "tool_events": []},
            version="v2",
        ):
            # 每次循环检查 triage 是否已完成（done() 无阻塞）
            if not _triage_ok and triage_task.done():
                safe, emergency_response = await _resolve_triage()
                if not safe:
                    triage_task.cancel()
                    yield _sse({"type": "emergency", "text": emergency_response})
                    yield _sse({"type": "done"})
                    log_pipeline_event(trace_id, "pipeline_end", exit="triage_short_circuit")
                    save_content = original_user_content or user_message
                    for hook in self.on_stop_hooks:
                        _make_task(_safe_hook(hook, trace_id, save_content, emergency_response, user_id, session_id), "on_stop_hook")
                    return
                # triage 通过 — 先发 intent 标识，再推积累的 chunk
                yield _sse({"type": "intent", "intent": intent})
                for buffered in _pre_buf:
                    yield buffered
                _pre_buf.clear()

            kind = event["event"]

            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if not chunk.tool_call_chunks and chunk.content:
                    content = chunk.content
                    if isinstance(content, str):
                        token = content
                    elif isinstance(content, list):
                        token = "".join(
                            block.get("text", "")
                            for block in content
                            if isinstance(block, dict) and block.get("type") == "text"
                        )
                    else:
                        token = ""
                    if token:
                        token = soften_absolutes(token)
                        assistant_text += token
                        sse_chunk = _sse({"type": "text", "delta": token})
                        if _triage_ok:
                            if _first_token:
                                import time as _time
                                ttft_ms = int((_time.monotonic() - t_start) * 1000)
                                log_pipeline_event(trace_id, "first_token", ms=ttft_ms)
                                _first_token = False
                            yield sse_chunk
                        else:
                            _pre_buf.append(sse_chunk)

            elif kind == "on_chain_end":
                state_out = event["data"].get("output", {})
                if isinstance(state_out, dict):
                    if "tool_contents" in state_out:
                        _tool_contents = state_out["tool_contents"]
                    if "web_sources" in state_out:
                        _web_sources = state_out["web_sources"]

            elif kind == "on_chat_model_end":
                output = event["data"].get("output")
                tool_calls = getattr(output, "tool_calls", None) or []
                for tc in tool_calls:
                    tool_name = tc.get("name", "")
                    args = tc.get("args", {})
                    query = args.get("query", "") if isinstance(args, dict) else ""
                    summary = f"查询：{query}" if query else tool_name
                    tool_events_from_graph.append({"tool": tool_name, "summary": summary})
                    sse_chunk = _sse({"type": "tool_call", "tool": tool_name, "summary": summary})
                    if _triage_ok:
                        yield sse_chunk
                    else:
                        _pre_buf.append(sse_chunk)
                meta = getattr(output, "usage_metadata", None) or {}
                if meta:
                    _last_usage = meta

        # Agent 结束后 triage 仍未完成（极少见：agent 比 triage 更快时）
        if not _triage_ok:
            safe, emergency_response = await _resolve_triage()
            if not safe:
                yield _sse({"type": "emergency", "text": emergency_response})
                yield _sse({"type": "done"})
                log_pipeline_event(trace_id, "pipeline_end", exit="triage_short_circuit")
                save_content = original_user_content or user_message
                for hook in self.on_stop_hooks:
                    _make_task(_safe_hook(hook, trace_id, save_content, emergency_response, user_id, session_id), "on_stop_hook")
                return
            yield _sse({"type": "intent", "intent": intent})
            for buffered in _pre_buf:
                yield buffered
            _pre_buf.clear()

        log_llm_call(
            trace_id=trace_id,
            node="orchestrator",
            tokens_input=_last_usage.get("input_tokens", 0),
            tokens_output=_last_usage.get("output_tokens", 0),
            ms=int((_time.monotonic() - t_start) * 1000),
            tools_requested=[e["tool"] for e in tool_events_from_graph],
        )
        log_pipeline_event(trace_id, "agent_done", reply_len=len(assistant_text))

        # ── web_search 来源追加 ───────────────────────────────────────────────
        if _web_sources:
            seen: set[str] = set()
            items = []
            for title, url in _web_sources:
                if url not in seen:
                    seen.add(url)
                    items.append(f"- [{title}]({url})")
            if items:
                sources_delta = "\n\n---\n\n**参考来源**\n" + "\n".join(items)
                assistant_text += sources_delta
                yield _sse({"type": "text", "delta": sources_delta})

        # ── 工具结果预览（供前端可视化）────────────────────────────────────────
        for tool_name, result_text in _tool_contents.items():
            if isinstance(result_text, str) and result_text:
                preview = result_text[:400] + ("…" if len(result_text) > 400 else "")
                yield _sse({"type": "tool_result", "tool": tool_name, "preview": preview})

        # ── L3: Grounding Check（数据引用溯源，SSE done 前拦截）────────────────
        log_pipeline_event(trace_id, "critic_start")
        note = await grounding_check(assistant_text, _tool_contents)
        log_pipeline_event(trace_id, "critic_done", passed=(note is None))

        if note:
            assistant_text = f"{assistant_text}\n\n{note}"
            yield _sse({"type": "text", "delta": f"\n\n{note}"})

        yield _sse({"type": "done"})

        # ── on_stop_hooks ─────────────────────────────────────────────────────
        save_content = original_user_content or user_message
        for hook in self.on_stop_hooks:
            _make_task(_safe_hook(hook, trace_id, save_content, assistant_text, user_id, session_id), "on_stop_hook")

        log_pipeline_event(trace_id, "pipeline_end", exit="normal")


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _sse(payload: dict) -> str:
    import json
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _safe_hook(hook: OnStopHook, *args) -> None:
    try:
        await hook(*args)
    except Exception as exc:
        logger.error("[safe_hook] error: %s", exc)
