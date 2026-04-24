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

from agent.critic import critic_review
from agent.graph import build_graph
from agent.observability import log_llm_call, log_pipeline_event, new_trace_id
from agent.triage import triage
from memory.mem0_client import Mem0Client
from memory.session_buffer import SessionBuffer

# ── Hook 类型别名 ──────────────────────────────────────────────────────────────
PreLLMHook   = Callable[[str, str, str], None]
PostToolHook = Callable[[str, list[dict]], None]
OnStopHook   = Callable[[str, str, str, str, str], Awaitable[None]]


async def _memory_write_hook(
    trace_id: str,
    user_content: str,
    assistant_text: str,
    user_id: str,
    session_id: str,
) -> None:
    """写入 Mem0，完成后主动刷新 Redis prefetch 缓存。"""
    try:
        client = Mem0Client(user_id=user_id)
        await client.add(user_content)
        fresh = await client.search("健康 症状 用药 记录", limit=20)
        if fresh:
            await SessionBuffer.set_mem_prefetch(user_id, fresh)
        log_pipeline_event(trace_id, "memory_write_done", user_id=user_id)
    except Exception as exc:
        logger.error("[memory_write_hook] failed user=%r: %s", user_id, exc)


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
        self.on_stop_hooks:   list[OnStopHook]    = [_memory_write_hook]
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
        trace_id = new_trace_id()
        log_pipeline_event(trace_id, "pipeline_start", user_id=user_id, session_id=session_id)

        # ── L1: Triage Gate ───────────────────────────────────────────────────
        log_pipeline_event(trace_id, "triage_start")
        is_emergency, emergency_response = await triage(user_message)
        log_pipeline_event(trace_id, "triage_done", emergency=is_emergency)

        if is_emergency:
            yield _sse({"type": "emergency", "text": emergency_response})
            yield _sse({"type": "done"})
            log_pipeline_event(trace_id, "pipeline_end", exit="triage_short_circuit")
            save_content = original_user_content or user_message
            for hook in self.on_stop_hooks:
                _make_task(_safe_hook(hook, trace_id, save_content, emergency_response, user_id, session_id), "on_stop_hook")
            return

        # ── pre_llm_hooks ─────────────────────────────────────────────────────
        for hook in self.pre_llm_hooks:
            try:
                hook(trace_id, user_message, user_id)
            except Exception as exc:
                log_pipeline_event(trace_id, "hook_error", hook="pre_llm", error=str(exc))

        # ── L2: Main Agent Graph ──────────────────────────────────────────────
        log_pipeline_event(trace_id, "agent_start")
        graph = self._get_graph()
        conversation = history + [{"role": "user", "content": user_message}]
        assistant_text = ""
        tool_events_from_graph: list[dict] = []
        _last_usage: dict = {}

        async for event in graph.astream_events(
            {"messages": conversation, "user_id": user_id, "tool_events": []},
            version="v2",
        ):
            kind = event["event"]

            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                # 跳过工具调用轮次，只流出纯文本 token
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
                        assistant_text += token
                        yield _sse({"type": "text", "delta": token})

            elif kind == "on_tool_start":
                tool_name = event["name"]
                args = event["data"].get("input", {})
                query = args.get("query", "") if isinstance(args, dict) else ""
                summary = f"查询：{query}" if query else tool_name
                tool_events_from_graph.append({"tool": tool_name, "summary": summary})
                yield _sse({"type": "tool_call", "tool": tool_name, "summary": summary})

            elif kind == "on_chat_model_end":
                output = event["data"].get("output")
                meta = getattr(output, "usage_metadata", None) or {}
                if meta:
                    _last_usage = meta

        log_llm_call(
            trace_id=trace_id,
            node="orchestrator",
            tokens_input=_last_usage.get("input_tokens", 0),
            tokens_output=_last_usage.get("output_tokens", 0),
            ms=0,
            tools_requested=[e["tool"] for e in tool_events_from_graph],
        )
        log_pipeline_event(trace_id, "agent_done", reply_len=len(assistant_text))

        # ── L3: Critic Review ─────────────────────────────────────────────────
        log_pipeline_event(trace_id, "critic_start")
        note = await critic_review(user_message, assistant_text)
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
