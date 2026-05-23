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
  [L2] Main Agent Graph ← LangGraph Runtime，工具按需调用
      │ 生成回复
      ▼
  [L3] Critic Review    ← output guard，质检回复合规性
      │
      ▼
  on_stop_hooks         ← fire-and-forget：session 持久化、Mem0 写入 + 缓存刷新
      │
      ▼
  最终回复（AsyncGenerator[str, None] → SSE chunks）

可观测性:
  - 每个 pipeline 阶段是一个 OpenTelemetry Span，形成嵌套 Span Tree
  - 每个阶段结束时写入 Prometheus 指标（duration, tokens, counter）
  - fire-and-forget hooks 通过 W3C TraceContext link 回主 trace

面试要点:
  - Agent 系统的 Trace 必须是 Span Tree，不是扁平事件流
  - 火焰图直接暴露瓶颈在哪个阶段（LLM vs Tool vs Network）
  - TTFT 是流式 Agent 的核心体验指标
  - Critic 拦截率是输出质量的先行指标
"""
import asyncio
import logging
import time as _time
import os
from collections.abc import AsyncGenerator
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)

_background_tasks: set[asyncio.Task] = set()

from agent.critic import soften_absolutes, grounding_check, update_badcase
from agent.graph import build_graph
from agent.intent import classify_intent
from agent.observability import new_trace_id
from agent.telemetry import get_tracer, current_span_context, span_from_context, mark_span_ok, mark_span_error
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


# ── Metrics import (graceful fallback) ─────────────────────────────────────────

def _record_metric(fn, *args, **kwargs):
    """安全调用 prometheus 指标，import 失败或记录失败时静默跳过。"""
    try:
        fn(*args, **kwargs)
    except Exception:
        pass


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
    from agent.observability import log_pipeline_event
    try:
        session = SessionBuffer(session_id=session_id)
        turn_count = await session.append_mem_pending(user_content, assistant_text)
        await turn_store.save_turn(user_id, session_id, user_content, assistant_text)
        log_pipeline_event(trace_id, "memory_pending_queued", user_id=user_id, turn=turn_count)

        if turn_count % MEM_WRITE_EVERY_N_TURNS == 0:
            t0 = _time.monotonic()
            await _flush_pending_to_mem0(session, session_id, user_id, trace_id, trigger="n_turns")
            _record_metric(
                _import_metric("memory_write_latency").labels(trigger="n_turns").observe,
                _time.monotonic() - t0,
            )
    except Exception as exc:
        logger.error("[memory_write_hook] failed user=%r: %s", user_id, exc)


def _import_metric(name: str):
    """惰性导入单个 metric，避免模块级循环依赖。"""
    import importlib
    mod = importlib.import_module("agent.metrics")
    return getattr(mod, name)


async def _flush_pending_to_mem0(
    session: SessionBuffer,
    session_id: str,
    user_id: str,
    trace_id: str,
    trigger: str,
) -> None:
    from agent.observability import log_pipeline_event
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
        t_start = _time.monotonic()
        trace_id = new_trace_id()

        # ── 注入 request 上下文到日志 ────────────────────────────────────────
        try:
            from agent.logging_config import bind_context, unbind_context
            bind_context(trace_id=trace_id, user_id=user_id, session_id=session_id)
        except Exception:
            pass

        # ── 创建 root Span ────────────────────────────────────────────────────
        tracer = get_tracer()
        root = tracer.start_span("pipeline")
        root.set_attributes({
            "user.id": user_id,
            "session.id": session_id,
            "trace.id": trace_id,
            "pipeline.entry": "/chat",
        })

        exit_reason = "normal"
        intent_label = "general"

        try:
            # ── 上报 pipeline 开始 ────────────────────────────────────────────
            from agent.observability import log_pipeline_event
            log_pipeline_event(trace_id, "pipeline_start", user_id=user_id, session_id=session_id)

            # ── L1: Triage 与 L2: Agent 并发启动 ─────────────────────────────
            log_pipeline_event(trace_id, "triage_start")
            triage_task = asyncio.create_task(triage(user_message))
            intent_task = asyncio.create_task(classify_intent(user_message))

            for hook in self.pre_llm_hooks:
                try:
                    hook(trace_id, user_message, user_id)
                except Exception as exc:
                    log_pipeline_event(trace_id, "hook_error", hook="pre_llm", error=str(exc))

            # intent 比 triage 快，先 await 拿到结果注入 graph state
            intent = await intent_task
            intent_label = intent
            log_pipeline_event(trace_id, "intent_done", intent=intent)

            # ── L2: Agent Graph Span ──────────────────────────────────────────
            agent_span = tracer.start_span("agent")
            agent_span.set_attributes({
                "agent.intent": intent,
                "user.id": user_id,
                "session.id": session_id,
            })

            log_pipeline_event(trace_id, "agent_start")
            graph = self._get_graph()
            conversation = history + [{"role": "user", "content": user_message}]
            assistant_text = ""
            tool_events_from_graph: list[dict] = []
            _last_usage: dict = {}
            _web_sources: list[tuple[str, str]] = []
            _tool_contents: dict = {}

            _first_token    = True
            _triage_ok      = False
            _pre_buf: list[str] = []
            _tool_round_count = 0

            async def _resolve_triage() -> tuple[bool, str]:
                nonlocal _triage_ok
                is_emergency, emergency_response = await triage_task
                log_pipeline_event(trace_id, "triage_done", emergency=is_emergency)
                if is_emergency:
                    _record_metric(
                        _import_metric("triage_emergency").labels(intent=intent_label).inc
                    )
                return (not is_emergency), emergency_response

            async for event in graph.astream_events(
                {"messages": conversation, "user_id": user_id, "trace_id": trace_id,
                 "intent": intent, "tool_events": []},
                version="v2",
            ):
                if not _triage_ok and triage_task.done():
                    safe, emergency_response = await _resolve_triage()
                    if not safe:
                        exit_reason = "triage_short_circuit"
                        triage_task.cancel()
                        agent_span.set_attribute("agent.exit", "preempted_by_triage")
                        mark_span_ok(agent_span)
                        agent_span.end()
                        yield _sse({"type": "emergency", "text": emergency_response})
                        yield _sse({"type": "done"})
                        log_pipeline_event(trace_id, "pipeline_end", exit=exit_reason)
                        save_content = original_user_content or user_message
                        for hook in self.on_stop_hooks:
                            ctx = current_span_context()
                            _make_task(
                                _safe_hook_with_trace(hook, ctx, trace_id, save_content, emergency_response, user_id, session_id),
                                "on_stop_hook",
                            )
                        return
                    _triage_ok = True
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
                                    ttft_ms = int((_time.monotonic() - t_start) * 1000)
                                    log_pipeline_event(trace_id, "first_token", ms=ttft_ms)
                                    _record_metric(
                                        _import_metric("ttft").labels(intent=intent_label).observe,
                                        ttft_ms / 1000.0,
                                    )
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
                        if "tool_events" in state_out:
                            _tool_round_count = len(state_out["tool_events"])

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

            # Agent 结束后 triage 仍未完成（极少见）
            if not _triage_ok:
                safe, emergency_response = await _resolve_triage()
                if not safe:
                    exit_reason = "triage_short_circuit"
                    agent_span.set_attribute("agent.exit", "preempted_by_triage")
                    agent_span.end()
                    yield _sse({"type": "emergency", "text": emergency_response})
                    yield _sse({"type": "done"})
                    log_pipeline_event(trace_id, "pipeline_end", exit=exit_reason)
                    save_content = original_user_content or user_message
                    for hook in self.on_stop_hooks:
                        ctx = current_span_context()
                        _make_task(
                            _safe_hook_with_trace(hook, ctx, trace_id, save_content, emergency_response, user_id, session_id),
                            "on_stop_hook",
                        )
                    return
                _triage_ok = True
                yield _sse({"type": "intent", "intent": intent})
                for buffered in _pre_buf:
                    yield buffered
                _pre_buf.clear()

            # ── Agent 结束: 记录 LLM 指标 ────────────────────────────────────
            input_tokens = _last_usage.get("input_tokens", 0)
            output_tokens = _last_usage.get("output_tokens", 0)
            tools_list = [e["tool"] for e in tool_events_from_graph]
            logger.info(
                "[harness] agent_done trace=%s tokens_in=%d tokens_out=%d tools=%s rounds=%d",
                trace_id, input_tokens, output_tokens, tools_list, _tool_round_count,
                extra={"node": "orchestrator", "tokens_input": input_tokens,
                       "tokens_output": output_tokens, "tools": tools_list},
            )

            # 写入 Prometheus: LLM token + 工具轮次
            _record_metric(
                _import_metric("llm_calls").labels(node="orchestrator", model="gpt-4o").inc
            )
            _record_metric(
                _import_metric("llm_tokens").labels(node="orchestrator", model="gpt-4o", direction="input").inc,
                input_tokens,
            )
            _record_metric(
                _import_metric("llm_tokens").labels(node="orchestrator", model="gpt-4o", direction="output").inc,
                output_tokens,
            )
            _record_metric(
                _import_metric("tool_loop_rounds").observe, _tool_round_count
            )

            from agent.observability import log_llm_call
            log_llm_call(
                trace_id=trace_id,
                node="orchestrator",
                tokens_input=input_tokens,
                tokens_output=output_tokens,
                ms=int((_time.monotonic() - t_start) * 1000),
                tools_requested=tools_list,
            )
            log_pipeline_event(trace_id, "agent_done", reply_len=len(assistant_text))
            agent_span.set_attributes({
                "agent.reply_len": len(assistant_text),
                "agent.tool_rounds": _tool_round_count,
                "agent.tokens_input": input_tokens,
                "agent.tokens_output": output_tokens,
            })
            mark_span_ok(agent_span)
            agent_span.end()

            # ── web_search 来源追加 ──────────────────────────────────────────
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

            # ── 工具结果预览 ─────────────────────────────────────────────────
            for tool_name, result_text in _tool_contents.items():
                if isinstance(result_text, str) and result_text:
                    preview = result_text[:400] + ("…" if len(result_text) > 400 else "")
                    yield _sse({"type": "tool_result", "tool": tool_name, "preview": preview})

            # ── L3: Critic Span ──────────────────────────────────────────────
            critic_span = tracer.start_span("critic")
            critic_span.set_attributes({"critic.assistant_len": len(assistant_text)})

            log_pipeline_event(trace_id, "critic_start")
            note = await grounding_check(assistant_text, _tool_contents)
            critic_passed = note is None
            log_pipeline_event(trace_id, "critic_done", passed=critic_passed)

            # 写入 Prometheus: critic 指标
            _record_metric(
                _import_metric("critic_checks").labels(type="grounding").inc
            )
            if not critic_passed:
                _record_metric(
                    _import_metric("critic_failures").labels(type="grounding").inc
                )

            critic_span.set_attributes({"critic.passed": critic_passed})
            mark_span_ok(critic_span)
            critic_span.end()

            if note:
                assistant_text = f"{assistant_text}\n\n{note}"
                yield _sse({"type": "text", "delta": f"\n\n{note}"})

            yield _sse({"type": "done"})

            # ── on_stop_hooks (fire-and-forget, 携带 trace context) ──────────
            save_content = original_user_content or user_message
            ctx = current_span_context()
            for hook in self.on_stop_hooks:
                _make_task(
                    _safe_hook_with_trace(hook, ctx, trace_id, save_content, assistant_text, user_id, session_id),
                    "on_stop_hook",
                )

            log_pipeline_event(trace_id, "pipeline_end", exit=exit_reason)
            _record_metric(
                _import_metric("pipeline_requests").labels(intent=intent_label, exit=exit_reason).inc
            )
            _record_metric(
                _import_metric("pipeline_duration").labels(intent=intent_label, exit=exit_reason).observe,
                _time.monotonic() - t_start,
            )
            mark_span_ok(root)

        except Exception as exc:
            exit_reason = "error"
            logger.error("[harness] pipeline error trace=%s: %s", trace_id, exc, exc_info=True)
            mark_span_error(root, exc)
            _record_metric(
                _import_metric("pipeline_requests").labels(intent=intent_label, exit="error").inc
            )
            yield _sse({"type": "error", "text": "系统内部错误，请稍后重试。"})
            yield _sse({"type": "done"})

        finally:
            root.end()
            try:
                from agent.logging_config import unbind_context
                unbind_context()
            except Exception:
                pass


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _sse(payload: dict) -> str:
    import json
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _safe_hook(hook: OnStopHook, *args) -> None:
    try:
        await hook(*args)
    except Exception as exc:
        logger.error("[safe_hook] error: %s", exc)


async def _safe_hook_with_trace(
    hook: OnStopHook,
    ctx: dict,
    trace_id: str,
    user_content: str,
    assistant_text: str,
    user_id: str,
    session_id: str,
) -> None:
    """
    带 trace context 的 hook 执行。
    在 fire-and-forget task 中恢复 Span 上下文，使 hook 内部的调用链
    在 Jaeger 中显示为 pipeline trace 的子节点。

    面试要点:
      fire-and-forget 不等于 fire-and-forget-observability。
      异步后处理（记忆写入、合规审查）也要有 Span，否则这些耗时的
      可见性为零，运维永远不知道"记个忆怎么花了 2 秒"。
    """
    with span_from_context(ctx, f"hook.{hook.__name__}") as hook_span:
        if hook_span:
            hook_span.set_attributes({
                "hook.name": hook.__name__,
                "user.id": user_id,
                "session.id": session_id,
            })
        try:
            await hook(trace_id, user_content, assistant_text, user_id, session_id)
            mark_span_ok(hook_span) if hook_span else None
        except Exception as exc:
            logger.error("[safe_hook_with_trace] %s error: %s", hook.__name__, exc)
            if hook_span:
                mark_span_error(hook_span, exc)
