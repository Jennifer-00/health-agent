"""
Orchestrator node — LangGraph 主推理节点。

负责:
  1. 构建 system prompt（档案 + 策略 + 反例）
  2. tool loop（LLM invoke ↔ 工具执行，最多 5 轮）
  3. 工具结果压缩 + 幻觉溯源源数据收集

可观测性:
  - @traced 装饰器: orchestrator_node 整体作为一个 Span
  - 每轮 tool loop 内有 LLM Span + 各工具 Span
  - 工具执行指标: 耗时、结果大小、错误率

面试要点:
  - Agent 最贵的是 tool loop 里的 LLM 调用，必须单独 Span 才能看到每轮的 token 消耗
  - 工具并行执行时，Span 的时间线重叠天然反映并行关系
  - 工具结果压缩 是 Agent 特有的埋点场景: compressed_len vs raw_len
"""
import asyncio
import logging
import re
import time
from datetime import date

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, ToolMessage

from agent.observability import log_tool_use
from agent.prompts import SYSTEM_PROMPT, INTENT_STRATEGY
from agent.intent import CHRONIC_FOLLOWUP
from agent.state import AgentState
from agent.tools import TOOL_SPECS, dispatch_tool
from agent.tool_result_compressor import compress_tool_result
from agent.critic_store import load_recent_failures, format_failure_examples
from agent.telemetry import get_tracer, mark_span_ok, mark_span_error

logger = logging.getLogger(__name__)

_llm = ChatOpenAI(model="gpt-4o", streaming=True)
_llm_with_tools = _llm.bind_tools(TOOL_SPECS)

_MAX_TOOL_ROUNDS = 5

# ── Metric helpers ──────────────────────────────────────────────────────────────

def _metric_inc(name: str, **labels):
    try:
        import importlib
        m = importlib.import_module("agent.metrics")
        metric = getattr(m, name)
        if labels:
            metric = metric.labels(**labels)
        metric.inc()
    except Exception:
        pass


def _metric_observe(name: str, value: float, **labels):
    try:
        import importlib
        m = importlib.import_module("agent.metrics")
        metric = getattr(m, name)
        if labels:
            metric = metric.labels(**labels)
        metric.observe(value)
    except Exception:
        pass


async def orchestrator_node(state: AgentState) -> dict:
    trace_id = state.get("trace_id", "")
    tracer = get_tracer()
    today = date.today().strftime("%Y年%m月%d日")

    # ── ProfileGraph 档案 ─────────────────────────────────────────────────
    profile_text = ""
    user_id = state.get("user_id", "")
    if user_id:
        try:
            from memory import profile_graph
            profile_text = await profile_graph.fetch_profile(user_id)
        except Exception as exc:
            logger.warning("[orchestrator] profile fetch failed: %s", exc)

    intent = state.get("intent", "general")

    # ── 构建 system prompt ─────────────────────────────────────────────────
    prompt = f"今天是 {today}。\n\n"
    if profile_text:
        prompt += f"<user_profile>\n{profile_text}\n</user_profile>\n\n"
    prompt += SYSTEM_PROMPT

    strategy = INTENT_STRATEGY.get(intent, "")
    if strategy:
        prompt += f"\n\n<strategy>\n{strategy}\n</strategy>"

    failures = load_recent_failures()
    failure_examples = format_failure_examples(failures)
    if failure_examples:
        prompt += f"\n\n{failure_examples}"

    system = SystemMessage(content=prompt)

    messages = list(state["messages"])
    response = None
    tool_events: list[dict] = []
    web_sources: list[tuple[str, str]] = []
    tool_contents: dict[str, str] = {}

    for _round in range(_MAX_TOOL_ROUNDS):
        # ── LLM Span (每轮独立) ────────────────────────────────────────────
        llm_span = tracer.start_span(f"orchestrator.round.{_round}.llm")
        llm_span.set_attributes({
            "llm.round": _round,
            "llm.model": "gpt-4o",
            "user.id": user_id,
        })
        t_llm = time.monotonic()

        response = await _llm_with_tools.ainvoke([system] + messages)

        llm_ms = int((time.monotonic() - t_llm) * 1000)
        meta = getattr(response, "usage_metadata", None) or {}
        tokens_in = meta.get("input_tokens", 0)
        tokens_out = meta.get("output_tokens", 0)
        tool_calls_list = getattr(response, "tool_calls", None) or []

        llm_span.set_attributes({
            "gen_ai.usage.input_tokens": tokens_in,
            "gen_ai.usage.output_tokens": tokens_out,
            "gen_ai.tools.requested": [tc["name"] for tc in tool_calls_list],
            "gen_ai.tools.count": len(tool_calls_list),
            "llm.ms": llm_ms,
        })
        mark_span_ok(llm_span)
        llm_span.end()

        # 记录 LLM 指标
        _metric_inc("llm_calls", node="orchestrator", model="gpt-4o")
        _metric_inc("llm_tokens", node="orchestrator", model="gpt-4o", direction="input")
        _metric_inc("llm_tokens", node="orchestrator", model="gpt-4o", direction="output")

        if not response.tool_calls:
            break

        messages.append(response)

        # ── 工具 Span (并行执行, 各自独立 Span) ──────────────────────────────
        tools_span = tracer.start_span(f"orchestrator.round.{_round}.tools")
        tools_span.set_attributes({
            "tools.round": _round,
            "tools.count": len(response.tool_calls),
        })

        async def _run_one(tc):
            t0 = time.monotonic()
            tool_name = tc["name"]
            tool_span = tracer.start_span(f"tool.{tool_name}")
            tool_span.set_attributes({
                "tool.name": tool_name,
                "tool.round": _round,
            })
            try:
                args = dict(tc["args"]) if isinstance(tc["args"], dict) else {}
                if tc["name"] == "search_memory" and intent == CHRONIC_FOLLOWUP:
                    args.setdefault("limit", 12)
                result, is_error = await dispatch_tool(
                    tc["name"], args, user_id=state.get("user_id", "")
                )
            except Exception as exc:
                logger.error(
                    "[orchestrator] dispatch_tool unexpected error round=%d tool=%s: %s",
                    _round, tc["name"], exc, exc_info=True,
                )
                result, is_error = f"工具 {tc['name']} 执行异常，请根据已有知识作答。", True

            ms = int((time.monotonic() - t0) * 1000)
            result_len = len(result) if isinstance(result, str) else 0

            # OTEL: tool span attributes
            tool_span.set_attributes({
                "tool.ms": ms,
                "tool.result_len": result_len,
                "tool.is_error": is_error,
            })
            if is_error:
                mark_span_error(tool_span)
            else:
                mark_span_ok(tool_span)
            tool_span.end()

            # Prometheus: tool metrics
            _metric_inc("tool_calls", tool=tool_name, status="error" if is_error else "success")
            _metric_observe("tool_duration", ms / 1000.0, tool=tool_name)
            if result_len:
                _metric_observe("tool_result_size", result_len, tool=tool_name)

            # 保留原有 JSONL 日志
            log_tool_use(
                trace_id=trace_id,
                tool=tc["name"],
                ms=ms,
                result_len=result_len,
            )
            return tc, result, is_error

        gathered = await asyncio.gather(*[_run_one(tc) for tc in response.tool_calls])
        mark_span_ok(tools_span)
        tools_span.end()

        tool_results: list[ToolMessage] = []
        for tc, result, is_error in gathered:
            if tc["name"] == "web_search" and not is_error:
                for title, url in re.findall(r'\*\*\[([^\]]+)\]\(([^)]+)\)\*\*', result):
                    web_sources.append((title, url))

            if isinstance(result, str):
                tool_contents[tc["name"]] = result

            query = tc["args"].get("query", "") if isinstance(tc["args"], dict) else ""

            tool_events.append({
                "tool": tc["name"],
                "summary": f"查询：{query}" if query else tc["name"],
            })

            if is_error or not isinstance(result, str):
                llm_result = result
            else:
                llm_result = compress_tool_result(tc["name"], result, query=query)

            logger.info(
                "[orchestrator] round=%d tool=%s is_error=%s result_len=%d llm_len=%d",
                _round, tc["name"], is_error,
                len(result) if isinstance(result, str) else 0,
                len(llm_result) if isinstance(llm_result, str) else 0,
                extra={
                    "round": _round, "tool": tc["name"],
                    "is_error": is_error,
                    "result_len": len(result) if isinstance(result, str) else 0,
                    "llm_len": len(llm_result) if isinstance(llm_result, str) else 0,
                },
            )
            tool_results.append(ToolMessage(
                content=llm_result,
                tool_call_id=tc["id"],
                name=tc["name"],
                status="error" if is_error else "success",
            ))
        messages.extend(tool_results)
    else:
        logger.warning(
            "[orchestrator] tool loop reached max rounds (%d), returning last response",
            _MAX_TOOL_ROUNDS,
        )

    return {
        "messages":     [response],
        "tool_events":  tool_events,
        "web_sources":  web_sources,
        "tool_contents": tool_contents,
    }
