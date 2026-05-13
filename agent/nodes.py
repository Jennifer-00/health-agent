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
from agent.critic_store import load_recent_failures, format_failure_examples

logger = logging.getLogger(__name__)

_llm = ChatOpenAI(model="gpt-4o", streaming=True)
_llm_with_tools = _llm.bind_tools(TOOL_SPECS)

# 防止模型陷入无限工具调用循环
_MAX_TOOL_ROUNDS = 5


async def orchestrator_node(state: AgentState) -> dict:
    today = date.today().strftime("%Y年%m月%d日")

    # 从 ProfileGraph 拉取确认性档案（直接按 user_id fetch，无语义搜索）
    profile_text = ""
    user_id = state.get("user_id", "")
    if user_id:
        try:
            from memory import profile_graph
            profile_text = await profile_graph.fetch_profile(user_id)
        except Exception as exc:
            logger.warning("[orchestrator] profile fetch failed: %s", exc)

    intent = state.get("intent", "general")

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

    # 本地消息列表仅在 tool loop 内部流转，不写回 LangGraph state
    messages = list(state["messages"])
    response = None
    tool_events: list[dict] = []
    web_sources: list[tuple[str, str]] = []
    tool_contents: dict[str, str] = {}       # {tool_name: result_text} for grounding check

    for _round in range(_MAX_TOOL_ROUNDS):
        response = await _llm_with_tools.ainvoke([system] + messages)

        if not response.tool_calls:
            # 模型输出纯文本，tool loop 结束
            break

        # 并行执行本轮所有工具调用，结果追加到本地消息列表后继续推理
        messages.append(response)

        async def _run_one(tc):
            t0 = time.monotonic()
            try:
                args = dict(tc["args"]) if isinstance(tc["args"], dict) else {}
                # 慢病随访时扩大记忆检索范围以支持纵向趋势分析
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
            log_tool_use(
                trace_id=state.get("trace_id", ""),
                tool=tc["name"],
                ms=ms,
                result_len=len(result) if isinstance(result, str) else 0,
            )
            return tc, result, is_error

        gathered = await asyncio.gather(*[_run_one(tc) for tc in response.tool_calls])

        tool_results: list[ToolMessage] = []
        for tc, result, is_error in gathered:
            if tc["name"] == "web_search" and not is_error:
                for title, url in re.findall(r'\*\*\[([^\]]+)\]\(([^)]+)\)\*\*', result):
                    web_sources.append((title, url))

            query = tc["args"].get("query", "") if isinstance(tc["args"], dict) else ""
            tool_events.append({
                "tool": tc["name"],
                "summary": f"查询：{query}" if query else tc["name"],
            })
            # 收集工具结果文本，供幻觉溯源检查和前端可视化（含错误信息）
            if isinstance(result, str):
                tool_contents[tc["name"]] = result

            logger.info(
                "[orchestrator] round=%d tool=%s is_error=%s result_len=%d",
                _round, tc["name"], is_error, len(result),
            )
            tool_results.append(ToolMessage(
                content=result,
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
