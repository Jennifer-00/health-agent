import logging
import re
from datetime import date

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, ToolMessage

from agent.prompts import SYSTEM_PROMPT
from agent.state import AgentState
from agent.tools import TOOL_SPECS, dispatch_tool
from agent.critic_store import load_recent_failures, format_failure_examples

logger = logging.getLogger(__name__)

_llm = ChatAnthropic(model="claude-sonnet-4-6", streaming=True)
_llm_with_tools = _llm.bind_tools(TOOL_SPECS)

# 防止模型陷入无限工具调用循环
_MAX_TOOL_ROUNDS = 5


async def orchestrator_node(state: AgentState) -> dict:
    today = date.today().strftime("%Y年%m月%d日")
    prompt = f"今天是 {today}。\n\n" + SYSTEM_PROMPT

    failures = load_recent_failures()
    failure_examples = format_failure_examples(failures)
    if failure_examples:
        prompt += f"\n\n{failure_examples}"

    system = SystemMessage(content=[
        {
            "type": "text",
            "text": prompt,
            "cache_control": {"type": "ephemeral"},
        }
    ])

    # 本地消息列表仅在 tool loop 内部流转，不写回 LangGraph state
    messages = list(state["messages"])
    response = None
    tool_events: list[dict] = []
    web_sources: list[tuple[str, str]] = []   # (title, url) from web_search results

    for _round in range(_MAX_TOOL_ROUNDS):
        response = await _llm_with_tools.ainvoke([system] + messages)

        if not response.tool_calls:
            # 模型输出纯文本，tool loop 结束
            break

        # 执行本轮所有工具调用，结果追加到本地消息列表后继续推理
        messages.append(response)
        tool_results: list[ToolMessage] = []
        for tc in response.tool_calls:
            try:
                result, is_error = await dispatch_tool(tc["name"], tc["args"], user_id=state.get("user_id", ""))
            except Exception as exc:
                # dispatch_tool 自身有 try/except，此处作为最后一道防线：
                # 单个工具意外抛出时不中断本轮其余工具调用
                logger.error(
                    "[orchestrator] dispatch_tool unexpected error round=%d tool=%s: %s",
                    _round, tc["name"], exc, exc_info=True,
                )
                result, is_error = f"工具 {tc['name']} 执行异常，请根据已有知识作答。", True

            if tc["name"] == "web_search" and not is_error:
                for title, url in re.findall(r'\*\*\[([^\]]+)\]\(([^)]+)\)\*\*', result):
                    web_sources.append((title, url))

            query = tc["args"].get("query", "") if isinstance(tc["args"], dict) else ""
            tool_events.append({
                "tool": tc["name"],
                "summary": f"查询：{query}" if query else tc["name"],
            })
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

    # 只将最终文本回复写入 LangGraph state；tool_events/web_sources 供 harness 发送 SSE
    return {"messages": [response], "tool_events": tool_events, "web_sources": web_sources}
