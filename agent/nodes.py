from datetime import date

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, ToolMessage

from agent.prompts import SYSTEM_PROMPT
from agent.skills.memory_search import memory_search
from agent.skills.memory_write import memory_write
from agent.skills.summary_gen import summary_gen
from agent.state import AgentState
from memory.write_queue import enqueue_write

_tools = [memory_search, memory_write, summary_gen]
_tool_map = {t.name: t for t in _tools}
_llm = ChatAnthropic(model="claude-sonnet-4-6").bind_tools(_tools)


async def orchestrator_node(state: AgentState) -> dict:
    """调用 LLM，收集所有待执行的工具调用。"""
    today = date.today().strftime("%Y年%m月%d日")

    prompt = f"今天是 {today}。\n\n" + SYSTEM_PROMPT
    if state.get("skip_memory_search"):
        prompt += (
            "\n\n[Context 提示] 本轮无需调用 memory_search，"
            "记忆上下文已在本次会话近期获取，请直接基于对话历史回复。"
        )

    system = SystemMessage(content=[
        {
            "type": "text",
            "text": prompt,
            "cache_control": {"type": "ephemeral"},
        }
    ])
    response = await _llm.ainvoke([system] + state["messages"])

    pending = list(response.tool_calls or [])

    return {
        "messages": [response],
        "pending_memories": pending,
    }


async def tool_executor_node(state: AgentState) -> dict:
    """执行所有挂起的工具调用，将结果封装为 ToolMessage 送回 LLM。

    memory_write：提交到串行写入队列，不阻塞 LLM。
    读操作：asyncio.gather 并行执行。
    """
    import asyncio

    tool_messages: list[ToolMessage] = []
    write_placeholders: dict[str, str] = {}
    read_calls: list[dict] = []

    for call in state["pending_memories"]:
        args = {**call["args"], "user_id": state["user_id"]}
        if call["name"] == "memory_write":
            tool = _tool_map.get("memory_write")
            if tool:
                enqueue_write(state["user_id"], tool.ainvoke(args))
            write_placeholders[call["id"]] = "已在后台写入"
        else:
            read_calls.append(call)

    if read_calls:
        async def _invoke(call: dict) -> str:
            tool = _tool_map.get(call["name"])
            args = {**call["args"], "user_id": state["user_id"]}
            if not tool:
                return f"未知工具：{call['name']}"
            try:
                return await tool.ainvoke(args)
            except Exception as exc:
                return f"[工具执行失败] {call['name']}: {exc}"

        read_results = await asyncio.gather(*(_invoke(c) for c in read_calls))
        for call, result in zip(read_calls, read_results):
            tool_messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    for call in state["pending_memories"]:
        if call["id"] in write_placeholders:
            tool_messages.append(
                ToolMessage(content=write_placeholders[call["id"]], tool_call_id=call["id"])
            )

    call_id_order = {c["id"]: i for i, c in enumerate(state["pending_memories"])}
    tool_messages.sort(key=lambda m: call_id_order.get(m.tool_call_id, 999))

    return {
        "messages": tool_messages,
        "pending_memories": [],
    }


