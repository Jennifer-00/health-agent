import asyncio
from datetime import date

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from agent.prompts import SYSTEM_PROMPT
from agent.skills.memory_consolidate import memory_consolidate
from agent.skills.memory_search import memory_search
from agent.skills.memory_write import memory_write
from agent.skills.record_link import record_link
from agent.skills.summary_gen import summary_gen
from agent.state import AgentState

_tools = [memory_search, memory_write, record_link, summary_gen, memory_consolidate]
_tool_map = {t.name: t for t in _tools}
_llm = ChatAnthropic(model="claude-sonnet-4-6").bind_tools(_tools)

# 只写入、不需要返回值的工具 → 后台执行，不阻塞 LLM 生成回复
_WRITE_ONLY_TOOLS = {"memory_write", "record_link", "memory_consolidate"}

# 持有后台 task 引用，防止 GC 在任务完成前销毁
_background_tasks: set[asyncio.Task] = set()


def _fire_and_forget(coro) -> None:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


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

    写操作（_WRITE_ONLY_TOOLS）：fire-and-forget，不阻塞 LLM。
    读操作：asyncio.gather 并行执行，多个 memory_search 同时发出，
    总耗时 = max(单次耗时) 而非 sum(所有耗时)。
    """
    tool_events: list[dict] = []
    tool_messages: list[ToolMessage] = []

    # 先处理写操作（直接 fire-and-forget，不进并行池）
    write_placeholders: dict[str, str] = {}  # call_id → "已在后台写入"
    read_calls: list[dict] = []

    for call in state["pending_memories"]:
        args = {**call["args"], "user_id": state["user_id"]}
        if call["name"] in _WRITE_ONLY_TOOLS:
            tool = _tool_map.get(call["name"])
            if tool:
                _fire_and_forget(tool.ainvoke(args))
            write_placeholders[call["id"]] = "已在后台写入"
            tool_events.append({"tool": call["name"], "summary": "已在后台写入"})
        else:
            read_calls.append(call)

    # 所有读操作并行执行
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

    # 写操作的占位 ToolMessage（LLM 需要每个 tool_call_id 都有对应回复）
    for call in state["pending_memories"]:
        if call["id"] in write_placeholders:
            tool_messages.append(
                ToolMessage(content=write_placeholders[call["id"]], tool_call_id=call["id"])
            )

    # 保持 ToolMessage 顺序与 pending_memories 一致，避免 LLM 混淆
    call_id_order = {c["id"]: i for i, c in enumerate(state["pending_memories"])}
    tool_messages.sort(key=lambda m: call_id_order.get(m.tool_call_id, 999))

    return {
        "messages": tool_messages,
        "pending_memories": [],
        "tool_events": tool_events,
    }


async def reply_node(state: AgentState) -> dict:
    """
    回复质量兜底节点（graph 层，只管回复内容）。

    反复发作的"检测与告警"已由 harness 的 post_tool_hook（_recurrence_alert_hook）负责。
    这里只做一件事：若 record_link 找到了关联历史，但最终回复没有提及，
    则注入 recurrence_hint 重新调 LLM，确保回复包含发作时间线和就医建议。

    正常路径：orchestrator 末轮回复已提及发作历史，直接透传。
    兜底路径：回复遗漏了，重新生成。
    """
    recurrence_events = [
        e for e in state.get("tool_events", [])
        if e["tool"] == "record_link" and "关联历史" in e["summary"]
    ]

    if not recurrence_events:
        return {}

    # 取 messages 里最后一条 AIMessage（orchestrator 末轮生成的回复）
    last_ai: AIMessage | None = next(
        (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
        None,
    )

    # 若回复已提到发作历史，不重复调用
    if last_ai and any(
        kw in last_ai.content
        for kw in ("发作", "历史", "上次", "次", "之前")
    ):
        return {}

    # 兜底：注入反复发作提示，重新生成回复
    recurrence_hint = (
        "\n\n[系统提示：本次已通过 record_link 检测到反复发作历史，"
        "以下是关联记录摘要：\n"
        + "\n".join(e["summary"] for e in recurrence_events)
        + "\n请在回复中明确告知用户历次发作的时间和次数，"
        "若发作超过 2 次须建议就医。]"
    )
    system = SystemMessage(content=[
        {
            "type": "text",
            "text": SYSTEM_PROMPT + recurrence_hint,
            "cache_control": {"type": "ephemeral"},
        }
    ])
    response = await _llm.ainvoke([system] + state["messages"])
    return {"messages": [response]}
