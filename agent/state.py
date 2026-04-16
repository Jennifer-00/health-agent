from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """LangGraph 图状态：消息、用户标识、待写记忆和工具事件。"""

    messages: Annotated[list, add_messages]
    user_id: str
    pending_memories: list[dict]
    tool_events: list[dict]
    skip_memory_search: bool  # Context 层冷却机制：True 时提示 LLM 跳过 memory_search
