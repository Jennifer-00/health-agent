from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    pending_memories: list[dict]
    skip_memory_search: bool  # Context 层冷却机制：True 时提示 LLM 跳过 memory_search
