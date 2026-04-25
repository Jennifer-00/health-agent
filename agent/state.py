from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    tool_events: list   # populated by orchestrator_node, read by harness for SSE
    web_sources: list   # [(title, url), ...] collected from web_search results
