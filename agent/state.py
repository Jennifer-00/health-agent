from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    trace_id: str
    intent: str              # L1.5 意图分类结果，供 orchestrator_node 选策略 hint
    tool_events: list        # populated by orchestrator_node, read by harness for SSE
    web_sources: list        # [(title, url), ...] collected from web_search results
    tool_contents: dict      # {tool_name: result_text} for hallucination grounding check
