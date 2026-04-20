from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import orchestrator_node, tool_executor_node


def should_use_tools(state: AgentState) -> str:
    if state.get("pending_memories"):
        return "tool_executor"
    return END


def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("orchestrator", orchestrator_node)
    g.add_node("tool_executor", tool_executor_node)

    g.set_entry_point("orchestrator")
    g.add_conditional_edges("orchestrator", should_use_tools)
    g.add_edge("tool_executor", "orchestrator")

    return g.compile()
