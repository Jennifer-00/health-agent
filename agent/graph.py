from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import orchestrator_node


def build_graph() -> StateGraph:
    g = StateGraph(AgentState)
    g.add_node("orchestrator", orchestrator_node)
    g.set_entry_point("orchestrator")
    g.add_edge("orchestrator", END)
    return g.compile()
