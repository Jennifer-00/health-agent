from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import orchestrator_node, tool_executor_node, reply_node


def should_use_tools(state: AgentState) -> str:
    """条件路由：有待执行工具则执行，否则直接回复"""
    if state.get("pending_memories"):
        return "tool_executor"
    return "reply"


def build_graph() -> StateGraph:
    """LangGraph StateGraph，定义节点、边和条件路由逻辑"""
    g = StateGraph(AgentState)

    g.add_node("orchestrator", orchestrator_node)
    g.add_node("tool_executor", tool_executor_node)
    g.add_node("reply", reply_node)

    g.set_entry_point("orchestrator")
    g.add_conditional_edges("orchestrator", should_use_tools)
    g.add_edge("tool_executor", "orchestrator")  # 工具结果送回 LLM，形成 ReAct 循环
    g.add_edge("reply", END)

    return g.compile()
