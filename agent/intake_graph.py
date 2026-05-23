import logging
import operator
from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from agent.prompts import INTAKE_PROMPT

logger = logging.getLogger(__name__)

_llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=512)


class IntakeState(TypedDict):
    messages: Annotated[list, operator.add]
    done: bool
    summary: str


async def intake_node(state: IntakeState) -> dict:
    turns = len([m for m in state["messages"] if isinstance(m, HumanMessage)])
    system = SystemMessage(content=INTAKE_PROMPT)
    response = await _llm.ainvoke([system] + state["messages"])
    raw = response.content if isinstance(response.content, str) else ""

    if "[INTAKE_DONE]" in raw:
        before, _, after = raw.partition("[INTAKE_DONE]")
        summary = after.strip()
        display = before.strip() or "好的，信息收集完毕，正在为您分析…"
        logger.info("[intake] done turns=%d summary_len=%d", turns, len(summary))
        return {
            "messages": [AIMessage(content=display)],
            "done": True,
            "summary": summary,
        }

    logger.info("[intake] ongoing turns=%d", turns)
    return {"messages": [response], "done": False, "summary": ""}


def build_intake_graph():
    g = StateGraph(IntakeState)
    g.add_node("intake", intake_node)
    g.set_entry_point("intake")
    g.add_edge("intake", END)
    return g.compile()


def history_to_lc(history: list[dict]) -> list:
    """将 {"role": ..., "content": ...} 列表转为 LangChain Message 列表。"""
    result = []
    for m in history:
        if m["role"] == "user":
            result.append(HumanMessage(content=m["content"]))
        else:
            result.append(AIMessage(content=m["content"]))
    return result
