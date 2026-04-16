from langchain_core.tools import tool
from memory.zep_client import ZepClient


@tool
async def record_link(user_id: str, current_content: str) -> str:
    """
    关联历史病历节点。检测"又开始…"语义，在知识图谱中标记反复发作关系。

    Args:
        user_id: 用户唯一标识
        current_content: 当前症状描述
    """
    zep = ZepClient(user_id=user_id)
    linked = await zep.find_recurrence(current_content)

    if not linked:
        return "未发现相关历史记录。"

    # 写入反复发作边（传可读摘要，避免 UUID 污染图谱文本）
    for node in linked:
        await zep.add_recurrence_edge(node["summary"], current_content)

    summaries = [f"- {n['summary']} ({n['date']})" for n in linked]
    return "发现关联历史记录：\n" + "\n".join(summaries)
