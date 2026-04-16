import asyncio

from langchain_core.tools import tool
from memory.mem0_client import Mem0Client
from memory.session_buffer import SessionBuffer
from memory.zep_client import ZepClient


@tool
async def memory_search(user_id: str, query: str, top_k: int = 5) -> str:
    """
    检索相关健康记忆。向量相似检索 + Zep 时序图谱遍历，返回相关记忆列表。

    Args:
        user_id: 用户唯一标识
        query: 检索语义，如"头痛"或"上次用药"
        top_k: 返回条数
    """
    # 优先读取预热缓存（页面加载时触发的 warmup 预拉取结果）
    # 命中则跳过 Mem0 / Zep 网络调用，消除冷启动延迟
    cached = await SessionBuffer.get_mem_prefetch(user_id)
    if cached is not None:
        lines = [f"- {r.get('memory') or r.get('content', '')}" for r in cached[:top_k]]
        return "\n".join(lines) if lines else "未找到相关记忆。"

    mem0 = Mem0Client(user_id=user_id)
    zep = ZepClient(user_id=user_id)

    # 并行调用：Mem0 向量检索 + Zep 图谱遍历同时发出，取较慢那个的时间而非两者之和
    vector_results, graph_results = await asyncio.gather(
        mem0.search(query, limit=top_k),
        zep.traverse(query, hops=2),
    )

    combined = vector_results + graph_results
    if not combined:
        return "未找到相关记忆。"

    lines = [f"- {r.get('memory') or r.get('content', '')}" for r in combined[:top_k]]
    return "\n".join(lines)
