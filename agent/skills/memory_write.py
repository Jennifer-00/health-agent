from langchain_core.tools import tool
from memory.mem0_client import Mem0Client
from memory.zep_client import ZepClient


@tool
async def memory_write(user_id: str, content: str, category: str = "general") -> str:
    """
    写入健康记忆。调用 Mem0 ADD/UPDATE，同时写入 Zep 知识图谱。

    Args:
        user_id: 用户唯一标识
        content: 要记录的健康信息（只包含用户陈述的事实）
        category: 分类（症状/用药/就医/检查/其他）
    """
    print(f"\n[memory_write] user={user_id!r} category={category!r}")
    print(f"[memory_write] content: {content!r}")

    client = Mem0Client(user_id=user_id)
    result = await client.add(content, metadata={"category": category})
    print(f"[memory_write] mem0 done: {result.get('results', [{}])[0].get('status', 'ok')}")

    await ZepClient(user_id=user_id).add_fact(content, category)
    print(f"[memory_write] zep fact done")

    return f"已记录：{result.get('id', 'ok')}"
