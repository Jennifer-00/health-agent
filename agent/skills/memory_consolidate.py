import json
import re
from collections import defaultdict

import redis.asyncio as aioredis
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from memory.mem0_client import Mem0Client
from memory.session_buffer import _pool  # 复用现有连接池

_llm = ChatAnthropic(model="claude-haiku-4-5-20251001")

# Redis key: consolidated:{user_id}  → set of memory IDs already processed
_CONSOLIDATED_TTL = 86400 * 60  # 60 天后自动过期


def _parse_json(text: str) -> list:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()
    return json.loads(text)


_CONSOLIDATE_PROMPT = """以下是用户某一天的健康记录（JSON 格式，每条含 id 和 content 字段）：

{records}

请识别其中语义重复或高度相似的条目，将它们合并为一条简洁、完整的描述。

输出严格为 JSON，格式如下：
[
  {{
    "merged": "合并后的内容",
    "keep_id": "保留的那条记录的 id",
    "delete_ids": ["要删除的其他条目 id", ...]
  }},
  ...
]

注意：
- 只输出需要合并的组，无需合并的条目不用列出
- 只保留用户陈述的事实，去掉推断和诊断倾向
- 如果没有需要合并的内容，输出空数组 []
"""


async def _get_consolidated(user_id: str) -> set[str]:
    """取出该用户已整理过的 memory ID 集合。"""
    key = f"consolidated:{user_id}"
    try:
        r = aioredis.Redis(connection_pool=_pool)
        members = await r.smembers(key)
        return {m if isinstance(m, str) else m.decode() for m in members}
    except Exception:
        return set()


async def _mark_consolidated(user_id: str, ids: list[str]) -> None:
    """将这批 ID 写入已整理集合。"""
    if not ids:
        return
    key = f"consolidated:{user_id}"
    try:
        r = aioredis.Redis(connection_pool=_pool)
        await r.sadd(key, *ids)
        await r.expire(key, _CONSOLIDATED_TTL)
    except Exception:
        pass


async def _consolidate_mem0(user_id: str) -> str:
    lock_key = f"consolidate_lock:{user_id}"
    r = aioredis.Redis(connection_pool=_pool)

    acquired = await r.set(lock_key, "1", nx=True, ex=30)  # nx=仅不存在时写入，ex=30s 超时兜底
    if not acquired:
        return "整理正在进行中，请稍后重试。"

    try:
        return await _do_consolidate(user_id)
    finally:
        await r.delete(lock_key)


async def _do_consolidate(user_id: str) -> str:
    client = Mem0Client(user_id=user_id)
    memories = await client.get_all()
    valid = [m for m in memories if m.get("memory")]

    if len(valid) < 2:
        return "条数不足，跳过。"

    # 已整理的 ID
    done_ids = await _get_consolidated(user_id)

    # 按日期分组，过滤已整理条目
    by_date: dict[str, list] = defaultdict(list)
    for m in valid:
        if m["id"] in done_ids:
            continue
        day = (m.get("created_at") or "")[:10] or "unknown"
        by_date[day].append(m)

    # 每天都有未整理条目才进入处理
    to_process = {d: ms for d, ms in by_date.items() if len(ms) >= 1}

    if not to_process:
        return "所有记录均已整理，无需处理。"

    total_merged = total_deleted = 0

    for day, day_mems in sorted(to_process.items()):
        if len(day_mems) == 1:
            # 只有一条，直接打标，不用发给 LLM
            await _mark_consolidated(user_id, [day_mems[0]["id"]])
            continue

        # 多条 → 发给 LLM 合并
        slim = [{"id": m["id"], "content": m["memory"]} for m in day_mems]
        prompt = _CONSOLIDATE_PROMPT.format(
            records=json.dumps(slim, ensure_ascii=False, indent=2)
        )
        response = await _llm.ainvoke([HumanMessage(content=prompt)])

        try:
            groups: list[dict] = _parse_json(response.content)
        except Exception:
            continue  # 解析失败跳过这天

        merged_ids: set[str] = set()  # 参与合并的所有 ID

        for group in groups:
            keep_id = group.get("keep_id", "")
            merged_content = group.get("merged", "")
            delete_ids = group.get("delete_ids", [])
            if not keep_id or not merged_content:
                continue
            await client.update(keep_id, merged_content)
            for did in delete_ids:
                await client.delete(did)
            merged_ids.add(keep_id)
            merged_ids.update(delete_ids)
            total_merged += 1
            total_deleted += len(delete_ids)

        # 没参与合并的当天条目也打标（语义不重复，不需要再次处理）
        all_day_ids = {m["id"] for m in day_mems}
        await _mark_consolidated(user_id, list(all_day_ids))

    if total_merged == 0:
        return "各日期条目语义不重复，均已标记为已整理。"
    return f"整理完成：{total_merged} 组合并，删除 {total_deleted} 条。"


@tool
async def memory_consolidate(user_id: str) -> str:
    """
    合并 Mem0 中重复或高度相似的健康记录，去重降噪。

    Args:
        user_id: 用户唯一标识
    """
    return await _consolidate_mem0(user_id)
