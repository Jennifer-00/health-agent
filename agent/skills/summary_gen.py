import re

from langchain_core.tools import tool
from memory.mem0_client import Mem0Client

# 匹配含用药动作的句子片段
_MED_SENTENCE = re.compile(
    r"[^，。；\n]*"
    r"(服用|吃了|用了|使用|外用|涂抹|注射|滴|喷|贴)"
    r"[^，。；\n]*"
    r"(药|布洛芬|泰诺|阿司匹林|抗生素|消炎药|止痛药|眼药水|药膏|维生素|益生菌|激素)"
    r"[^，。；\n]*"
)

# 提取内容开头的日期（格式 YYYY-MM-DD）
_DATE_PREFIX = re.compile(r"(\d{4}-\d{2}-\d{2})")

# 报告输出的分类顺序
_CATEGORY_ORDER = ["症状", "用药", "就医", "生活习惯", "系统", "其他"]


def _extract_med_lines(content: str) -> list[str]:
    """从混合内容中抽取用药句子，附上内容里的日期前缀。"""
    date_match = _DATE_PREFIX.search(content)
    date_prefix = date_match.group(1) + " " if date_match else ""

    results = []
    for m in _MED_SENTENCE.finditer(content):
        sentence = m.group().strip("，。；、 ")
        if sentence:
            results.append(date_prefix + sentence)
    return results


@tool
async def summary_gen(user_id: str) -> str:
    """
    生成结构化健康摘要。从 Mem0 拉取所有记忆，按分类分节输出。

    Args:
        user_id: 用户唯一标识
    """
    client = Mem0Client(user_id=user_id)
    memories = await client.get_all()

    if not memories:
        return "暂无健康记录。"

    by_category: dict[str, list[str]] = {}
    for m in memories:
        content = m.get("memory", "")
        if not content:
            continue
        category = m.get("metadata", {}).get("category") or "其他"

        # 原条目放入原分类（保留完整内容）
        by_category.setdefault(category, []).append(content)

        # 兜底：若非"用药"分类的条目中含用药句子，抽出来单独加入"用药"节
        if category != "用药":
            med_lines = _extract_med_lines(content)
            for line in med_lines:
                by_category.setdefault("用药", []).append(line)

    # 按预定顺序输出，剩余分类追加在末尾
    ordered_cats = _CATEGORY_ORDER + [
        c for c in by_category if c not in _CATEGORY_ORDER
    ]

    sections = []
    for cat in ordered_cats:
        items = by_category.get(cat)
        if not items:
            continue
        lines = [f"  - {item}" for item in items]
        sections.append(f"### {cat}\n" + "\n".join(lines))

    return "\n\n".join(sections)
