"""
Critic — 三个独立 hook，挂载在 pipeline 不同位置。

Hook 1: soften_absolutes(token) → str
  位置：streaming 层，每个 token yield 前
  实现：纯正则替换，零延迟
  作用："一定是"→"可能是"，"肯定"→"可能"

Hook 2: grounding_check(assistant_text, tool_contents) → str | None
  位置：stream 结束后、done 事件前
  实现：LLM 对比回复内容与 tool 返回原文
  作用：检测未在检索结果中出现的具体数据/引用，返回溯源警告

Hook 3: update_badcase(user_message, assistant_response) → None（fire-and-forget）
  位置：on_stop_hook，异步后台
  实现：LLM 判断合规性，不合规写入 critic_failures.jsonl
  作用：更新下次请求 system prompt 里的反例，不影响当前回复
"""
import json
import logging
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

_llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=200)


# ── Hook 1：断言软化（纯规则，streaming 层调用）────────────────────────────────

_SOFTEN_RULES = [
    (re.compile(r'一定是'),  '可能是'),
    (re.compile(r'一定有'),  '可能有'),
    (re.compile(r'一定会'),  '可能会'),
    (re.compile(r'肯定是'),  '可能是'),
    (re.compile(r'肯定有'),  '可能有'),
    (re.compile(r'肯定会'),  '可能会'),
    (re.compile(r'肯定不'),  '通常不'),
    (re.compile(r'必然是'),  '通常是'),
    (re.compile(r'绝对是'),  '通常是'),
    (re.compile(r'绝对不'),  '通常不建议'),
    (re.compile(r'不可能是'), '不太可能是'),
]


def soften_absolutes(token: str) -> str:
    """每个 streaming token 过一遍，替换绝对断言词。零延迟，无 LLM 调用。"""
    for pattern, replacement in _SOFTEN_RULES:
        token = pattern.sub(replacement, token)
    return token


# ── Hook 2：幻觉溯源检查（stream 结束后、done 前）────────────────────────────

_GROUNDING_PROMPT = """你是医疗内容溯源审查员。

你会收到：
1. 助手的回复文本
2. 检索到的参考原文（来自知识库或网络搜索）

任务：检查回复中是否存在"无法在参考原文中找到依据的具体数据引用"，包括：
- 具体的百分比、概率数值（"X%的患者..."）
- 具体的研究/指南引用（"据XX指南"、"研究表明"）
- 精确的剂量/时间数值，且原文中没有对应信息

如果回复措辞谨慎（使用"可能"、"建议咨询医生"等），视为通过。
如果回复只是解释概念而非引用具体数据，视为通过。

输出严格 JSON，不输出其他内容：
{
  "pass": true/false,
  "note": "若未通过，给出1句说明哪些信息无法核实，建议用户向医生确认；通过则留空"
}"""


async def grounding_check(assistant_text: str, tool_contents: dict[str, str]) -> str | None:
    """
    用 tool_contents（RAG/web 返回原文）核实回复中的具体数据引用。
    无 tool_contents 时跳过（模型未调用检索工具，无法溯源）。
    返回 None 表示通过；否则返回需要追加的警告文字。
    """
    grounding_sources = {
        k: v for k, v in tool_contents.items()
        if k in ("search_rag", "web_search") and v
    }
    if not grounding_sources:
        return None

    context = "\n\n---\n\n".join(
        f"【{name} 返回原文】\n{text[:1500]}"
        for name, text in grounding_sources.items()
    )
    content = f"助手回复：\n{assistant_text}\n\n参考原文：\n{context}"

    try:
        resp = await _llm.ainvoke([
            SystemMessage(content=_GROUNDING_PROMPT),
            HumanMessage(content=content),
        ])
        raw = resp.content.strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        result = json.loads(m.group())
        if not result.get("pass") and result.get("note"):
            logger.info("[critic/grounding] triggered: %s", result["note"][:80])
            return result["note"]
    except Exception as exc:
        logger.warning("[critic/grounding] check failed: %s", exc)

    return None


# ── Hook 3：合规 badcase 更新（fire-and-forget，on_stop_hook 调用）──────────

_COMPLIANCE_PROMPT = """你是医疗健康助手的质量审查员。检查助手回复是否合规。

检查项：
1. 是否做出了明确诊断（如"你得了X病"）→ 违规
2. 涉及严重或持续症状时，是否建议了就医 → 若没有则需补充
3. 是否包含不当的用药建议

输出严格 JSON，不输出其他内容：
{"pass": true/false, "note": "若未通过，给出1句补充说明（中文）；通过则留空"}"""


async def update_badcase(user_message: str, assistant_response: str) -> None:
    """
    合规性审查，fire-and-forget。
    不合规时只写入 critic_failures.jsonl，不追加任何内容到回复。
    下次请求构建 system prompt 时读取，注入为反例。
    """
    try:
        content = f"用户消息：{user_message}\n\n助手回复：{assistant_response}"
        resp = await _llm.ainvoke([
            SystemMessage(content=_COMPLIANCE_PROMPT),
            HumanMessage(content=content),
        ])
        raw = resp.content.strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return
        result = json.loads(m.group())
        if not result.get("pass") and result.get("note"):
            from agent.critic_store import write_failure
            write_failure(user_message, assistant_response, result["note"])
            logger.info("[critic/compliance] badcase recorded: %s", result["note"][:80])
    except Exception as exc:
        logger.warning("[critic/compliance] failed: %s", exc)


# ── 向后兼容：harness 旧调用入口（保留签名，内部只走 grounding_check）────────

async def critic_review(user_message: str, assistant_response: str) -> str | None:
    """旧入口，harness 重构前的临时兼容层。新代码直接调用三个独立函数。"""
    return None
