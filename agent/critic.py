"""
Critic Agent — Harness 的质量审查层。

主 Agent 回复生成后，Critic 检查内容合规性：
- 是否做出了明确诊断（违规）
- 涉及严重症状时是否建议就医
- 是否包含不当医疗建议

若发现问题，返回一条补充说明追加到回复末尾。
通过则返回 None，不影响正常流程。
使用 Haiku 保证低延迟。
"""
import json
import logging
import re

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

_llm = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=128)

_CRITIC_PROMPT = """你是医疗健康助手的质量审查员。检查助手回复是否合规。

检查项：
1. 是否做出了明确诊断（如"你得了X病"）→ 违规
2. 涉及严重或持续症状时，是否建议了就医 → 若没有则需补充
3. 是否包含不当的用药建议

输出严格 JSON，不输出其他内容：
{"pass": true 或 false, "note": "若未通过，给出1句补充说明（中文）；通过则留空"}"""


async def critic_review(user_message: str, assistant_response: str) -> str | None:
    """
    审查回复质量。
    返回 None 表示通过；否则返回需要追加到回复末尾的补充说明。
    """
    try:
        content = f"用户消息：{user_message}\n\n助手回复：{assistant_response}"
        resp = await _llm.ainvoke([
            SystemMessage(content=_CRITIC_PROMPT),
            HumanMessage(content=content),
        ])
        raw = resp.content.strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            logger.warning("[critic] no JSON found in response: %r", raw[:200])
            return None
        result = json.loads(m.group())
        if not result.get("pass") and result.get("note"):
            from agent.critic_store import write_failure
            write_failure(user_message, assistant_response, result["note"])
            return result["note"]
    except Exception as exc:
        logger.warning("[critic] review failed: %s", exc)
    return None
