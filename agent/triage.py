"""
Triage Gate — Harness 的第一道安全关卡。

每条用户消息进入主 Agent 之前先经过这里，
快速判断是否存在需要立即就医的紧急情况。
使用 Haiku 保证低延迟（< 500ms）。
"""
import json

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

_llm = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=128)

_TRIAGE_PROMPT = """你是医疗紧急情况分诊系统。判断用户消息是否包含需要立即就医的紧急症状。

紧急情况包括：胸痛、心跳骤停、呼吸困难、意识丧失、严重出血、中毒、严重过敏、高烧40度以上、疑似脑卒中等。

输出严格 JSON，不输出其他内容：
{"emergency": true 或 false, "reason": "若紧急则说明原因，否则留空"}"""

_EMERGENCY_RESPONSE = """⚠️ 您描述的症状可能需要立即就医。

请立即采取以下措施：
1. 拨打急救电话 **120**
2. 前往最近的急诊室
3. 通知家人或朋友陪同

本系统无法替代专业医疗救助，请优先寻求紧急医疗帮助。"""


async def triage(message: str) -> tuple[bool, str]:
    """
    返回 (is_emergency, response_text)。
    is_emergency=True 时 response_text 为急救提示，否则为空字符串。
    """
    try:
        resp = await _llm.ainvoke([
            SystemMessage(content=_TRIAGE_PROMPT),
            HumanMessage(content=message),
        ])
        # Haiku 有时在 JSON 外包 ```json ... ``` 代码块，需剥离后再解析
        raw = resp.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]          # 取第一个 ``` 之后的内容
            raw = raw.lstrip("json").strip()       # 去掉可能的语言标识 "json"
        result = json.loads(raw)
        if result.get("emergency"):
            return True, _EMERGENCY_RESPONSE
    except Exception:
        pass  # 解析失败默认放行，不阻塞正常流程
    return False, ""
