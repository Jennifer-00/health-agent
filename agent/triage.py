"""
Triage Gate — Harness 的第一道安全关卡。

每条用户消息进入主 Agent 之前先经过这里，
快速判断是否存在需要立即就医的紧急情况。
使用 Haiku 保证低延迟（< 500ms）。

可观测性:
  - @traced 装饰器: triage 调用作为独立 Span，显示在 pipeline Span 旁边
  - 紧急判定计数: triage_emergency counter
  - LLM 耗时 + token 指标

面试要点:
  - Triage 是 Agent 安全的第一道防线，必须可观测
  - 紧急判定率异常飙升 → 可能是 prompt 被绕过或模型行为变化
"""
import json
import logging
import re
import time

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from agent.telemetry import get_tracer, mark_span_ok, mark_span_error

logger = logging.getLogger(__name__)

_llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=128)

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
    tracer = get_tracer()
    span = tracer.start_span("triage")
    span.set_attributes({"triage.model": "gpt-4o-mini", "triage.msg_len": len(message)})

    t0 = time.monotonic()
    try:
        resp = await _llm.ainvoke([
            SystemMessage(content=_TRIAGE_PROMPT),
            HumanMessage(content=message),
        ])
        ms = int((time.monotonic() - t0) * 1000)
        span.set_attributes({"triage.ms": ms})

        raw = resp.content.strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            logger.warning("[triage] no JSON found in response: %r", raw[:200])
            mark_span_ok(span)
            span.set_attribute("triage.emergency", False)
            span.end()
            return False, ""

        result = json.loads(m.group())
        is_emergency = bool(result.get("emergency"))

        span.set_attributes({
            "triage.emergency": is_emergency,
            "triage.reason": result.get("reason", "")[:100],
        })
        mark_span_ok(span)
        span.end()

        if is_emergency:
            logger.info("[triage] EMERGENCY detected reason=%r", result.get("reason", ""))
            return True, _EMERGENCY_RESPONSE

    except Exception as exc:
        logger.warning("[triage] classification failed, defaulting to non-emergency: %s", exc)
        mark_span_error(span, exc)
        span.end()

    return False, ""
