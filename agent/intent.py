"""
agent/intent.py — L1.5 意图分类

在 L1 Triage 通过后、L2 Orchestrator 启动前运行，与 Triage 并行。
结果注入 AgentState.intent，供 orchestrator_node 选择策略 hint。

分类逻辑：
  1. 关键词/正则快速匹配（零延迟，高置信场景直接返回）
  2. Haiku 兜底（< 300ms）
  3. 任何异常 → 降级为 "general"，不阻断主流程

可观测性:
  - @traced: intent 分类作为独立 Span
  - span 属性记录 method (rule/llm) 和结果
  - 规则命中率是效率指标: 越高越省成本

面试要点:
  - 意图分类有两条路径: 规则和 LLM。规则命中率高 = 省钱 + 低延迟
  - 如果 LLM 回退率突然升高，说明用户输入分布变了，规则需要更新
"""
import json
import logging
import re
import time

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from agent.telemetry import get_tracer, mark_span_ok, mark_span_error

logger = logging.getLogger(__name__)

# ── 意图常量 ──────────────────────────────────────────────────────────────────
GENERAL               = "general"
SYMPTOM_REPORT        = "symptom_report"
MEDICATION_CONSULT    = "medication_consultation"
ALLERGY               = "allergy_or_adverse_reaction"
CHRONIC_FOLLOWUP      = "chronic_followup"
LIFESTYLE             = "lifestyle_advice"
RECORD_UPDATE         = "record_update"
MEMORY_QUERY          = "memory_query"

_ALL = {SYMPTOM_REPORT, MEDICATION_CONSULT, ALLERGY, CHRONIC_FOLLOWUP,
        LIFESTYLE, RECORD_UPDATE, MEMORY_QUERY, GENERAL}

_RULES: list[tuple[str, re.Pattern]] = [
    (CHRONIC_FOLLOWUP, re.compile(
        r'(\d+\.?\d*)\s*(血糖|血压|心率|体重|体温|尿酸|血脂|hmol|mmhg|bpm|kg|℃|度)'
        r'|'
        r'(血糖|血压|心率|体重|体温|尿酸|血脂)\s*(\d+\.?\d*)',
        re.IGNORECASE,
    )),
    (RECORD_UPDATE, re.compile(
        r'记一下|帮我记|记住|记录一下|更新.*记录|告诉你.*我'
    )),
    (ALLERGY, re.compile(
        r'过敏|起疹|荨麻疹|脸肿|嘴肿|皮疹|红肿|眼睛肿|瘙痒.*吃药|吃药.*瘙痒'
    )),
    (MEMORY_QUERY, re.compile(
        r'之前.*吃|上次.*药|以前.*说|历史记录|查一下|有没有记|我说过'
    )),
    (MEDICATION_CONSULT, re.compile(
        r'能吃|可以吃|吃.*药|药.*能|剂量|副作用|禁忌|相互作用|换药|停药|怎么服'
    )),
    (LIFESTYLE, re.compile(
        r'怎么改善|怎么调|作息|饮食建议|运动|睡眠.*建议|减肥|减重|生活习惯'
    )),
    (SYMPTOM_REPORT, re.compile(
        r'头痛|头疼|发烧|咳嗽|恶心|呕吐|腹痛|胸痛|失眠|疲劳|乏力|不舒服|不适|症状|难受'
    )),
]

_llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=32)

_CLASSIFY_PROMPT = """将用户消息分类为健康 AI 助手场景下的一个意图，只输出 JSON，不输出其他任何内容。

意图选项（只选一个）：
- symptom_report            症状描述或上报
- medication_consultation   用药咨询（能不能吃、剂量、副作用）
- allergy_or_adverse_reaction 过敏或药物不良反应
- chronic_followup          慢性病随访（汇报血糖、血压等指标数值）
- lifestyle_advice          生活方式建议（睡眠、饮食、运动）
- record_update             主动要求记录信息
- memory_query              查询历史记录
- general                   其他

输出格式（仅此 JSON）：{"intent": "<选项>"}"""


async def classify_intent(message: str) -> str:
    tracer = get_tracer()
    span = tracer.start_span("intent")
    span.set_attribute("intent.msg_len", len(message))

    # 1. 快速规则
    for intent, pattern in _RULES:
        if pattern.search(message):
            logger.debug("[intent] rule → %s  msg=%r", intent, message[:40])
            span.set_attributes({
                "intent.method": "rule",
                "intent.result": intent,
                "intent.rule_matched": intent,
            })
            mark_span_ok(span)
            span.end()
            return intent

    # 2. Haiku 兜底
    t0 = time.monotonic()
    try:
        resp = await _llm.ainvoke([
            SystemMessage(content=_CLASSIFY_PROMPT),
            HumanMessage(content=message),
        ])
        ms = int((time.monotonic() - t0) * 1000)
        raw = resp.content.strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            data = json.loads(m.group())
            intent = data.get("intent", GENERAL)
            if intent not in _ALL:
                intent = GENERAL
            logger.debug("[intent] llm → %s  ms=%d msg=%r", intent, ms, message[:40])
            span.set_attributes({
                "intent.method": "llm",
                "intent.result": intent,
                "intent.llm_ms": ms,
            })
            mark_span_ok(span)
            span.end()
            return intent
    except Exception as exc:
        logger.warning("[intent] classification failed, fallback=general: %s", exc)
        mark_span_error(span, exc)

    span.set_attributes({"intent.method": "fallback", "intent.result": GENERAL})
    span.end()
    return GENERAL
