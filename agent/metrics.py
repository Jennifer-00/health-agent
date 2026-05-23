"""
Metrics — Prometheus 指标定义。

设计原则:
  1. RED 三元组: 每个组件都有 Rate / Errors / Duration
  2. 多维度标签: intent, model, tool, exit 等，支持 Grafana 多维过滤
  3. Agent 专属: TTFT、tool_loop_rounds、critic_failures 是 Agent 特有的
  4. Token 成本: 按 model + direction 拆分，可以按用户/按模型出账单

面试要点:
  - Metrics 回答"系统整体健康吗"——聚合后数据量小，适合实时告警
  - RED 方法论: Rate(请求量) + Errors(错误率) + Duration(延迟分布)
  - Agent 系统的三个黄金指标: TTFT(流式体验)、工具轮次(推理效率)、Critic拦截率(输出质量)

Usage:
  from agent.metrics import (
      pipeline_duration, llm_tokens, tool_calls, critic_failures, ttft
  )

  pipeline_duration.labels(exit="normal").observe(4.2)
  llm_tokens.labels(node="orchestrator", model="gpt-4o", direction="input").inc(3200)
"""

from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST, REGISTRY

# ── HTTP 层 (由 backend middleware 写入) ────────────────────────────────────────

http_requests = Counter(
    "agent_http_requests_total",
    "HTTP 请求总数",
    ["method", "endpoint", "status_code"],
)

http_request_duration = Histogram(
    "agent_http_request_duration_seconds",
    "HTTP 请求耗时",
    ["method", "endpoint"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30],
)

# ── Pipeline 层 (由 harness 写入) ───────────────────────────────────────────────

pipeline_duration = Histogram(
    "agent_pipeline_duration_seconds",
    "端到端 Pipeline 耗时 (从请求进入到最后一个 SSE chunk)",
    ["intent", "exit"],  # exit ∈ {normal, triage_short_circuit, error}
    buckets=[0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30],
)

pipeline_requests = Counter(
    "agent_pipeline_requests_total",
    "Pipeline 请求总数",
    ["intent", "exit"],
)

# ── 首个 Token 时间 (流式体验核心指标) ───────────────────────────────────────────

ttft = Histogram(
    "agent_ttft_seconds",
    "首个 Token 时间 (第一段文本 chunk 到达前)",
    ["intent"],
    buckets=[0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5],
)

# ── LLM 层 (每次 LLM 调用写入) ──────────────────────────────────────────────────

llm_calls = Counter(
    "agent_llm_calls_total",
    "LLM 调用总次数",
    ["node", "model"],  # node ∈ {triage, intent, orchestrator, critic}
)

llm_tokens = Counter(
    "agent_llm_tokens_total",
    "LLM Token 消耗总计 (按 input/output 拆分，便于成本核算)",
    ["node", "model", "direction"],  # direction ∈ {input, output}
)

llm_duration = Histogram(
    "agent_llm_duration_seconds",
    "LLM 单次调用耗时",
    ["node", "model"],
    buckets=[0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5],
)

# ── Tool 层 (每次工具执行写入) ──────────────────────────────────────────────────

tool_calls = Counter(
    "agent_tool_calls_total",
    "工具调用总次数",
    ["tool", "status"],  # status ∈ {success, error}
)

tool_duration = Histogram(
    "agent_tool_duration_seconds",
    "工具单次执行耗时",
    ["tool"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10],
)

tool_result_size = Histogram(
    "agent_tool_result_bytes",
    "工具返回结果大小 (压缩前)",
    ["tool"],
    buckets=[100, 500, 1000, 3000, 5000, 10000, 30000],
)

# ── 质量指标 ────────────────────────────────────────────────────────────────────

triage_emergency = Counter(
    "agent_triage_emergency_total",
    "Triage 紧急介入次数",
    ["intent"],
)

critic_failures = Counter(
    "agent_critic_failures_total",
    "Critic 审查拦截次数 (grounding hallucination + compliance)",
    ["type"],  # type ∈ {grounding, compliance}
)

critic_checks = Counter(
    "agent_critic_checks_total",
    "Critic 审查总数",
    ["type"],
)

redis_degradation = Counter(
    "agent_redis_degradation_total",
    "Redis 不可用导致降级为内存字典的次数",
    ["operation"],  # operation ∈ {session_load, session_save, prefetch}
)

tool_loop_rounds = Histogram(
    "agent_tool_loop_rounds",
    "每次请求的工具调用轮次分布 (正常 0-5)",
    buckets=[0, 1, 2, 3, 4, 5],
)

memory_write_latency = Histogram(
    "agent_memory_write_seconds",
    "Mem0 + Neo4j 并行写入耗时",
    ["trigger"],  # trigger ∈ {n_turns, session_end}
    buckets=[0.1, 0.5, 1, 2, 5, 10],
)

# ── 系统信息 (用于版本标记) ─────────────────────────────────────────────────────

agent_info = Info("agent", "Agent 服务信息")
agent_info.info({
    "version": "0.1.0",
})


# ── 辅助: 批量生成 Prometheus 文本 ──────────────────────────────────────────────

def get_metrics_text() -> bytes:
    """返回 Prometheus 文本格式的指标，供 /metrics 端点使用。"""
    return generate_latest(REGISTRY)
