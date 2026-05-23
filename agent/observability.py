"""
Observability — 本地 JSONL 追踪 (OTEL 的补充，不是替代)。

定位:
  - OTEL (telemetry.py) → 分布式 Trace，上报 Jaeger/Tempo
  - Prometheus (metrics.py) → 聚合指标，Grafana 面板 + 告警
  - 本模块 → 本地 JSONL，开发期 grep 调试 + OTEL 不可用时的 fallback

设计参考 AgentTrace (arxiv 2602.10133) 的三观测面模型：
  - operational : 方法调用层面（节点进入/退出、耗时）
  - cognitive   : LLM 认知层面（prompt tokens、工具请求数）
  - contextual  : 外部系统交互层面（工具执行耗时、结果长度）

三观测面现在体现为 Span 的 "surface" 属性，不再是独立日志流。
每个 Span 可以同时属于多个面——例如 search_memory span：
  - 它的 duration 属于 operational 面
  - 它的 result_len 属于 contextual 面
  - 它触发的 query rewrite LLM 调用属于 cognitive 面

面试要点:
  - 三观测面是概念模型，不是物理存储。物理上按 Span Tree 组织，按面过滤
  - JSONL 是 OTEL 的补充不是对立——开发时 tail -f JSONL，生产时查 Jaeger
"""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "agent_trace.jsonl"


def _ensure_log_dir() -> None:
    _LOG_DIR.mkdir(exist_ok=True)


def _emit(record: dict) -> None:
    """将一条事件追加写入 JSONL 文件，同时打印到 stdout 方便开发期调试。"""
    _ensure_log_dir()
    record.setdefault("ts", datetime.now(timezone.utc).isoformat())
    line = json.dumps(record, ensure_ascii=False)
    with _LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(f"[trace] {line}")


# ── 三个观测面的日志函数 (保留供开发调试 + OTEL fallback) ────────────────────

def log_pipeline_event(trace_id: str, event: str, **kwargs) -> None:
    """operational 面：pipeline 阶段事件（triage/agent/critic 进入与完成）。"""
    _emit({"trace_id": trace_id, "surface": "operational", "event": event, **kwargs})


def log_llm_call(
    trace_id: str,
    node: str,
    tokens_input: int,
    tokens_output: int,
    ms: int,
    tools_requested: list[str],
) -> None:
    """cognitive 面：每次 LLM 调用的 token 消耗与工具请求。"""
    _emit({
        "trace_id": trace_id,
        "surface": "cognitive",
        "event": "llm_call",
        "node": node,
        "tokens_input": tokens_input,
        "tokens_output": tokens_output,
        "ms": ms,
        "tools_requested": tools_requested,
    })


def log_tool_use(
    trace_id: str,
    tool: str,
    ms: int,
    result_len: int,
    mode: str = "sync",
) -> None:
    """contextual 面：工具执行耗时与结果长度。"""
    _emit({
        "trace_id": trace_id,
        "surface": "contextual",
        "event": "tool_use",
        "tool": tool,
        "ms": ms,
        "result_len": result_len,
        "mode": mode,
    })


# ── Trace ID 生成 ──────────────────────────────────────────────────────────────

def new_trace_id() -> str:
    """生成一个新的 trace_id，供 AgentHarness 在请求开始时调用。"""
    return uuid.uuid4().hex[:12]
