"""
Observability — Harness 的结构化日志层。

设计参考 AgentTrace (arxiv 2602.10133) 的三观测面模型：
  - operational : 方法调用层面（节点进入/退出、耗时）
  - cognitive   : LLM 认知层面（prompt tokens、工具请求数）
  - contextual  : 外部系统交互层面（工具执行耗时、结果长度）

所有事件写入 JSONL，字段统一：
  trace_id / surface / event / node / ms / ...
trace_id 由 AgentHarness 在每次请求开始时生成，
通过装饰器参数注入，使 triage → graph → critic 三层日志可关联查询。
"""
import functools
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

# 日志文件路径，可通过环境变量覆盖
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


# ── 三个观测面的直接日志函数 ─────────────────────────────────────────────────

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
    mode: str = "sync",  # "sync" | "background"
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


# ── 装饰器：非侵入式包装 LangGraph 节点 ──────────────────────────────────────
# 参考 AgentTrace 的 decorator injection pattern：
# 在 harness 层 wrap 节点函数，不修改 nodes.py 内部代码。

def trace_llm_node(node_name: str, trace_id_getter: Callable[[], str]):
    """
    装饰 orchestrator_node 等 LLM 节点。
    自动打 cognitive 面日志：耗时、token 数、工具请求列表。
    """
    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(state, *args, **kwargs):
            t0 = time.monotonic()
            result = await fn(state, *args, **kwargs)
            ms = int((time.monotonic() - t0) * 1000)

            trace_id = trace_id_getter()
            # 从节点输出里提取 token 使用量（LangChain AIMessage 携带 usage_metadata）
            tokens_input, tokens_output = 0, 0
            tools_requested: list[str] = []

            messages = result.get("messages", []) if isinstance(result, dict) else []
            if messages:
                last = messages[-1]
                meta = getattr(last, "usage_metadata", None)
                if meta:
                    tokens_input = meta.get("input_tokens", 0)
                    tokens_output = meta.get("output_tokens", 0)
                tool_calls = getattr(last, "tool_calls", None) or []
                tools_requested = [tc["name"] for tc in tool_calls]

            log_llm_call(
                trace_id=trace_id,
                node=node_name,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                ms=ms,
                tools_requested=tools_requested,
            )
            return result
        return wrapper
    return decorator


def trace_tool_node(trace_id_getter: Callable[[], str]):
    """
    装饰 tool_executor_node。
    自动打 contextual 面日志：每个工具的耗时与结果长度。
    """
    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(state, *args, **kwargs):
            t0 = time.monotonic()
            result = await fn(state, *args, **kwargs)
            ms = int((time.monotonic() - t0) * 1000)

            return result
        return wrapper
    return decorator


def new_trace_id() -> str:
    """生成一个新的 trace_id，供 AgentHarness 在请求开始时调用。"""
    return uuid.uuid4().hex[:12]
