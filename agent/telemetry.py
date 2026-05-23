"""
Telemetry — OpenTelemetry 初始化 + Span 辅助工具。

设计原则:
  1. 零配置可用: 无 OTEL_EXPORTER 时走 noop，业务代码不需要 if/else
  2. 按需导出: 配置 OTEL_EXPORTER_OTLP_ENDPOINT 后自动上报 Jaeger/Tempo
  3. 标准属性: 遵循 OpenTelemetry Semantic Conventions for LLM
  4. 火而忘技: 通过 SpanContext 注入让异步 hooks 链回主 trace

面试要点:
  - Agent 的 Trace 必须是 Span Tree，不是扁平事件流
  - 每个 LLM 调用、每个工具执行、每个 pipeline 阶段都是独立 Span
  - Span 之间通过父子关系表达调用链，火焰图直接定位瓶颈
  - fire-and-forget hooks 通过 link 机制关联，不丢失异步后处理的可观测性

Usage:
  from agent.telemetry import get_tracer, trace_llm, trace_tool, link_fire_and_forget

  tracer = get_tracer()
  with tracer.start_as_current_span("triage") as span:
      span.set_attribute("llm.model", "gpt-4o-mini")
      result = await triage(message)
"""

import os
import functools
from contextlib import contextmanager, asynccontextmanager
from typing import Any, Callable

# ── OpenTelemetry imports ──────────────────────────────────────────────────────
# 所有导入包裹在 try/except 中，依赖缺失时自动降级为 noop

_OTEL_AVAILABLE = False
try:
    from opentelemetry import trace, context
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    from opentelemetry.trace import (
        SpanKind,
        Status,
        StatusCode,
        SpanContext,
        TraceFlags,
        NonRecordingSpan,
        Link,
    )
    from opentelemetry.trace.propagation import TraceContextTextMapPropagator
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    _OTEL_AVAILABLE = True
except ImportError:
    pass

# ── 服务标识 ────────────────────────────────────────────────────────────────────

_SERVICE = os.getenv("OTEL_SERVICE_NAME", "health-agent")
_ENV     = os.getenv("ENV", "dev")

_tracer: Any = None
_provider: Any = None


def _build_exporter():
    """根据环境变量决定导出目标，都没配则只打 stdout。"""
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    if otlp_endpoint:
        return OTLPSpanExporter(endpoint=otlp_endpoint)
    # 开发环境用 ConsoleExporter 方便 debug
    if _ENV == "dev":
        return ConsoleSpanExporter()
    # 生产环境如果没配 collector，至少不丢数据 — 由调用方决定
    return ConsoleSpanExporter()


def _build_resource() -> Resource:
    return Resource.create({SERVICE_NAME: _SERVICE, "deployment.environment": _ENV})


# ── 初始化 (在 backend/main.py lifespan 中调用) ─────────────────────────────────

def init_telemetry() -> None:
    """初始化 OTEL SDK。缺依赖或已初始化时安全跳过。"""
    global _tracer, _provider
    if _tracer is not None:
        return
    if not _OTEL_AVAILABLE:
        _tracer = _NoopTracer()
        return
    try:
        exporter = _build_exporter()
        _provider = TracerProvider(resource=_build_resource())
        _provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(_provider)
        _tracer = trace.get_tracer(_SERVICE)
    except Exception:
        _tracer = _NoopTracer()


def get_tracer():
    """获取 tracer。未初始化时自动创建 noop。"""
    global _tracer
    if _tracer is None:
        _tracer = _NoopTracer()
    return _tracer


# ── Noop 降级 (OTEL SDK 不可用时仍保证业务代码正常运行) ──────────────────────────

class _NoopSpan:
    """No-op span that accepts all method calls silently."""
    def __init__(self, *args, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *args): pass
    def set_attribute(self, key, value): pass
    def set_attributes(self, attrs): pass
    def set_status(self, status, description=""): pass
    def record_exception(self, exception): pass
    def add_event(self, name, attributes=None): pass
    def end(self, end_time=None): pass
    def get_span_context(self):
        from random import randint
        class _FakeCtx:
            trace_id = randint(1, 2**63)
            span_id = randint(1, 2**63)
            trace_flags = type("F", (), {"sampled": True})()
            is_remote = False
            is_valid = True
        return _FakeCtx()


class _NoopTracer:
    def start_as_current_span(self, name, **kwargs):
        return _NoopSpan()
    def start_span(self, name, **kwargs):
        return _NoopSpan()


# ── 装饰器: 非侵入式 Span 包裹 ──────────────────────────────────────────────────

def traced(name: str | None = None, kind: str = "internal", attrs: dict | None = None):
    """
    装饰器，自动将函数调用包裹为 Span。
    支持同步和异步函数，自动从函数名推断 span name。

    面试要点:
      装饰器模式实现非侵入式埋点，业务代码不需要感知 OTEL。
      LangGraph 节点用这个 wrap 后，每个节点调用自动在 Jaeger 中显示为独立 Span。

    Usage:
      @traced("orchestrator_node")
      async def orchestrator_node(state): ...
    """
    def decorator(fn):
        span_name = name or fn.__name__
        _kind = {"llm": SpanKind.CLIENT, "tool": SpanKind.CLIENT}.get(kind, SpanKind.INTERNAL) if _OTEL_AVAILABLE else None

        if functools.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def wrapper(*args, **kwargs):
                tracer = get_tracer()
                with tracer.start_as_current_span(span_name, kind=_kind) as span:
                    if attrs:
                        span.set_attributes(attrs)
                    try:
                        result = await fn(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as exc:
                        span.record_exception(exc)
                        span.set_status(Status(StatusCode.ERROR, str(exc)))
                        raise
            return wrapper
        else:
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                tracer = get_tracer()
                with tracer.start_as_current_span(span_name, kind=_kind) as span:
                    if attrs:
                        span.set_attributes(attrs)
                    try:
                        result = fn(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as exc:
                        span.record_exception(exc)
                        span.set_status(Status(StatusCode.ERROR, str(exc)))
                        raise
            return wrapper
    return decorator


# ── 辅助: 从 LLM 响应中提取标准属性 ──────────────────────────────────────────────

def extract_llm_attributes(response) -> dict[str, Any]:
    """
    从 LangChain AIMessage 中提取标准 OTEL LLM 属性。

    面试要点:
      自动提取而非手动填写，避免遗漏，保持一致性。
      OTEL 定义了 gen_ai.* 语义约定，所有 LLM 监控工具都认识这些字段。
    """
    attrs = {}
    meta = getattr(response, "usage_metadata", None) or {}
    if meta:
        attrs["gen_ai.usage.input_tokens"] = meta.get("input_tokens", 0)
        attrs["gen_ai.usage.output_tokens"] = meta.get("output_tokens", 0)
    tool_calls = getattr(response, "tool_calls", None) or []
    if tool_calls:
        attrs["gen_ai.tools.requested"] = [tc["name"] for tc in tool_calls]
        attrs["gen_ai.tools.count"] = len(tool_calls)
    return attrs


# ── Fire-and-forget Span 链接 ───────────────────────────────────────────────────

# ── Span 状态辅助 (避免业务代码直接 import StatusCode) ──────────────────────────

_SPAN_OK = None
_SPAN_ERROR = None
_STATUS_CODE = None


def _resolve_status_constants():
    """惰性解析 OTEL 状态常量，未安装时返回 None。"""
    global _SPAN_OK, _SPAN_ERROR, _STATUS_CODE
    if _SPAN_OK is not None:
        return
    if _OTEL_AVAILABLE:
        try:
            from opentelemetry.trace import Status, StatusCode
            _STATUS_CODE = StatusCode
            _SPAN_OK = Status(StatusCode.OK)
            _SPAN_ERROR = Status(StatusCode.ERROR)
        except ImportError:
            _SPAN_OK = False
            _SPAN_ERROR = False
    else:
        _SPAN_OK = False
        _SPAN_ERROR = False


def mark_span_ok(span) -> None:
    """安全设置 Span 状态为 OK。noop span 时静默跳过。"""
    _resolve_status_constants()
    if _SPAN_OK:
        span.set_status(_SPAN_OK)


def mark_span_error(span, exc: Exception | None = None) -> None:
    """安全设置 Span 状态为 ERROR，可选记录异常。noop span 时静默跳过。"""
    _resolve_status_constants()
    if _SPAN_ERROR:
        if exc:
            span.record_exception(exc)
        span.set_status(_SPAN_ERROR)


def current_span_context() -> dict:
    """导出当前 Span 的 W3C TraceContext，注入到异步 task 中。"""
    if not _OTEL_AVAILABLE:
        return {}
    prop = TraceContextTextMapPropagator()
    carrier = {}
    prop.inject(carrier)
    return carrier


def span_from_context(carrier: dict, name: str):
    """
    从 carrier 恢复上下文并创建新 Span（作为原 trace 的子节点）。
    用于 fire-and-forget hooks 链回主 trace。

    Usage:
      ctx = current_span_context()
      task = asyncio.create_task(_fire_and_forget(ctx, ...))

      async def _fire_and_forget(ctx, ...):
          with span_from_context(ctx, "memory_write") as span: ...
    """
    if not _OTEL_AVAILABLE or not carrier:
        return _NoopSpan()
    prop = TraceContextTextMapPropagator()
    ctx = prop.extract(carrier)
    tracer = get_tracer()
    return tracer.start_as_current_span(name, context=ctx)
