"""
Structured logging — structlog + 日志轮换 + 请求上下文注入。

设计原则:
  1. JSON 格式: 每条日志是完整的 JSON 对象，可被 Loki/ELK 直接索引
  2. 上下文传播: trace_id / user_id / session_id 通过 contextvars 自动注入每条日志
  3. 日志轮换: 按天切分，保留 30 天
  4. 兼容现有代码: 现有 logging.getLogger(__name__) 调用不受影响

面试要点:
  - 日志必须结构化(JSON)，否则 grep 解析不了多行异常堆栈
  - request-scoped 字段(trace_id, user_id)不能通过函数参数逐层传，要用 contextvars
  - 日志轮换是必须的——JSONL 无限增长会撑爆磁盘

Usage:
  from agent.logging_config import setup_logging, bind_context, unbind_context

  setup_logging()                          # 在 main.py 启动时调用一次
  bind_context(trace_id="abc", user_id="u1")   # 请求开始时
  ...
  unbind_context()                              # 请求结束时

  # 然后所有模块里直接用标准 logging:
  import logging
  logger = logging.getLogger(__name__)
  logger.info("tool completed", extra={"tool": "search_memory", "ms": 320})
"""

import logging
import logging.handlers
import os
import sys
from contextvars import ContextVar

# ── Request-scoped context ──────────────────────────────────────────────────────

_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)
_user_id: ContextVar[str | None]  = ContextVar("user_id", default=None)
_session_id: ContextVar[str | None] = ContextVar("session_id", default=None)

_initialized = False


def bind_context(trace_id: str = "", user_id: str = "", session_id: str = "") -> None:
    """请求开始时调用，注入 trace_id/user_id/session_id 到当前协程上下文。"""
    if trace_id:
        _trace_id.set(trace_id)
    if user_id:
        _user_id.set(user_id)
    if session_id:
        _session_id.set(session_id)


def unbind_context() -> None:
    """请求结束时调用，清除上下文避免泄漏到下一个请求。"""
    _trace_id.set(None)
    _user_id.set(None)
    _session_id.set(None)


# ── JSON Formatter ──────────────────────────────────────────────────────────────

class _JsonFormatter(logging.Formatter):
    """
    将 LogRecord 格式化为单行 JSON。
    自动从 contextvars 注入 trace_id/user_id/session_id。
    extra 中传入的字段会合并到 JSON 对象中。
    """

    def format(self, record: logging.LogRecord) -> str:
        import json
        from datetime import datetime, timezone

        entry: dict = {
            "ts":       datetime.now(timezone.utc).isoformat(),
            "level":    record.levelname.lower(),
            "logger":   record.name,
            "msg":      record.getMessage(),
        }

        # 注入 request 上下文
        trace_id = _trace_id.get()
        if trace_id:
            entry["trace_id"] = trace_id
        user_id = _user_id.get()
        if user_id:
            entry["user_id"] = user_id
        session_id = _session_id.get()
        if session_id:
            entry["session_id"] = session_id

        # 合并 extra 中的自定义字段
        for key in ("tool", "node", "model", "ms", "round", "intent",
                    "result_len", "llm_len", "is_error", "cache_hit",
                    "query_hash", "span_id", "surface", "error"):
            val = getattr(record, key, None)
            if val is not None:
                entry[key] = val

        # 异常信息
        if record.exc_info and record.exc_info[1]:
            entry["exception"] = str(record.exc_info[1])
        if record.stack_info:
            entry["stack_info"] = record.stack_info

        return json.dumps(entry, ensure_ascii=False, default=str)


# ── Setup ───────────────────────────────────────────────────────────────────────

def setup_logging(
    log_dir: str = "logs",
    level: int = logging.INFO,
    file_retention: int = 30,
) -> None:
    """
    初始化结构化日志系统。安全多次调用（幂等）。

    配置:
      - 控制台: 开发环境输出人类可读，生产环境输出 JSON
      - 文件:   JSON 格式，按天轮换，保留 file_retention 天
      - 级别:   默认 INFO，可通过 LOG_LEVEL 环境变量覆盖
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    env = os.getenv("ENV", "dev")
    log_level = getattr(logging, os.getenv("LOG_LEVEL", "").upper(), level)

    root = logging.getLogger()
    root.setLevel(log_level)

    # 清除已有 handler (防止 uvicorn 重复添加)
    root.handlers.clear()

    # ── 控制台 handler ──────────────────────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(log_level)
    if env == "dev":
        # 开发环境: 人类可读格式
        console.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        ))
    else:
        console.setFormatter(_JsonFormatter())
    root.addHandler(console)

    # ── 文件 handler (JSON, 按天轮换) ───────────────────────────────────────
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=os.path.join(log_dir, "agent.log"),
        when="midnight",
        interval=1,
        backupCount=file_retention,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(_JsonFormatter())
    root.addHandler(file_handler)

    # ── 第三方库日志级别 ────────────────────────────────────────────────────
    for noisy in ("uvicorn.access", "httpx", "openai", "urllib3", "apscheduler"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root.info("[logging] structured logging initialized (env=%s, retention=%dd)", env, file_retention)
