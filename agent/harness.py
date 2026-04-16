"""
AgentHarness — Pipeline 编排层。

将 Triage → Main Agent → Critic 三层显式串联，
并提供 Hook 注册点供横切关注点（日志、Zep 写入等）接入。

Pipeline 流程（参考 Analytics Vidhya Framework/Runtime/Harness 三层模型）：
  用户消息
      │
      ▼
  [L1] Triage Gate      ← input guard，Haiku 快速判定，紧急则短路返回
      │ 通过
      ▼
  pre_llm_hooks         ← 注入上下文（日期已在 orchestrator_node 内部处理）
      │
      ▼
  [L2] Main Agent Graph ← LangGraph Runtime，工具权限在 nodes.py 内部管理
      │ 生成回复
      ▼
  post_tool_hooks       ← observability 装饰器在 graph 构建时已注入，此处可扩展
      │
      ▼
  [L3] Critic Review    ← output guard，质检回复合规性
      │
      ▼
  on_stop_hooks         ← fire-and-forget：Zep 写入、session 持久化
      │
      ▼
  最终回复（AsyncGenerator[str, None] → SSE chunks）

设计原则（参考 Anatomy of an Agent Harness）：
  - "薄控制、厚横切"：不干预 LangGraph 内部 ReAct 循环，专注 guardrails + observability
  - Harness 本身是 AsyncGenerator，chat.py 只做 transport（HTTP/SSE 转换）
"""
import asyncio
from collections.abc import AsyncGenerator
from typing import Callable, Awaitable

from langchain_core.messages import AIMessage

from agent.critic import critic_review
from agent.graph import build_graph
from agent.observability import (
    log_llm_call,
    log_pipeline_event,
    log_tool_use,
    new_trace_id,
)
from agent.triage import triage
from memory.session_buffer import SessionBuffer

# ── Hook 类型别名 ──────────────────────────────────────────────────────────────
# pre_llm_hook  : (trace_id, user_message, user_id) -> None
# post_tool_hook: (trace_id, tool_events) -> None
# on_stop_hook  : (trace_id, user_message, assistant_text, user_id, session_id) -> Awaitable
PreLLMHook   = Callable[[str, str, str], None]
PostToolHook = Callable[[str, list[dict]], None]
OnStopHook   = Callable[[str, str, str, str, str], Awaitable[None]]

# 工具名 → 前端状态提示文案（从 chat.py 迁移到 harness，属于业务逻辑）
TOOL_STATUS: dict[str, str] = {
    "memory_search":      "正在查询历史记录…",
    "memory_write":       "正在写入记忆…",
    "record_link":        "正在检测反复发作…",
    "memory_consolidate": "正在整理记忆…",
    "summary_gen":        "正在生成摘要…",
}

CHUNK_SIZE = 24


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item if isinstance(item, str)
            else item.get("text", "") if isinstance(item, dict) and item.get("type") == "text"
            else ""
            for item in content
        )
    return str(content)


# ── 内置 post_tool_hook：反复发作检测告警 ─────────────────────────────────────
# 职责划分：
#   - 这里（harness 层）：检测"record_link 有没有发现关联历史"，记录 operational 日志
#   - reply_node（graph 层）：确保最终回复文本里提到了发作历史（纯回复质量兜底）
# 两者各管一件事，互不干扰。

def _recurrence_alert_hook(trace_id: str, tool_events: list[dict]) -> None:
    """
    post_tool_hook：record_link 检测到反复发作时打告警日志。
    将来可在此扩展：写告警队列、发推送通知等，不需要改 reply_node。
    """
    recurrence_events = [
        e for e in tool_events
        if e.get("tool") == "record_link" and "关联历史" in e.get("summary", "")
    ]
    if not recurrence_events:
        return

    count = len(recurrence_events)
    summaries = [e["summary"] for e in recurrence_events]
    log_pipeline_event(
        trace_id,
        "recurrence_detected",
        count=count,
        summaries=summaries,
    )


class AgentHarness:
    """
    Harness 主类。实例化一次后可复用（graph 在首次 run 时懒加载）。

    Hook 列表对外暴露，调用方可在实例化后直接 append：
        harness.on_stop_hooks.append(zep_write_hook)
    """

    def __init__(self) -> None:
        self.pre_llm_hooks:   list[PreLLMHook]   = []
        self.post_tool_hooks: list[PostToolHook]  = [_recurrence_alert_hook]
        self.on_stop_hooks:   list[OnStopHook]    = []
        self._graph = None          # 懒加载，避免模块导入时就初始化 LLM

    # ── 内部：graph 懒加载 ────────────────────────────────────────────────────
    def _get_graph(self):
        """
        编译后的 CompiledStateGraph 是无状态的，模块生命周期内复用一个实例。
        Observability 通过读取 astream 输出（AIMessage.usage_metadata）实现，
        不修改 graph 节点内部——这是真正意义上的非侵入式 instrumentation。
        """
        if self._graph is None:
            self._graph = build_graph()
        return self._graph

    # ── 主入口：pipeline run ───────────────────────────────────────────────────
    async def run(
        self,
        user_message: str,
        user_id: str,
        session_id: str,
        history: list[dict],
        original_user_content: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        执行完整 pipeline，yield SSE JSON 字符串。
        chat.py 只需 async for chunk in harness.run(...): yield chunk
        """
        return self._run_pipeline(
            user_message, user_id, session_id, history, original_user_content
        )

    async def _run_pipeline(
        self,
        user_message: str,
        user_id: str,
        session_id: str,
        history: list[dict],
        original_user_content: str | None,
    ) -> AsyncGenerator[str, None]:
        trace_id = new_trace_id()
        log_pipeline_event(trace_id, "pipeline_start",
                           user_id=user_id, session_id=session_id)

        # ── L1: Triage Gate ───────────────────────────────────────────────────
        log_pipeline_event(trace_id, "triage_start")
        is_emergency, emergency_response = await triage(user_message)
        log_pipeline_event(trace_id, "triage_done", emergency=is_emergency)

        if is_emergency:
            # 专用事件类型，前端据此弹模态框而不是走普通消息流
            yield _sse({"type": "emergency", "text": emergency_response})
            yield _sse({"type": "done"})
            log_pipeline_event(trace_id, "pipeline_end", exit="triage_short_circuit")
            # 紧急症状也存档，方便后续 record_link 检测反复发作
            save_content = original_user_content or user_message
            for hook in self.on_stop_hooks:
                asyncio.create_task(
                    _safe_hook(hook, trace_id, save_content, emergency_response, user_id, session_id)
                )
            return

        # ── pre_llm_hooks ─────────────────────────────────────────────────────
        for hook in self.pre_llm_hooks:
            try:
                hook(trace_id, user_message, user_id)
            except Exception as exc:
                log_pipeline_event(trace_id, "hook_error", hook="pre_llm", error=str(exc))

        # ── L2: Main Agent Graph ──────────────────────────────────────────────
        log_pipeline_event(trace_id, "agent_start")
        graph = self._get_graph()
        conversation = history + [{"role": "user", "content": user_message}]

        # Context 层：检查 memory_search 冷却状态
        session_buf = SessionBuffer(session_id=session_id)
        cooldown = await session_buf.get_mem_search_cooldown()
        skip_memory_search = cooldown > 0
        if skip_memory_search:
            await session_buf.decrement_mem_search_cooldown()
            log_pipeline_event(trace_id, "mem_search_cooldown", remaining=cooldown)

        assistant_text = ""
        all_tool_events: list[dict] = []
        mem_search_requested = False  # 追踪本轮是否有 memory_search 被 LLM 请求

        async for chunk in graph.astream({
            "messages": conversation,
            "user_id": user_id,
            "pending_memories": [],
            "tool_events": [],
            "skip_memory_search": skip_memory_search,
        }):
            for node_name, node_output in chunk.items():
                if node_output is None:
                    continue

                if node_name == "orchestrator":
                    pending  = node_output.get("pending_memories", [])
                    messages = node_output.get("messages", [])

                    # 冷却硬拦截：LLM 请求了 memory_search 但冷却仍活跃，直接过滤掉
                    # 软提示（skip_memory_search 注入 prompt）不够可靠，LLM 会无视
                    if skip_memory_search:
                        pending = [c for c in pending if c.get("name") != "memory_search"]
                        node_output["pending_memories"] = pending

                    # memory_search 是读操作，不进 tool_events，在此处从 pending 里检测
                    if any(c.get("name") == "memory_search" for c in pending):
                        mem_search_requested = True

                    for call in pending:
                        status = TOOL_STATUS.get(call.get("name", ""), f"正在执行 {call.get('name')}…")
                        yield _sse({"type": "status", "text": status})

                    if messages:
                        last = messages[-1]
                        if isinstance(last, AIMessage) and not last.tool_calls:
                            assistant_text = _extract_text(last.content)

                        # cognitive 面：从 AIMessage.usage_metadata 提取 token 数
                        # 非侵入式——直接读 astream 输出，不 wrap graph 节点
                        meta = getattr(last, "usage_metadata", None) or {}
                        tools_req = [tc["name"] for tc in (getattr(last, "tool_calls", None) or [])]
                        log_llm_call(
                            trace_id=trace_id,
                            node="orchestrator",
                            tokens_input=meta.get("input_tokens", 0),
                            tokens_output=meta.get("output_tokens", 0),
                            ms=0,   # astream 模式下单节点耗时不可直接获取，记 0
                            tools_requested=tools_req,
                        )

                elif node_name == "tool_executor":
                    events = node_output.get("tool_events", [])
                    all_tool_events.extend(events)

                    # memory_search 从 orchestrator pending 检测（读操作不进 tool_events）
                    # 在 tool_executor 完成后统一重置冷却，确保工具已实际执行
                    if mem_search_requested:
                        mem_search_requested = False  # 消费掉，防止多次 tool_executor 重复触发
                        await session_buf.set_mem_search_cooldown()
                        log_pipeline_event(trace_id, "mem_search_cooldown_reset")

                    for event in events:
                        yield _sse({"type": "tool_call", **event})
                        # contextual 面：逐工具记录
                        mode = "background" if event.get("summary") == "已在后台写入" else "sync"
                        log_tool_use(
                            trace_id=trace_id,
                            tool=event.get("tool", "unknown"),
                            ms=0,
                            result_len=len(str(event.get("summary", ""))),
                            mode=mode,
                        )

                    # post_tool_hooks
                    for hook in self.post_tool_hooks:
                        try:
                            hook(trace_id, events)
                        except Exception as exc:
                            log_pipeline_event(trace_id, "hook_error",
                                               hook="post_tool", error=str(exc))

                    yield _sse({"type": "status", "text": "正在生成回复…"})

                elif node_name == "reply":
                    messages = node_output.get("messages", [])
                    if messages:
                        last = messages[-1]
                        if isinstance(last, AIMessage):
                            assistant_text = _extract_text(last.content)

        log_pipeline_event(trace_id, "agent_done",
                           tool_count=len(all_tool_events),
                           reply_len=len(assistant_text))

        # ── L3: Critic Review ─────────────────────────────────────────────────
        log_pipeline_event(trace_id, "critic_start")
        note = await critic_review(user_message, assistant_text)
        log_pipeline_event(trace_id, "critic_done", passed=(note is None))

        if note:
            assistant_text = f"{assistant_text}\n\n{note}"

        # ── 流出最终回复 ──────────────────────────────────────────────────────
        for start in range(0, len(assistant_text), CHUNK_SIZE):
            yield _sse({"type": "text", "delta": assistant_text[start:start + CHUNK_SIZE]})
        yield _sse({"type": "done"})

        # ── on_stop_hooks（fire-and-forget，Zep 写入等副作用在此注册）─────────
        save_content = original_user_content or user_message
        for hook in self.on_stop_hooks:
            asyncio.create_task(
                _safe_hook(hook, trace_id, save_content, assistant_text, user_id, session_id)
            )

        log_pipeline_event(trace_id, "pipeline_end", exit="normal")


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _sse(payload: dict) -> str:
    import json
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _safe_hook(hook: OnStopHook, *args) -> None:
    """包一层 try/except，hook 报错不影响已完成的回复。"""
    try:
        await hook(*args)
    except Exception as exc:
        # on_stop_hook 失败只记录，不抛出
        print(f"[harness] on_stop_hook error: {exc}")
