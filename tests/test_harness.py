"""
AgentHarness 三层 pipeline 测试。

每个测试只 mock 当前层关心的依赖，其余层用简单 fake 替代。
测试粒度：
  - test_triage_*        → L1 Input Guard
  - test_critic_*        → L3 Output Guard
  - test_on_stop_hooks_* → on_stop_hooks 注册与调用
  - test_observability_* → JSONL 日志输出
"""
import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from agent.harness import AgentHarness


# ── FakeGraph：实现 astream，模拟 LangGraph CompiledStateGraph ────────────────

class FakeGraph:
    """
    harness 只调用 graph.astream()，FakeGraph 直接 yield orchestrator 输出。
    AIMessage.usage_metadata 模拟真实 token 计数（供 observability 读取）。
    """
    def __init__(self, reply: str = "这是测试回复", tool_events: list[dict] | None = None):
        self.reply = reply
        self.tool_events = tool_events or []

    async def astream(self, state: dict):
        msg = AIMessage(content=self.reply)
        msg.tool_calls = []
        msg.usage_metadata = {"input_tokens": 100, "output_tokens": 50}
        yield {"orchestrator": {"messages": [msg], "pending_memories": []}}

        if self.tool_events:
            yield {"tool_executor": {"tool_events": self.tool_events}}


# ── 辅助：消费 harness.run() 的 AsyncGenerator，收集所有 SSE 文本 ─────────────

async def collect(harness: AgentHarness, **kwargs) -> str:
    """运行 harness 并把所有 SSE chunks 拼成一个字符串。"""
    chunks = []
    async for chunk in await harness.run(
        user_message=kwargs.get("user_message", "头痛"),
        user_id=kwargs.get("user_id", "u-test"),
        session_id=kwargs.get("session_id", "s-test"),
        history=kwargs.get("history", []),
        original_user_content=kwargs.get("original_user_content"),
    ):
        chunks.append(chunk)
    # 让 on_stop_hook 后台任务有机会执行
    await asyncio.sleep(0)
    return "".join(chunks)


# ── L1: Triage Gate ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_triage_short_circuits_on_emergency():
    """triage 判定为紧急时，pipeline 立即返回急救提示，不进入主 Agent。"""
    harness = AgentHarness()

    with patch("agent.harness.triage", new=AsyncMock(return_value=(True, "请立即拨打120"))):
        with patch("agent.harness.build_graph") as mock_build:
            output = await collect(harness, user_message="我胸口剧痛，喘不上气")

    # graph 不应被构建（说明 triage 短路生效）
    mock_build.assert_not_called()
    assert "120" in output


@pytest.mark.asyncio
async def test_triage_pass_lets_agent_run():
    """triage 判定为非紧急时，流程正常进入主 Agent。"""
    harness = AgentHarness()
    fake_graph = FakeGraph(reply="普通回复")

    with patch("agent.harness.triage", new=AsyncMock(return_value=(False, ""))):
        with patch("agent.harness.critic_review", new=AsyncMock(return_value=None)):
            with patch("agent.harness.build_graph", return_value=fake_graph):
                output = await collect(harness, user_message="我最近睡眠不好")

    assert "普通回复" in output


# ── L3: Critic Review ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_critic_appends_note_when_fails():
    """critic 审查不通过时，补充说明被追加到回复末尾。"""
    harness = AgentHarness()
    fake_graph = FakeGraph(reply="你得了高血压。")

    with patch("agent.harness.triage", new=AsyncMock(return_value=(False, ""))):
        with patch("agent.harness.critic_review", new=AsyncMock(return_value="建议您就医进行专业诊断。")):
            with patch("agent.harness.build_graph", return_value=fake_graph):
                output = await collect(harness)

    # 原始回复和补充说明都应出现
    assert "你得了高血压" in output
    assert "建议您就医" in output


@pytest.mark.asyncio
async def test_critic_pass_no_extra_content():
    """critic 审查通过时，回复不被修改。"""
    harness = AgentHarness()
    fake_graph = FakeGraph(reply="建议多喝水，注意休息。")

    with patch("agent.harness.triage", new=AsyncMock(return_value=(False, ""))):
        with patch("agent.harness.critic_review", new=AsyncMock(return_value=None)):
            with patch("agent.harness.build_graph", return_value=fake_graph):
                output = await collect(harness)

    assert "多喝水" in output
    # 没有额外的补充说明（只有原文）
    assert output.count("建议") == 1


# ── on_stop_hooks ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_on_stop_hook_called_with_correct_args():
    """on_stop_hook 在回复生成后被调用，且拿到正确的 user/assistant 内容。"""
    harness = AgentHarness()
    fake_graph = FakeGraph(reply="测试回复内容")
    received: list[dict] = []

    async def capture_hook(trace_id, user_content, assistant_text, user_id, session_id):
        received.append({
            "user_content":    user_content,
            "assistant_text":  assistant_text,
            "user_id":         user_id,
            "session_id":      session_id,
        })

    harness.on_stop_hooks.append(capture_hook)

    with patch("agent.harness.triage", new=AsyncMock(return_value=(False, ""))):
        with patch("agent.harness.critic_review", new=AsyncMock(return_value=None)):
            with patch("agent.harness.build_graph", return_value=fake_graph):
                await collect(
                    harness,
                    user_message="今天头很痛",
                    user_id="u-123",
                    session_id="s-456",
                )

    assert len(received) == 1
    assert received[0]["user_content"]   == "今天头很痛"
    assert received[0]["assistant_text"] == "测试回复内容"
    assert received[0]["user_id"]        == "u-123"
    assert received[0]["session_id"]     == "s-456"


@pytest.mark.asyncio
async def test_on_stop_hook_called_on_emergency():
    """triage 短路时 on_stop_hook 也应被调用，紧急症状存档。"""
    harness = AgentHarness()
    received: list[dict] = []

    async def capture_hook(trace_id, user_content, assistant_text, user_id, session_id):
        received.append({"user_content": user_content, "assistant_text": assistant_text})

    harness.on_stop_hooks.append(capture_hook)

    with patch("agent.harness.triage", new=AsyncMock(return_value=(True, "请立即拨打120"))):
        with patch("agent.harness.build_graph") as mock_build:
            await collect(harness, user_message="我胸口剧痛")

    mock_build.assert_not_called()          # 主 Agent 没有运行
    assert len(received) == 1               # hook 被调用了
    assert received[0]["user_content"]  == "我胸口剧痛"
    assert received[0]["assistant_text"] == "请立即拨打120"


@pytest.mark.asyncio
async def test_on_stop_hook_error_does_not_break_response():
    """on_stop_hook 报错不影响已完成的回复。"""
    harness = AgentHarness()
    fake_graph = FakeGraph(reply="正常回复")

    async def bad_hook(*args):
        raise RuntimeError("Zep 写入失败")

    harness.on_stop_hooks.append(bad_hook)

    with patch("agent.harness.triage", new=AsyncMock(return_value=(False, ""))):
        with patch("agent.harness.critic_review", new=AsyncMock(return_value=None)):
            with patch("agent.harness.build_graph", return_value=fake_graph):
                output = await collect(harness)

    # hook 报错，但回复正常
    assert "正常回复" in output


# ── post_tool_hooks: recurrence_alert ────────────────────────────────────────

@pytest.mark.asyncio
async def test_recurrence_hook_fires_when_record_link_finds_history(tmp_path, monkeypatch):
    """record_link 发现关联历史时，post_tool_hook 打 recurrence_detected 日志。"""
    log_file = tmp_path / "agent_trace.jsonl"
    monkeypatch.setattr("agent.observability._LOG_DIR",  tmp_path)
    monkeypatch.setattr("agent.observability._LOG_FILE", log_file)

    harness = AgentHarness()
    fake_graph = FakeGraph(
        reply="您上次也有类似症状。",
        tool_events=[{"tool": "record_link", "summary": "发现关联历史记录：- 头痛 (2026-04-09)"}],
    )

    with patch("agent.harness.triage", new=AsyncMock(return_value=(False, ""))):
        with patch("agent.harness.critic_review", new=AsyncMock(return_value=None)):
            with patch("agent.harness.build_graph", return_value=fake_graph):
                await collect(harness)

    lines = [json.loads(l) for l in log_file.read_text(encoding="utf-8").splitlines() if l]
    alert = next((e for e in lines if e["event"] == "recurrence_detected"), None)
    assert alert is not None
    assert alert["count"] == 1


@pytest.mark.asyncio
async def test_recurrence_hook_silent_when_no_history(tmp_path, monkeypatch):
    """record_link 未找到历史时，不打 recurrence_detected 日志。"""
    log_file = tmp_path / "agent_trace.jsonl"
    monkeypatch.setattr("agent.observability._LOG_DIR",  tmp_path)
    monkeypatch.setattr("agent.observability._LOG_FILE", log_file)

    harness = AgentHarness()
    fake_graph = FakeGraph(
        reply="没有历史记录。",
        tool_events=[{"tool": "record_link", "summary": "未发现相关历史记录。"}],
    )

    with patch("agent.harness.triage", new=AsyncMock(return_value=(False, ""))):
        with patch("agent.harness.critic_review", new=AsyncMock(return_value=None)):
            with patch("agent.harness.build_graph", return_value=fake_graph):
                await collect(harness)

    lines = [json.loads(l) for l in log_file.read_text(encoding="utf-8").splitlines() if l]
    assert not any(e["event"] == "recurrence_detected" for e in lines)


# ── Observability ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_observability_emits_jsonl(tmp_path, monkeypatch):
    """每次 run 都应写入 JSONL，且包含 trace_id 和三个观测面的事件。"""
    log_file = tmp_path / "agent_trace.jsonl"
    monkeypatch.setattr("agent.observability._LOG_DIR",  tmp_path)
    monkeypatch.setattr("agent.observability._LOG_FILE", log_file)

    harness = AgentHarness()
    fake_graph = FakeGraph(reply="观测测试回复")

    with patch("agent.harness.triage", new=AsyncMock(return_value=(False, ""))):
        with patch("agent.harness.critic_review", new=AsyncMock(return_value=None)):
            with patch("agent.harness.build_graph", return_value=fake_graph):
                await collect(harness)

    lines = [json.loads(l) for l in log_file.read_text(encoding="utf-8").splitlines() if l]
    surfaces = {l["surface"] for l in lines}
    events   = {l["event"]   for l in lines}
    trace_ids = {l["trace_id"] for l in lines}

    # 所有日志属于同一个 trace
    assert len(trace_ids) == 1

    # 三个观测面都有记录
    assert "operational" in surfaces
    assert "cognitive"   in surfaces

    # pipeline 阶段事件齐全
    assert "pipeline_start" in events
    assert "triage_done"    in events
    assert "agent_done"     in events
    assert "critic_done"    in events
    assert "pipeline_end"   in events
