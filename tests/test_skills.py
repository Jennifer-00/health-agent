"""skill 单元测试"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_generate_report_returns_markdown():
    mock_block = MagicMock()
    mock_block.text = "# 健康摘要报告\n\n## 症状记录\n- 头痛持续两天"

    with patch("agent.skills.report_gen.scripts.run.Mem0Client") as MockClient, \
         patch("agent.skills.report_gen.scripts.run.anthropic.Anthropic") as MockAnthropic:
        MockClient.return_value.get_all = AsyncMock(return_value=[
            {"memory": "用户头痛持续两天"},
            {"memory": "用户服用布洛芬"},
        ])
        MockAnthropic.return_value.messages.create.return_value.content = [mock_block]

        from agent.skills.report_gen.scripts.run import generate_report
        result = await generate_report("user-1")

        assert isinstance(result, str)
        assert len(result) > 0
        assert "健康摘要报告" in result


@pytest.mark.asyncio
async def test_generate_report_no_memories():
    with patch("agent.skills.report_gen.scripts.run.Mem0Client") as MockClient:
        MockClient.return_value.get_all = AsyncMock(return_value=[])

        from agent.skills.report_gen.scripts.run import generate_report
        result = await generate_report("user-empty")

        assert "暂无健康记录" in result


@pytest.mark.asyncio
async def test_generate_report_empty_llm_response():
    with patch("agent.skills.report_gen.scripts.run.Mem0Client") as MockClient, \
         patch("agent.skills.report_gen.scripts.run.anthropic.Anthropic") as MockAnthropic:
        MockClient.return_value.get_all = AsyncMock(return_value=[
            {"memory": "用户头痛"},
        ])
        MockAnthropic.return_value.messages.create.return_value.content = []

        from agent.skills.report_gen.scripts.run import generate_report
        result = await generate_report("user-1")

        assert "报告生成失败" in result
