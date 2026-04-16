"""agent graph 集成测试"""
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_graph_runs_without_error():
    """验证图能正常编译和执行一轮"""
    with patch("agent.nodes._llm") as mock_llm:
        from langchain_core.messages import AIMessage
        mock_response = AIMessage(content="您好，请问有什么健康问题？")
        mock_response.tool_calls = []
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        from agent.graph import build_graph
        graph = build_graph()
        state = {
            "messages": [{"role": "user", "content": "你好"}],
            "user_id": "test-user",
            "pending_memories": [],
        }
        result = await graph.ainvoke(state)
        assert "messages" in result
