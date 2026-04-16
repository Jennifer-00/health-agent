"""skill 单元测试"""
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_memory_write_calls_mem0():
    with patch("agent.skills.memory_write.Mem0Client") as MockClient:
        instance = MockClient.return_value
        instance.add = AsyncMock(return_value={"id": "mem-123"})

        from agent.skills.memory_write import memory_write
        result = await memory_write.ainvoke({
            "user_id": "user-1",
            "content": "头痛持续两天",
            "category": "症状",
        })

        instance.add.assert_called_once()
        assert "mem-123" in result


@pytest.mark.asyncio
async def test_memory_search_returns_combined():
    with patch("agent.skills.memory_search.Mem0Client") as MockMem0, \
         patch("agent.skills.memory_search.ZepClient") as MockZep:
        MockMem0.return_value.search = AsyncMock(return_value=[{"content": "头痛记录"}])
        MockZep.return_value.traverse = AsyncMock(return_value=[])

        from agent.skills.memory_search import memory_search
        result = await memory_search.ainvoke({
            "user_id": "user-1",
            "query": "头痛",
        })

        assert "头痛记录" in result
