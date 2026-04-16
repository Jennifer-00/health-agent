"""Unit tests for the memory consolidation skill."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.skills import memory_consolidate as consolidate


def test_parse_json_supports_fenced_blocks():
    payload = """```json
    [
      {"keep_id": "m1", "merged": "头痛持续两天", "delete_ids": ["m2"]}
    ]
    ```"""

    result = consolidate._parse_json(payload)

    assert result == [
        {"keep_id": "m1", "merged": "头痛持续两天", "delete_ids": ["m2"]}
    ]


@pytest.mark.asyncio
async def test_do_consolidate_merges_same_day_records(monkeypatch):
    fake_memories = [
        {"id": "m1", "memory": "头痛两天", "created_at": "2026-04-16T09:00:00"},
        {"id": "m2", "memory": "今天还在头痛", "created_at": "2026-04-16T11:00:00"},
    ]
    fake_client = SimpleNamespace(
        get_all=AsyncMock(return_value=fake_memories),
        update=AsyncMock(),
        delete=AsyncMock(),
    )
    mark_mock = AsyncMock()
    llm_response = SimpleNamespace(
        content='[{"keep_id":"m1","merged":"头痛持续两天","delete_ids":["m2"]}]'
    )

    monkeypatch.setattr(consolidate, "Mem0Client", lambda user_id: fake_client)
    monkeypatch.setattr(consolidate, "_get_consolidated", AsyncMock(return_value=set()))
    monkeypatch.setattr(consolidate, "_mark_consolidated", mark_mock)
    monkeypatch.setattr(
        consolidate,
        "_llm",
        SimpleNamespace(ainvoke=AsyncMock(return_value=llm_response)),
    )

    result = await consolidate._do_consolidate("demo-user")

    fake_client.update.assert_awaited_once_with("m1", "头痛持续两天")
    fake_client.delete.assert_awaited_once_with("m2")
    mark_mock.assert_awaited_once()
    marked_ids = set(mark_mock.await_args.args[1])
    assert marked_ids == {"m1", "m2"}
    assert "1 组" in result
    assert "删除 1 条" in result


@pytest.mark.asyncio
async def test_do_consolidate_returns_early_for_insufficient_records(monkeypatch):
    fake_memories = [
        {"id": "m1", "memory": "昨晚有些头痛", "created_at": "2026-04-16T09:00:00"},
    ]
    fake_client = SimpleNamespace(
        get_all=AsyncMock(return_value=fake_memories),
        update=AsyncMock(),
        delete=AsyncMock(),
    )
    mark_mock = AsyncMock()
    llm_mock = AsyncMock()

    monkeypatch.setattr(consolidate, "Mem0Client", lambda user_id: fake_client)
    monkeypatch.setattr(consolidate, "_get_consolidated", AsyncMock(return_value=set()))
    monkeypatch.setattr(consolidate, "_mark_consolidated", mark_mock)
    monkeypatch.setattr(consolidate, "_llm", SimpleNamespace(ainvoke=llm_mock))

    result = await consolidate._do_consolidate("demo-user")

    fake_client.update.assert_not_awaited()
    fake_client.delete.assert_not_awaited()
    llm_mock.assert_not_awaited()
    mark_mock.assert_not_awaited()
    assert result == "条数不足，跳过。"
