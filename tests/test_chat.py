"""
chat.py 传输层测试。

职责边界：只测 HTTP transport 层——路由、session 读取、SSE 格式。
业务逻辑（triage / critic / Zep）由 test_harness.py 负责，这里用 FakeHarness 隔离。
"""
import os

import jwt
import pytest
from httpx import ASGITransport, AsyncClient

# 固定测试用密钥，覆盖 .env 里的真实密钥，让 middleware 能验证下面生成的 token
os.environ["JWT_SECRET"] = "test-secret"

from backend.main import app  # noqa: E402（必须在 env 设置之后 import）


# ── Fakes ────────────────────────────────────────────────────────────────────

class FakeSessionBuffer:
    store: dict[str, list[dict]] = {}

    def __init__(self, session_id: str):
        self.session_id = session_id

    async def get(self) -> list[dict]:
        return list(self.store.get(self.session_id, []))

    async def set(self, messages: list[dict]) -> None:
        self.store[self.session_id] = list(messages)

    async def get_consult(self) -> dict:
        return {}

    async def clear_consult(self) -> None:
        pass


class FakeHarness:
    """
    替换 _harness 单例，验证 chat.py 把正确的参数传给 harness，
    并将 harness 产生的 SSE chunks 透传给客户端。
    """
    captured: list[dict] = []
    on_stop_hooks: list = []

    async def run(
        self,
        user_message: str,
        user_id: str,
        session_id: str,
        history: list[dict],
        original_user_content=None,
    ):
        FakeHarness.captured.append({
            "user_message": user_message,
            "user_id": user_id,
            "session_id": session_id,
            "history": history,
        })

        async def _gen():
            yield 'data: {"type": "text", "delta": "这是新的回复"}\n\n'
            yield 'data: {"type": "done"}\n\n'

        return _gen()


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_passes_history_to_harness(monkeypatch):
    """chat.py 从 session 读出历史，正确传给 harness.run()。"""
    FakeHarness.captured = []
    FakeSessionBuffer.store = {
        "demo-session": [
            {"role": "user",      "content": "之前的问题"},
            {"role": "assistant", "content": "之前的回答"},
        ]
    }

    monkeypatch.setattr("backend.routers.chat.SessionBuffer", FakeSessionBuffer)
    monkeypatch.setattr("backend.routers.chat._harness", FakeHarness())

    token = jwt.encode({"sub": "demo-user"}, "test-secret", algorithm="HS256")
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.post(
            "/chat",
            json={"message": "这次的新问题", "session_id": "demo-session"},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 200
    assert len(FakeHarness.captured) == 1
    call = FakeHarness.captured[0]
    # 验证历史被正确传入
    assert call["history"] == [
        {"role": "user",      "content": "之前的问题"},
        {"role": "assistant", "content": "之前的回答"},
    ]
    assert call["user_message"] == "这次的新问题"
    assert call["user_id"] == "demo-user"


@pytest.mark.asyncio
async def test_chat_sse_chunks_forwarded(monkeypatch):
    """harness 产生的 SSE chunks 被原封不动转发给客户端。"""
    FakeHarness.captured = []
    FakeSessionBuffer.store = {}

    monkeypatch.setattr("backend.routers.chat.SessionBuffer", FakeSessionBuffer)
    monkeypatch.setattr("backend.routers.chat._harness", FakeHarness())

    token = jwt.encode({"sub": "demo-user"}, "test-secret", algorithm="HS256")
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.post(
            "/chat",
            json={"message": "你好"},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 200
    assert '"type": "text"' in response.text
    assert '"type": "done"' in response.text
    assert "这是新的回复" in response.text
