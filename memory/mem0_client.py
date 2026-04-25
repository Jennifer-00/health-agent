import os
from mem0 import AsyncMemoryClient

_mem = AsyncMemoryClient(api_key=os.getenv("MEM0_API_KEY", ""))


class Mem0Client:
    """封装 memory.add() / search() / update()，绑定 user_id"""

    def __init__(self, user_id: str):
        self.user_id = user_id

    async def add(self, content: str, metadata: dict | None = None) -> dict:
        return await _mem.add(
            content,
            user_id=self.user_id,
            metadata=metadata or {},
        )

    async def search(self, query: str, limit: int = 5) -> list[dict]:
        results = await _mem.search(
            query,
            filters={"user_id": self.user_id},
            limit=limit,
        )
        raw = results if isinstance(results, list) else results.get("results", [])
        return [m for m in raw if m]

    async def update(self, memory_id: str, content: str) -> dict:
        return await _mem.update(memory_id, content)

    async def get_all(self) -> list[dict]:
        results = await _mem.get_all(filters={"user_id": self.user_id})
        raw = results if isinstance(results, list) else results.get("results", [])
        return [m for m in raw if m]

    async def delete(self, memory_id: str) -> None:
        await _mem.delete(memory_id)
