import os
from mem0 import AsyncMemoryClient


class Mem0Client:
    """封装 memory.add() / search() / update()，绑定 user_id"""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self._mem = AsyncMemoryClient(api_key=os.getenv("MEM0_API_KEY", ""))

    async def add(self, content: str, metadata: dict | None = None) -> dict:
        return await self._mem.add(
            content,
            user_id=self.user_id,
            metadata=metadata or {},
        )

    async def search(self, query: str, limit: int = 5) -> list[dict]:
        results = await self._mem.search(
            query,
            filters={"user_id": self.user_id},
            limit=limit,
        )
        if isinstance(results, list):
            return results
        return results.get("results", [])

    async def update(self, memory_id: str, content: str) -> dict:
        return await self._mem.update(memory_id, content)

    async def get_all(self) -> list[dict]:
        results = await self._mem.get_all(filters={"user_id": self.user_id})
        if isinstance(results, list):
            return results
        return results.get("results", [])

    async def delete(self, memory_id: str) -> None:
        await self._mem.delete(memory_id)
