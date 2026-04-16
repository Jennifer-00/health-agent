import datetime
import os

from zep_cloud.client import AsyncZep
from zep_cloud.types.message import Message


class ZepClient:
    """Graphiti 图谱操作：Thread 写入、时序边查询、反复发作检测"""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self._zep = AsyncZep(api_key=os.getenv("ZEP_API_KEY", ""))

    async def _ensure_user(self) -> None:
        """用户不存在时自动创建，已存在则忽略。"""
        try:
            await self._zep.user.add(user_id=self.user_id)
        except Exception:
            pass

    async def _ensure_thread(self, session_id: str) -> None:
        """Thread 不存在时自动创建，已存在则忽略。"""
        try:
            await self._zep.thread.create(thread_id=session_id, user_id=self.user_id)
        except Exception:
            pass

    async def add_thread_messages(
        self, session_id: str, user_content: str, assistant_content: str
    ) -> None:
        """将一轮完整对话写入 Zep Thread，触发图谱实体和边的提取。"""
        await self._ensure_user()
        await self._ensure_thread(session_id)
        await self._zep.thread.add_messages(
            thread_id=session_id,
            messages=[
                Message(role="user", content=user_content),
                Message(role="assistant", content=assistant_content),
            ],
            ignore_roles=["assistant"],
        )

    async def traverse(self, query: str, hops: int = 2) -> list[dict]:
        """时序图谱遍历，返回相关节点。hops 映射为检索条数上限。"""
        await self._ensure_user()
        try:
            resp = await self._zep.graph.search(
                user_id=self.user_id,
                query=query,
                limit=hops * 3,
                reranker="mmr",  # cross_encoder 最准但最慢；mmr 速度快且结果多样性好
            )
        except Exception:
            return []

        edges = resp if isinstance(resp, list) else getattr(resp, "edges", [])
        return [{"content": e.fact} for e in edges if e.fact]

    async def find_recurrence(self, content: str) -> list[dict]:
        """检测当前症状是否与历史节点语义相似，返回候选节点。"""
        await self._ensure_user()
        try:
            resp = await self._zep.graph.search(
                user_id=self.user_id,
                query=content,
                limit=5,
                reranker="cross_encoder",
            )
        except Exception:
            return []

        edges = resp if isinstance(resp, list) else getattr(resp, "edges", [])
        nodes = []
        for e in edges:
            if not e.fact:
                continue
            date_str = ""
            if e.created_at:
                try:
                    date_str = datetime.datetime.fromisoformat(e.created_at).strftime("%Y-%m-%d")
                except Exception:
                    date_str = str(e.created_at)[:10]
            nodes.append({"id": e.uuid_ or "", "summary": e.fact, "date": date_str})

        return nodes

    async def add_fact(self, content: str, category: str) -> None:
        """显式写入一条健康事实，用主谓宾结构帮助 Zep 提取关系边。"""
        await self._ensure_user()
        templates = {
            "症状": f"用户报告症状：{content}",
            "用药": f"用户服用药物：{content}",
            "就医": f"用户就诊记录：{content}",
            "检查": f"用户检查结果：{content}",
        }
        data = templates.get(category, f"用户健康记录（{category}）：{content}")
        await self._zep.graph.add(
            user_id=self.user_id,
            type="text",
            data=data,
        )

    async def add_recurrence_edge(self, prior_summary: str, content: str) -> None:
        """在图谱中写入一条反复发作的关联描述，Zep 自动建立边关系。"""
        await self._ensure_user()
        await self._zep.graph.add(
            user_id=self.user_id,
            type="text",
            data=f"[反复发作] {content}，与此前记录\"{prior_summary}\"为同一症状再次发作。",
        )
