"""
TopKGraphMemory — 替换 Mem0 图记忆的 search()。

原逻辑：每个实体 → Neo4j 余弦搜索 → 限制三元组数（LIMIT 100）→ BM25 重排
新逻辑：每个实体 → 只取相似度最高的 top-N 节点 → 全量取这些节点的三元组（通常不多）
        → 去重 → 送给外层 cross-encoder 统一重排

优点：
- 不被三元组数量截断（同一节点的所有边都保留）
- 节点数可控，总三元组量自然有界
- 跳过中文效果极差的 BM25
"""
import logging
import os

logger = logging.getLogger(__name__)

NODES_PER_ENTITY = int(os.getenv("GRAPH_NODES_PER_ENTITY", "2"))


class TopKGraphMemory:
    """
    包装 Mem0 graph memory，只替换 search()，其余方法透传。
    """

    def __init__(self, base_graph, nodes_per_entity: int = NODES_PER_ENTITY):
        self._base = base_graph
        self.nodes_per_entity = nodes_per_entity

    def search(self, query: str, filters: dict, limit: int = 100) -> list[dict]:
        # Step 1: LLM 从 query 抽实体名
        entity_type_map = self._base._retrieve_nodes_from_data(query, filters)
        if not entity_type_map:
            return []

        node_props = {"user_id": filters["user_id"]}
        node_props_filter = "user_id: $user_id"
        if filters.get("agent_id"):
            node_props["agent_id"] = filters["agent_id"]
            node_props_filter += ", agent_id: $agent_id"
        if filters.get("run_id"):
            node_props["run_id"] = filters["run_id"]
            node_props_filter += ", run_id: $run_id"

        # Cypher: 先取相似度最高的 top-N 节点，再全量拿它们的三元组
        cypher = f"""
        MATCH (n {self._base.node_label} {{{node_props_filter}}})
        WHERE n.embedding IS NOT NULL
        WITH n, round(2 * vector.similarity.cosine(n.embedding, $n_embedding) - 1, 4) AS similarity
        WHERE similarity >= $threshold
        ORDER BY similarity DESC
        LIMIT $node_limit
        CALL {{
            WITH n
            MATCH (n)-[r]->(m {self._base.node_label} {{{node_props_filter}}})
            RETURN n.name AS source, type(r) AS relationship, m.name AS destination
            UNION
            WITH n
            MATCH (n)<-[r]-(m {self._base.node_label} {{{node_props_filter}}})
            RETURN m.name AS source, type(r) AS relationship, n.name AS destination
        }}
        RETURN source, relationship, destination
        """

        seen: set[tuple] = set()
        all_triplets: list[dict] = []

        # Step 2: 对每个实体分别查询，去重合并
        for entity in entity_type_map:
            n_embedding = self._base.embedding_model.embed(entity)
            params = {
                **node_props,
                "n_embedding": n_embedding,
                "threshold":   self._base.threshold,
                "node_limit":  self.nodes_per_entity,
            }
            rows = self._base.graph.query(cypher, params=params)
            for row in rows:
                key = (row["source"], row["relationship"], row["destination"])
                if key not in seen:
                    seen.add(key)
                    all_triplets.append({
                        "source":       row["source"],
                        "relationship": row["relationship"],
                        "destination":  row["destination"],
                    })

        logger.debug(
            "[graph] entities=%d nodes_per_entity=%d → triplets=%d",
            len(entity_type_map), self.nodes_per_entity, len(all_triplets),
        )
        return all_triplets

    def __getattr__(self, name: str):
        return getattr(self._base, name)
