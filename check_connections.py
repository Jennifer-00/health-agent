"""
连通性测试 — 逐个检测 .env 中配置的所有服务
用法：python check_connections.py
"""
import posthog
posthog.disabled = True
posthog.api_key = "disabled"

import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv(override=True)

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
SKIP = "\033[33m-\033[0m"


def result(label: str, ok: bool, detail: str = ""):
    icon = PASS if ok else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  {icon}  {label}{suffix}")


# ── Anthropic ──────────────────────────────────────────────────────────────
async def check_anthropic():
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key or key.startswith("sk-ant-..."):
        result("Anthropic", False, "未填写 API Key")
        return
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=8,
            messages=[{"role": "user", "content": "hi"}],
        )
        result("Anthropic", True, msg.model)
    except Exception as e:
        result("Anthropic", False, str(e)[:80])


# ── Mem0 ───────────────────────────────────────────────────────────────────
async def check_mem0():
    key = os.getenv("MEM0_API_KEY", "")
    if not key:
        result("Mem0", False, "未填写 API Key")
        return
    try:
        from mem0 import MemoryClient
        client = MemoryClient(api_key=key)
        client.get_all(user_id="__test__")
        result("Mem0", True)
    except Exception as e:
        err = str(e)
        # 服务器返回 "Filters are required" 说明网络通且 Key 有效，只是调用参数问题
        if "Filters are required" in err:
            result("Mem0", True, "connected (API key valid)")
        else:
            result("Mem0", False, err[:80])


# ── Zep ────────────────────────────────────────────────────────────────────
async def check_zep():
    key = os.getenv("ZEP_API_KEY", "")
    if not key:
        result("Zep", False, "未填写 API Key")
        return
    try:
        from zep_cloud.client import Zep
        client = Zep(api_key=key)
        # zep-cloud 3.x API：通过 user 接口验证连通性
        client.user.list_ordered(page_size=1)
        result("Zep", True)
    except Exception as e:
        result("Zep", False, str(e)[:80])


# ── PostgreSQL (Supabase) ──────────────────────────────────────────────────
async def check_postgres():
    url = os.getenv("DATABASE_URL", "")
    if not url or "user:pass" in url:
        result("PostgreSQL", False, "未填写连接串")
        return
    try:
        import psycopg2
        clean_url = url.replace("postgresql+asyncpg://", "postgresql://")
        conn = psycopg2.connect(clean_url, sslmode="require", connect_timeout=8)
        cur = conn.cursor()
        cur.execute("SELECT version()")
        version = cur.fetchone()[0].split(",")[0]
        conn.close()
        result("PostgreSQL", True, version)
    except Exception as e:
        result("PostgreSQL", False, str(e)[:80])


# ── Redis (Upstash) ────────────────────────────────────────────────────────
async def check_redis():
    url = os.getenv("REDIS_URL", "")
    if not url or "<PASSWORD>" in url:
        result("Redis", False, "REDIS_URL 中 <PASSWORD> 尚未替换")
        return
    try:
        import redis.asyncio as aioredis
        client = aioredis.from_url(url, socket_connect_timeout=8)
        pong = await client.ping()
        await client.aclose()
        result("Redis", pong, "PONG" if pong else "no response")
    except Exception as e:
        result("Redis", False, str(e)[:80])


# ── Neo4j ──────────────────────────────────────────────────────────────────
async def check_neo4j():
    uri  = os.getenv("NEO4J_URI", "")
    user = os.getenv("NEO4J_USER", "neo4j")
    pwd  = os.getenv("NEO4J_PASSWORD", "")
    if not uri or uri == "bolt://localhost:7687" and not pwd:
        result("Neo4j", False, "未填写连接信息")
        return
    try:
        from neo4j import AsyncGraphDatabase
        driver = AsyncGraphDatabase.driver(uri, auth=(user, pwd))
        async with driver.session() as session:
            rec = await session.run("RETURN 1 AS n")
            await rec.single()
        await driver.close()
        result("Neo4j", True, uri.split("//")[-1])
    except Exception as e:
        result("Neo4j", False, str(e)[:80])


# ── Qdrant ─────────────────────────────────────────────────────────────────
async def check_qdrant():
    url     = os.getenv("QDRANT_URL", "")
    api_key = os.getenv("QDRANT_API_KEY", "")
    if not url or url == "http://localhost:6333":
        result("Qdrant", False, "未填写连接 URL")
        return
    try:
        from qdrant_client import AsyncQdrantClient
        client = AsyncQdrantClient(url=url, api_key=api_key or None, timeout=8)
        info = await client.get_collections()
        await client.close()
        result("Qdrant", True, f"{len(info.collections)} collections")
    except Exception as e:
        result("Qdrant", False, str(e)[:80])


# ── main ───────────────────────────────────────────────────────────────────
async def main():
    print("\n=== 健康 Agent 连通性检测 ===\n")
    await check_anthropic()
    await check_mem0()
    await check_zep()
    await check_postgres()
    await check_redis()
    await check_neo4j()
    await check_qdrant()
    print()


if __name__ == "__main__":
    asyncio.run(main())
