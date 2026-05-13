"""
agent/tools.py — Claude Tool 定义与分发层。

search_memory : 检索用户历史健康记录（Mem0，prefetch 缓存优先）
search_rag    : BGE-M3 dense + sparse 双路 Prefetch，Qdrant 原生 RRF 融合
web_search    : 联网搜索占位（待填入搜索服务）
generate_report : 生成个人健康摘要报告
"""
import asyncio
import logging
import os

from memory.mem0_client import Mem0Client
from memory.session_buffer import SessionBuffer

logger = logging.getLogger(__name__)

# 可重试的瞬时异常类型（网络超时、连接中断）
_TRANSIENT_EXCEPTIONS = (OSError, ConnectionError, TimeoutError)


async def _with_retry(fn, *args, retries: int = 2, base_delay: float = 0.5, **kwargs):
    """对瞬时网络异常做指数退避重试，永久性错误直接抛出。"""
    for attempt in range(retries + 1):
        try:
            return await fn(*args, **kwargs)
        except _TRANSIENT_EXCEPTIONS as exc:
            if attempt == retries:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning("[retry] %s attempt=%d/%d, retry in %.1fs: %s",
                           fn.__name__, attempt + 1, retries, delay, exc)
            await asyncio.sleep(delay)

# ── Tool 规格，传给 _llm.bind_tools() ────────────────────────────────────────

TOOL_SPECS: list[dict] = [
    {
        "name": "search_memory",
        "description": (
            "检索该用户的历史健康记录（症状、用药、就诊等）。"
            "当用户提到'又'、'老是'、'经常'、'上次'、'之前'、'一直'、'还是'等词，"
            "或需要了解用户过往症状与当前是否相关时调用。"
            "普通问候、首次描述症状、闲聊时无需调用。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "检索关键词，简洁描述要查找的历史信息。"
                        "必须是自包含的完整描述，不能含'那个''之前''上次'等未解析指代，"
                        "需先从对话历史中确定具体症状或事件再填入。"
                    ),
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "generate_report",
        "description": (
            "生成用户的个人健康摘要报告（Markdown 格式）。"
            '当用户请求"健康报告"、"生成报告"、"查看健康摘要"、"/report"或类似指令时调用。'
            "报告涵盖症状、用药、就医、生活习惯、健康趋势五个维度。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "search_rag",
        "description": (
            "从本地医疗健康知识库检索权威参考信息。"
            "当需要药物说明、疾病描述、临床指南等权威内容时调用。"
            "勿用于日常问候或简单症状记录。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "检索查询语句，必须是自包含的完整医学术语查询，不能含未解析的指代词。"
                        "填写前先消解上下文中的指代：若用户说'那个药''之前说的药'，"
                        "必须先从对话历史中确定具体药物名称，再填入 query。"
                        "例：用户说'上次那个药有副作用吗'且上文提到布洛芬，"
                        "应填'布洛芬 不良反应 胃肠道'，而非'那个药副作用'。"
                    ),
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "联网搜索最新医疗健康资讯或研究进展。"
            "当需要近期新闻、最新研究或知识库中没有的内容时调用。"
            "勿用于日常问候或简单症状记录。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询语句",
                }
            },
            "required": ["query"],
        },
    },
]


# ── RAG 配置 ──────────────────────────────────────────────────────────────────

_QDRANT_HOST    = os.getenv("QDRANT_HOST", "localhost")
_QDRANT_PORT    = int(os.getenv("QDRANT_PORT", "6334"))
_COLLECTION     = "medical_knowledge"
_EMBED_MODEL    = "BAAI/bge-m3"
# 与 eval_retrieval.py 保持一致的 query 指令前缀
_QUERY_PREFIX   = "为这个句子生成表示以用于检索相关文章："
_PREFETCH_LIMIT = 20   # 每路 Prefetch 候选数（与 eval 脚本一致）
_RERANK_POOL    = 20   # RRF 后送入 cross-encoder 的候选数
_TOP_K          = 5    # cross-encoder 重排后最终返回条数
_RERANK_MODEL   = "BAAI/bge-reranker-v2-m3"


# ── Lazy 单例 ─────────────────────────────────────────────────────────────────

_embed_model   = None
_qdrant_client = None
_reranker      = None


def _get_embed_device() -> str:
    device = os.getenv("EMBED_DEVICE")
    if device:
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from FlagEmbedding import BGEM3FlagModel
        device = _get_embed_device()
        use_fp16 = device != "cpu"
        try:
            _embed_model = BGEM3FlagModel(_EMBED_MODEL, use_fp16=use_fp16, devices=[device])
        except TypeError:
            _embed_model = BGEM3FlagModel(_EMBED_MODEL, use_fp16=use_fp16)
        logger.info("[search_rag] BGE-M3 loaded: %s device=%s", _EMBED_MODEL, device)
    return _embed_model


def _get_reranker():
    global _reranker
    if _reranker is None:
        from FlagEmbedding import FlagReranker
        device = _get_embed_device()
        use_fp16 = device != "cpu"
        try:
            _reranker = FlagReranker(_RERANK_MODEL, use_fp16=use_fp16, device=device)
        except TypeError:
            _reranker = FlagReranker(_RERANK_MODEL, use_fp16=use_fp16)
        logger.info("[search_rag] reranker loaded: %s device=%s", _RERANK_MODEL, device)
    return _reranker


def _get_qdrant():
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import AsyncQdrantClient
        _qdrant_client = AsyncQdrantClient(
            host=_QDRANT_HOST, port=_QDRANT_PORT, prefer_grpc=True,
            check_compatibility=False,
            grpc_options=[("grpc.enable_http_proxy", 0)],
        )
        logger.info("[search_rag] Qdrant -> %s:%s (gRPC)", _QDRANT_HOST, _QDRANT_PORT)
    return _qdrant_client


# ── 编码（同步模型跑在线程池，避免阻塞事件循环）────────────────────────────────

async def _encode_query(query: str) -> tuple[list[float], dict]:
    """返回 (dense_vec, sparse_dict)，与 eval_retrieval.py 编码方式完全一致。"""
    model = _get_embed_model()
    loop  = asyncio.get_running_loop()
    enc   = await loop.run_in_executor(
        None,
        lambda: model.encode(
            _QUERY_PREFIX + query,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        ),
    )
    # 单条输入：dense_vecs 是 1-D ndarray，lexical_weights 是 dict
    return enc["dense_vecs"].tolist(), enc["lexical_weights"]


# ── Tool 实现 ─────────────────────────────────────────────────────────────────

import re as _re

# 口语化特征词：匹配到任意一个则触发改写
_COLLOQUIAL_RE = _re.compile(
    r'我|你|他|她|咋|吗|呢|吧|怎么办|怎么回事|好不好|能不能|可以吗|有没有|是不是|'
    r'管用|严不严重|要不要紧|肚子|嗓子|脑袋|拉肚|老是|总是|一直|'
    r'什么原因|好难受|好痛|好痒|好疼|很厉害|死了'
)


def _needs_rewrite(query: str) -> bool:
    """口语化特征检测：命中任一模式则需要改写，否则跳过改写节省 ~3s RTT。"""
    return bool(_COLLOQUIAL_RE.search(query))


async def _rewrite_query(query: str) -> str:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=128,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是医疗检索优化器。将用户输入改写为适合向量数据库检索的标准医疗术语查询，"
                    "只输出改写后的查询词，不加任何解释。"
                ),
            },
            {"role": "user", "content": query},
        ],
    )
    return resp.choices[0].message.content.strip()


async def _search_rag_once(query: str) -> str:
    """单次检索，不含重试逻辑。"""
    import time as _time
    from qdrant_client.models import SparseVector, Prefetch, FusionQuery, Fusion

    if _needs_rewrite(query):
        t0 = _time.monotonic()
        rewritten = await _rewrite_query(query)
        rewrite_ms = int((_time.monotonic() - t0) * 1000)
        logger.info("[search_rag] rewrite=yes original=%r rewritten=%r ms=%d",
                    query, rewritten, rewrite_ms)
    else:
        rewritten = query
        logger.info("[search_rag] rewrite=skip query=%r", query)
    client = _get_qdrant()
    dense_vec, sparse_dict = await _encode_query(rewritten)
    sparse_vec = SparseVector(
        indices=[int(k) for k in sparse_dict.keys()],
        values=[float(v) for v in sparse_dict.values()],
    )
    resp = await client.query_points(
        collection_name=_COLLECTION,
        prefetch=[
            Prefetch(query=dense_vec,  using="dense",  limit=_PREFETCH_LIMIT),
            Prefetch(query=sparse_vec, using="sparse", limit=_PREFETCH_LIMIT),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=_RERANK_POOL,
        with_payload=True,
    )
    hits = resp.points
    if not hits:
        return "[RAG] 未检索到相关内容。"

    # cross-encoder 重排：在线程池中运行同步模型，避免阻塞事件循环
    reranker = _get_reranker()
    loop = asyncio.get_running_loop()
    payloads = [h.payload for h in hits]
    contents = [p.get("content", "") for p in payloads]
    pairs = [[rewritten, c] for c in contents]
    scores = await loop.run_in_executor(
        None, lambda: reranker.compute_score(pairs, normalize=True)
    )
    ranked = sorted(zip(scores, payloads), key=lambda x: x[0], reverse=True)[:_TOP_K]

    parts: list[str] = []
    for rank, (score, p) in enumerate(ranked, 1):
        headings = " > ".join(
            h for h in [p.get("h1"), p.get("h2"), p.get("h3"), p.get("h4")] if h
        )
        parts.append(
            f"【结果 {rank}】来源：{p.get('book', '未知')}\n"
            f"章节：{headings}\n"
            f"内容：{p.get('content', '')}"
        )
    logger.info("[search_rag] query=%r rrf=%d rerank→%d", query, len(hits), len(ranked))
    return "\n\n".join(parts)


async def _search_rag(query: str) -> str:
    try:
        return await asyncio.wait_for(_with_retry(_search_rag_once, query), timeout=15.0)
    except asyncio.TimeoutError:
        logger.warning("[search_rag] timed out query=%r, falling back", query)
        return "[RAG] 本地知识库检索超时，请改用 web_search 工具补充检索。"
    except Exception as exc:
        err_str = str(exc).lower()
        # collection 不存在或 Qdrant 未就绪，降级而非抛出
        if any(k in err_str for k in ("not found", "doesn't exist", "collection", "unavailable", "connect")):
            logger.warning("[search_rag] collection unavailable query=%r: %s", query, exc)
            return "[RAG] 本地知识库暂不可用，请改用 web_search 工具补充检索。"
        logger.error("[search_rag] failed: %s", exc, exc_info=True)
        return "[RAG] 本地知识库检索失败，请改用 web_search 工具补充检索。"


async def _search_memory(query: str, user_id: str = "", limit: int = 5) -> str:
    try:
        try:
            memories = await Mem0Client(user_id=user_id).search(query, limit=limit)
        except asyncio.TimeoutError:
            logger.warning("[search_memory] Mem0 超时，降级 prefetch user=%r", user_id)
            memories = await SessionBuffer.get_mem_prefetch(user_id) or []
        lines = [f"- {m.get('memory') or m.get('content', '')}" for m in memories if m and (m.get('memory') or m.get('content'))]
        return "\n".join(lines) if lines else "未找到相关历史记录。"
    except Exception as exc:
        logger.error("[search_memory] failed user=%r: %s", user_id, exc, exc_info=True)
        raise


async def _generate_report(user_id: str = "") -> str:
    import json
    from openai import AsyncOpenAI
    from pathlib import Path

    skill_dir = Path(__file__).parent / "skills" / "report_gen"
    skill_md = (skill_dir / "SKILL.md").read_text(encoding="utf-8")

    _skill_tools = [
        {
            "type": "function",
            "function": {
                "name": "read_reference",
                "description": "从 references/ 目录读取参考文件",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string", "description": "文件名，如 output_format.md"},
                    },
                    "required": ["filename"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_script",
                "description": "运行 scripts/run.py 生成健康报告，返回最终 Markdown 文本",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    ]

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    messages = [
        {
            "role": "system",
            "content": "你是 Skill 执行引擎，严格按 SKILL.md 步骤完成任务，不得偏离，不得自行推测结果。",
        },
        {
            "role": "user",
            "content": f"执行以下 Skill，user_id={user_id!r}：\n\n{skill_md}",
        },
    ]

    try:
        for _ in range(5):
            response = await client.chat.completions.create(
                model=os.getenv("REPORT_MODEL", "gpt-4o-mini"),
                max_tokens=2048,
                tools=_skill_tools,
                messages=messages,
            )

            choice = response.choices[0]
            if choice.finish_reason == "stop":
                return choice.message.content or "[报告生成失败] 未获得文本输出。"

            # tool_calls 轮：执行工具调用，结果追加后继续
            messages.append(choice.message)
            for tc in (choice.message.tool_calls or []):
                args = json.loads(tc.function.arguments)
                if tc.function.name == "read_reference":
                    ref_path = skill_dir / "references" / args["filename"]
                    try:
                        content = ref_path.read_text(encoding="utf-8")
                    except FileNotFoundError:
                        content = f"[错误] 文件不存在: {args['filename']}"
                elif tc.function.name == "run_script":
                    from agent.skills.report_gen.scripts.run import generate_report
                    content = await generate_report(user_id)
                else:
                    content = f"[错误] 未知工具: {tc.function.name}"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": content,
                })

        logger.error("[generate_report] exceeded max rounds user=%r", user_id)
        return "[报告生成失败] 报告生成暂时不可用，请稍后重试。"

    except Exception as exc:
        logger.error("[generate_report] failed user=%r: %s", user_id, exc, exc_info=True)
        return "[报告生成失败] 报告生成暂时不可用，请稍后重试。"


async def _web_search(query: str) -> str:
    import httpx
    api_key = os.getenv("BRAVE_API_KEY", "")
    if not api_key:
        logger.warning("[web_search] BRAVE_API_KEY 未配置，query=%r", query)
        return f'[web_search] 搜索服务未配置，无法查询 "{query}"。'
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": api_key,
                },
                params={"q": query, "count": 5, "search_lang": "zh-hans"},
            )
            resp.raise_for_status()
            data = resp.json()
        results = data.get("web", {}).get("results", [])
        if not results:
            return f'[web_search] 未找到与 "{query}" 相关的结果。'
        parts = [
            f"**[{r['title']}]({r['url']})**\n{r.get('description', '')}"
            for r in results[:5]
        ]
        logger.info("[web_search] query=%r returned=%d", query, len(results))
        return "\n\n".join(parts)
    except Exception as exc:
        logger.error("[web_search] failed query=%r: %s", query, exc)
        return f'[web_search] 搜索失败，请依据已有知识作答。'


# ── Dispatcher ────────────────────────────────────────────────────────────────

_DISPATCH = {
    "search_memory": _search_memory,
    "generate_report": _generate_report,
    "search_rag": _search_rag,
    "web_search": _web_search,
}


async def dispatch_tool(name: str, args: dict, user_id: str = "") -> tuple[str, bool]:
    """
    路由工具调用，返回 (content, is_error)。
    search_rag 失败后自动降级到 web_search。
    """
    fn = _DISPATCH.get(name)
    if fn is None:
        logger.error("[dispatch_tool] unknown tool: %s", name)
        return f"未知工具: {name}", True

    try:
        if name in ("generate_report", "search_memory"):
            return await fn(**args, user_id=user_id), False
        return await fn(**args), False

    except Exception as exc:
        logger.error("[dispatch_tool] %s failed: %s", name, exc, exc_info=True)

        return f"[工具错误] {name} 执行失败，请立即改用 web_search 工具补充检索，不得使用训练知识直接作答。", True
