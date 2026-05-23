"""
evals/rag_eval_ragas.py — RAGAS TestsetGenerator 驱动的 RAG 质量评测

流程：
  1. 从 --docs-dir 加载知识库原始文档（.txt / .md / .pdf）
  2. RAGAS TestsetGenerator 自动合成 QA 测试集（可缓存跳过）
  3. 对每条问题，并行跑两路检索 + LLM 生成答案：
       • 原始 query  → Qdrant RRF → answer
       • 重写后 query → Qdrant RRF → answer
  4. 用 faithfulness / answer_relevancy / context_precision / context_recall 评测
  5. 输出对比表格，结果写入 evals/results/

安装依赖（项目基础上额外需要）：
  pip install "ragas>=0.4" langchain-community pypdf

运行示例：
  # 完整流程（使用默认文档目录）
  python evals/rag_eval_ragas.py --testset-size 20

  # 指定其他文档目录
  python evals/rag_eval_ragas.py --docs-dir /path/to/md_output --testset-size 20

  # 跳过生成，复用已有测试集（生成很慢，建议缓存后复用）
  python evals/rag_eval_ragas.py --load-testset evals/results/testset_20260426.json

  # 调整检索 top-k
  python evals/rag_eval_ragas.py --testset-size 10 --top-k 10
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")
from dotenv import load_dotenv
load_dotenv(override=True)


# ── 共享配置（与 agent/tools.py 保持一致）────────────────────────────────────

_QUERY_PREFIX   = "为这个句子生成表示以用于检索相关文章："
_COLLECTION     = "medical_knowledge"
_EMBED_MODEL    = "BAAI/bge-m3"
_PREFETCH_LIMIT = 20

_embed_model   = None
_qdrant_client = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from FlagEmbedding import BGEM3FlagModel
        _embed_model = BGEM3FlagModel(_EMBED_MODEL, use_fp16=True)
        print(f"[init] BGE-M3 loaded")
    return _embed_model


def _get_qdrant():
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import AsyncQdrantClient
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6334"))
        _qdrant_client = AsyncQdrantClient(host=host, port=port, prefer_grpc=True)
    return _qdrant_client


# ── LangChain Embeddings 包装（给 RAGAS 用）──────────────────────────────────

from langchain_core.embeddings import Embeddings as _LCEmbeddings


class _BGELCEmbeddings(_LCEmbeddings):
    """将 BGE-M3 包装成 LangChain Embeddings 接口，供 TestsetGenerator / evaluate 使用。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = _get_embed_model()
        enc = model.encode(texts, return_dense=True,
                           return_sparse=False, return_colbert_vecs=False)
        vecs = enc["dense_vecs"]
        # encode 单条时返回 1-D，多条时返回 2-D
        if hasattr(vecs, "tolist"):
            result = vecs.tolist()
            return result if isinstance(result[0], list) else [result]
        return [list(v) for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        model = _get_embed_model()
        enc = model.encode(
            _QUERY_PREFIX + text,
            return_dense=True, return_sparse=False, return_colbert_vecs=False,
        )
        vec = enc["dense_vecs"]
        return vec.tolist() if hasattr(vec, "tolist") else list(vec)


# ── Qdrant 检索 ───────────────────────────────────────────────────────────────

async def _encode_query(query: str) -> tuple[list[float], dict]:
    model = _get_embed_model()
    loop  = asyncio.get_running_loop()
    enc   = await loop.run_in_executor(
        None,
        lambda: model.encode(
            _QUERY_PREFIX + query,
            return_dense=True, return_sparse=True, return_colbert_vecs=False,
        ),
    )
    return enc["dense_vecs"].tolist(), enc["lexical_weights"]


async def _qdrant_search(query_text: str, top_k: int) -> list[str]:
    """返回 top-k 检索结果的纯文本列表（供 RAGAS retrieved_contexts 使用）。"""
    from qdrant_client.models import SparseVector, Prefetch, FusionQuery, Fusion

    client = _get_qdrant()
    dense_vec, sparse_dict = await _encode_query(query_text)
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
        limit=top_k,
        with_payload=True,
    )
    contexts = []
    for hit in resp.points:
        p = hit.payload or {}
        headings = " > ".join(h for h in [p.get("h1"), p.get("h2"), p.get("h3")] if h)
        text = f"[{p.get('book', '')} {headings}]\n{p.get('content', '')}".strip()
        contexts.append(text)
    return contexts


# ── Query 重写 ────────────────────────────────────────────────────────────────

async def _rewrite_query(query: str) -> str:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
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


# ── 答案生成（RAG answer synthesis）─────────────────────────────────────────

async def _generate_answer(question: str, contexts: list[str]) -> str:
    """用检索到的 contexts 生成回答，供 faithfulness / answer_relevancy 评测。"""
    from openai import AsyncOpenAI

    if not contexts:
        return "未检索到相关内容，无法回答。"

    ctx_text = "\n\n".join(f"[参考{i+1}]\n{c}" for i, c in enumerate(contexts))
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=512,
        messages=[
            {"role": "system", "content": "你是医疗健康助手，仅根据提供的参考内容回答问题，不得编造。"},
            {"role": "user", "content": f"参考内容：\n{ctx_text}\n\n问题：{question}"},
        ],
    )
    return resp.choices[0].message.content.strip()


# ── Step 1：加载文档 ──────────────────────────────────────────────────────────

_DEFAULT_DOCS_DIR = r"C:\Users\Administrator\Desktop\Agent\embedding处理\md_output"


def load_documents(docs_dir: str) -> list:
    """
    加载 docs_dir 下的 .md 文件，用 MarkdownHeaderTextSplitter 按标题切块。
    每个切块保留所属标题层级作为 metadata，与 Qdrant 索引时的切块逻辑对应。
    """
    from langchain_text_splitters import MarkdownHeaderTextSplitter
    from langchain_core.documents import Document

    docs_path = Path(docs_dir)
    if not docs_path.exists():
        print(f"[error] 文档目录不存在: {docs_path.resolve()}")
        sys.exit(1)

    md_files = list(docs_path.glob("**/*.md"))
    if not md_files:
        print(f"[error] {docs_path} 中未找到 .md 文件。")
        sys.exit(1)

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#",   "h1"),
            ("##",  "h2"),
            ("###", "h3"),
            ("####","h4"),
        ],
        strip_headers=False,   # 保留标题文本，让 TestsetGenerator 看到章节上下文
    )

    all_chunks: list[Document] = []
    for md_file in sorted(md_files):
        book_name = md_file.stem
        text = md_file.read_text(encoding="utf-8")
        chunks = splitter.split_text(text)
        for chunk in chunks:
            if len(chunk.page_content.strip()) < 50:   # 过滤过短的碎片
                continue
            chunk.metadata["book"]   = book_name
            chunk.metadata["source"] = f"{book_name} / {chunk.metadata.get('h1','')}"
            all_chunks.append(chunk)

    print(f"[info] 已加载 {len(md_files)} 个文件，切分为 {len(all_chunks)} 个文档块")
    return all_chunks


# ── Step 2：生成测试集 ────────────────────────────────────────────────────────

def generate_testset(docs: list, testset_size: int, output_json: str) -> list[dict]:
    """
    用 RAGAS TestsetGenerator 从文档合成 QA 对。
    返回 list[dict]，每条含 user_input / reference / reference_contexts。
    """
    from langchain_openai import ChatOpenAI
    from ragas.testset import TestsetGenerator

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.environ["OPENAI_API_KEY"],
        max_tokens=1024,
    )
    emb = _BGELCEmbeddings()

    print(f"[info] 初始化 TestsetGenerator，生成 {testset_size} 条 QA …")
    generator = TestsetGenerator.from_langchain(llm=llm, embedding_model=emb)
    testset   = generator.generate_with_langchain_docs(
        documents=docs,
        testset_size=testset_size,
        raise_exceptions=False,
    )

    samples = []
    for ts in testset.samples:
        es = ts.eval_sample
        # eval_sample 是 SingleTurnSample
        samples.append({
            "user_input":         es.user_input or "",
            "reference":          es.reference  or "",
            "reference_contexts": es.reference_contexts or [],
        })

    # 过滤掉空问题
    samples = [s for s in samples if s["user_input"].strip()]
    print(f"[info] 有效测试用例 {len(samples)} 条")

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"[info] 测试集已保存 → {output_json}")
    return samples


def load_testset(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        samples = json.load(f)
    print(f"[info] 从缓存加载测试集 {len(samples)} 条 ← {path}")
    return samples


# ── Step 3：双路检索 + 生成答案 ───────────────────────────────────────────────

async def build_eval_rows(samples: list[dict], top_k: int) -> list[dict]:
    """
    对每条 sample 并行执行：
      raw  : 直接检索 → 生成答案
      rew  : 重写 query → 检索 → 生成答案
    返回包含两路结果的行列表。
    """
    sem = asyncio.Semaphore(3)  # 同时最多 3 条，受 BGE-M3 内存限制

    async def process_one(s: dict) -> dict:
        async with sem:
            question  = s["user_input"]
            reference = s["reference"]

            # 重写
            t0 = time.monotonic()
            rewritten = await _rewrite_query(question)
            rewrite_ms = (time.monotonic() - t0) * 1000

            # 两路并行检索
            raw_ctxs, rew_ctxs = await asyncio.gather(
                _qdrant_search(question,  top_k),
                _qdrant_search(rewritten, top_k),
            )

            # 两路并行生成答案
            raw_answer, rew_answer = await asyncio.gather(
                _generate_answer(question, raw_ctxs),
                _generate_answer(question, rew_ctxs),
            )

            return {
                "user_input":   question,
                "reference":    reference,
                "rewritten":    rewritten,
                "rewrite_ms":   rewrite_ms,
                # 原始路径
                "raw_contexts": raw_ctxs,
                "raw_answer":   raw_answer,
                # 重写路径
                "rew_contexts": rew_ctxs,
                "rew_answer":   rew_answer,
            }

    print(f"[info] 双路检索 + 答案生成，共 {len(samples)} 条…")
    t0 = time.monotonic()
    rows = await asyncio.gather(*[process_one(s) for s in samples])
    print(f"[info] 完成，耗时 {time.monotonic()-t0:.1f}s")
    return list(rows)


# ── Step 4：RAGAS 评测 ────────────────────────────────────────────────────────

def run_ragas_eval(rows: list[dict], tag: str) -> dict:
    """
    对 raw 或 rew 路径构建 EvaluationDataset 并调用 evaluate()。
    tag: "raw" | "rew"
    返回 {metric_name: score} dict。
    """
    from langchain_openai import ChatOpenAI
    from ragas import evaluate, EvaluationDataset
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        LLMContextRecall,
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.environ["OPENAI_API_KEY"],
        max_tokens=1024,
    )
    emb = _BGELCEmbeddings()

    ctx_key = f"{tag}_contexts"
    ans_key = f"{tag}_answer"

    eval_samples = [
        SingleTurnSample(
            user_input=r["user_input"],
            response=r[ans_key],
            retrieved_contexts=r[ctx_key],
            reference=r["reference"],
        )
        for r in rows
    ]
    dataset = EvaluationDataset(samples=eval_samples)

    metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), LLMContextRecall()]

    print(f"[info] RAGAS evaluate ({tag})…")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=emb,
        show_progress=True,
        raise_exceptions=False,
    )
    # result 是 EvaluationResult，支持 dict-like 访问
    scores = dict(result)
    print(f"[info] {tag} scores: {scores}")
    return scores


# ── Step 5：报告输出 ──────────────────────────────────────────────────────────

METRIC_LABELS = {
    "faithfulness":     "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "context_precision":"Context Precision",
    "context_recall":   "Context Recall",
    "llm_context_recall": "Context Recall",   # ragas 0.4.x 可能用此名
}


def _find_score(scores: dict, *candidate_keys: str) -> float | None:
    for k in candidate_keys:
        if k in scores:
            return scores[k]
    return None


def print_comparison(raw_scores: dict, rew_scores: dict, top_k: int) -> None:
    pairs = [
        ("faithfulness",      "faithfulness",     "faithfulness"),
        ("answer_relevancy",  "answer_relevancy",  "answer_relevancy"),
        ("context_precision", "context_precision", "context_precision"),
        ("context_recall",    "context_recall",    "llm_context_recall"),
    ]

    print("\n" + "=" * 68)
    print("  RAG 质量评测：query 重写前 vs 后（RAGAS 四指标）")
    print(f"  top_k = {top_k}")
    print("=" * 68)
    print(f"  {'指标':<22}  {'重写前':>8}  {'重写后':>8}  {'Delta':>8}")
    print("  " + "─" * 58)

    for label, raw_key, rew_key in pairs:
        r = _find_score(raw_scores, raw_key, rew_key)
        w = _find_score(rew_scores, raw_key, rew_key)
        r_str = f"{r:.4f}" if r is not None else "  N/A  "
        w_str = f"{w:.4f}" if w is not None else "  N/A  "
        d_str = f"{w-r:+.4f}" if (r is not None and w is not None) else "  N/A  "
        display = METRIC_LABELS.get(label, label)
        print(f"  {display:<22}  {r_str:>8}  {w_str:>8}  {d_str:>8}")

    print("=" * 68)


def build_md_report(raw_scores: dict, rew_scores: dict,
                    rows: list[dict], top_k: int) -> str:
    lines = ["# RAGAS RAG 评测报告：query 重写前 vs 后\n"]
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  top_k = {top_k}\n")

    # 指标汇总表
    lines.append("## 整体指标对比\n")
    lines.append("| 指标 | 重写前 | 重写后 | Delta |")
    lines.append("|------|-------:|-------:|------:|")
    pairs = [
        ("Faithfulness",      "faithfulness",      "faithfulness"),
        ("Answer Relevancy",  "answer_relevancy",   "answer_relevancy"),
        ("Context Precision", "context_precision",  "context_precision"),
        ("Context Recall",    "context_recall",     "llm_context_recall"),
    ]
    for label, raw_key, rew_key in pairs:
        r = _find_score(raw_scores, raw_key, rew_key)
        w = _find_score(rew_scores, raw_key, rew_key)
        r_str = f"{r:.4f}" if r is not None else "N/A"
        w_str = f"{w:.4f}" if w is not None else "N/A"
        d_str = f"{w-r:+.4f}" if (r is not None and w is not None) else "N/A"
        lines.append(f"| {label} | {r_str} | {w_str} | **{d_str}** |")
    lines.append("")

    # 逐条明细（query + rewritten）
    lines.append("## 测试集明细（query 重写对照）\n")
    lines.append("| # | 原始 query | 重写后 query |")
    lines.append("|---|-----------|-------------|")
    for i, r in enumerate(rows, 1):
        q  = r["user_input"][:40] + ("…" if len(r["user_input"]) > 40 else "")
        rw = r["rewritten"][:45]  + ("…" if len(r["rewritten"])  > 45 else "")
        lines.append(f"| {i:02d} | {q} | {rw} |")
    lines.append("")

    return "\n".join(lines) + "\n"


# ── 主入口 ────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── 获取测试集 ──
    if args.load_testset:
        samples = load_testset(args.load_testset)
        testset_json = args.load_testset
    else:
        docs         = load_documents(args.docs_dir)
        testset_json = str(results_dir / f"testset_{ts}.json")
        samples      = generate_testset(docs, args.testset_size, testset_json)

    # ── 双路检索 + 答案 ──
    rows = await build_eval_rows(samples, args.top_k)

    rows_json = str(results_dir / f"rag_rows_{ts}.json")
    with open(rows_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"[info] 检索结果已保存 → {rows_json}")

    # ── RAGAS 评测 ──
    raw_scores = run_ragas_eval(rows, "raw")
    rew_scores = run_ragas_eval(rows, "rew")

    # ── 输出 ──
    print_comparison(raw_scores, rew_scores, args.top_k)

    result_json = str(results_dir / f"rag_eval_ragas_{ts}.json")
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump({"raw": raw_scores, "rew": rew_scores,
                   "top_k": args.top_k, "n_samples": len(rows)},
                  f, ensure_ascii=False, indent=2)
    print(f"[info] 汇总已保存 → {result_json}")

    md_path = result_json.replace(".json", ".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(build_md_report(raw_scores, rew_scores, rows, args.top_k))
    print(f"[info] 报告已保存 → {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGAS RAG 检索质量评测")
    parser.add_argument("--docs-dir",     default=_DEFAULT_DOCS_DIR,
                        help=f"知识库 .md 文档目录，默认 {_DEFAULT_DOCS_DIR}")
    parser.add_argument("--testset-size", type=int, default=20,
                        help="生成测试集大小（默认 20）")
    parser.add_argument("--top-k",        type=int, default=5,
                        help="Qdrant 检索返回条数（默认 5）")
    parser.add_argument("--load-testset", default=None,
                        help="跳过生成步骤，直接加载已有测试集 JSON 路径")
    args = parser.parse_args()
    asyncio.run(main(args))
