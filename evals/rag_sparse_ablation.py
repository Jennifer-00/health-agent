"""
evals/rag_sparse_ablation.py — 稀疏向量消融实验

对比三种检索策略（均使用改写后 query，不含 cross-encoder 重排）：
  dense_only   : 仅语义向量（1024-dim BGE-M3 dense）
  sparse_only  : 仅稀疏向量（BGE-M3 lexical_weights）
  hybrid_rrf   : dense + sparse 双路 Prefetch + RRF（生产方案）

测试集：
  手写 40 条（rag_retrieval.CASES）+ 自动生成 N 条（evals/results/generated_cases.json）
  相关性判断：关键词匹配（relevant_terms）；
              若有 ground_truth_id 则同时做精确匹配（命中原始 chunk 视为相关）

运行：
  python evals/rag_sparse_ablation.py
  python evals/rag_sparse_ablation.py --top-k 10 --no-generated  # 只跑手写 40 条
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
from dotenv import load_dotenv
load_dotenv(override=True)

from evals.rag_retrieval import CASES as HANDWRITTEN_CASES, RagCase, _is_relevant


@dataclass
class EvalCase:
    id: int
    query: str
    relevant_terms: list[str]
    ground_truth_id: str = ""    # Qdrant point UUID，精确匹配用
    note: str = ""
    source: str = "handwritten"  # handwritten | generated


def load_all_cases(include_generated: bool = True) -> list[EvalCase]:
    cases = [
        EvalCase(id=c.id, query=c.query, relevant_terms=c.relevant_terms,
                 note=c.note, source="handwritten")
        for c in HANDWRITTEN_CASES
    ]
    if not include_generated:
        return cases

    gen_path = Path(__file__).parent / "results" / "generated_cases.json"
    if not gen_path.exists():
        print(f"[WARN] 未找到生成的测试集 {gen_path}，只使用手写 40 条")
        return cases

    generated = json.loads(gen_path.read_text(encoding="utf-8"))
    for g in generated:
        cases.append(EvalCase(
            id=g["id"],
            query=g["query"],
            relevant_terms=g.get("relevant_terms", []),
            ground_truth_id=g.get("ground_truth_id", ""),
            note=g.get("note", ""),
            source="generated",
        ))
    print(f"[info] 测试集：手写 {len(HANDWRITTEN_CASES)} 条 + 生成 {len(generated)} 条 = {len(cases)} 条")
    return cases


async def _rewrite_query(query: str) -> str:
    import os
    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )
    resp = await client.chat.completions.create(
        model="deepseek-chat",
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
    return _embed_model


def _get_qdrant():
    import os
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import AsyncQdrantClient
        host = os.getenv("QDRANT_HOST", "localhost")
        _qdrant_client = AsyncQdrantClient(host=host, port=6333, prefer_grpc=False)
    return _qdrant_client


async def _encode(query: str) -> tuple[list[float], dict]:
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
    # fp16 → float32，避免 Qdrant 客户端类型不匹配
    import numpy as np
    dense = enc["dense_vecs"].astype(np.float32).tolist()
    return dense, enc["lexical_weights"]


def _to_hits(resp) -> list[dict]:
    results = []
    for hit in resp.points:
        p = hit.payload or {}
        combined = " ".join(str(v) for v in [
            p.get("content", ""), p.get("book", ""),
            p.get("h1", ""), p.get("h2", ""), p.get("h3", ""), p.get("h4", ""),
        ])
        results.append({"point_id": str(hit.id), "payload": p, "combined_text": combined})
    return results


async def _search_dense(dense_vec: list[float], top_k: int) -> list[dict]:
    resp = await _get_qdrant().query_points(
        collection_name=_COLLECTION,
        query=dense_vec,
        using="dense",
        limit=top_k,
        with_payload=True,
    )
    return _to_hits(resp)


async def _search_sparse(sparse_dict: dict, top_k: int) -> list[dict]:
    from qdrant_client.models import SparseVector
    sparse_vec = SparseVector(
        indices=[int(k) for k in sparse_dict.keys()],
        values=[float(v) for v in sparse_dict.values()],
    )
    resp = await _get_qdrant().query_points(
        collection_name=_COLLECTION,
        query=sparse_vec,
        using="sparse",
        limit=top_k,
        with_payload=True,
    )
    return _to_hits(resp)


async def _search_hybrid(dense_vec: list[float], sparse_dict: dict, top_k: int) -> list[dict]:
    from qdrant_client.models import SparseVector, Prefetch, FusionQuery, Fusion
    sparse_vec = SparseVector(
        indices=[int(k) for k in sparse_dict.keys()],
        values=[float(v) for v in sparse_dict.values()],
    )
    resp = await _get_qdrant().query_points(
        collection_name=_COLLECTION,
        prefetch=[
            Prefetch(query=dense_vec,  using="dense",  limit=_PREFETCH_LIMIT),
            Prefetch(query=sparse_vec, using="sparse", limit=_PREFETCH_LIMIT),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    return _to_hits(resp)


def _is_relevant_doc(doc: dict, case: EvalCase) -> bool:
    if case.ground_truth_id and doc["point_id"] == case.ground_truth_id:
        return True
    return _is_relevant(doc, case.relevant_terms)


def _metrics(docs: list[dict], case: EvalCase) -> dict:
    relevance = [_is_relevant_doc(d, case) for d in docs]
    hit_k = int(any(relevance))
    mrr = next((1.0 / (i + 1) for i, r in enumerate(relevance) if r), 0.0)
    prec_k = sum(relevance) / len(docs) if docs else 0.0
    return {"hit_k": hit_k, "mrr": mrr, "prec_k": prec_k}


async def evaluate_case(case: EvalCase, top_k: int) -> dict:
    # query 改写（固定变量）
    rewritten = await _rewrite_query(case.query)

    # 编码一次，三路共用
    dense_vec, sparse_dict = await _encode(rewritten)

    # 三路检索并行发出
    t0 = time.monotonic()
    results_dense, results_sparse, results_hybrid = await asyncio.gather(
        _search_dense(dense_vec, top_k),
        _search_sparse(sparse_dict, top_k),
        _search_hybrid(dense_vec, sparse_dict, top_k),
    )
    search_ms = int((time.monotonic() - t0) * 1000)

    m_dense  = _metrics(results_dense,  case)
    m_sparse = _metrics(results_sparse, case)
    m_hybrid = _metrics(results_hybrid, case)

    return {
        "id":        case.id,
        "query":     case.query,
        "rewritten": rewritten,
        "note":      case.note,
        "top_k":     top_k,
        "search_ms": search_ms,
        # dense only
        "dense_hit":  m_dense["hit_k"],
        "dense_mrr":  m_dense["mrr"],
        "dense_prec": m_dense["prec_k"],
        # sparse only
        "sparse_hit":  m_sparse["hit_k"],
        "sparse_mrr":  m_sparse["mrr"],
        "sparse_prec": m_sparse["prec_k"],
        # hybrid RRF
        "hybrid_hit":  m_hybrid["hit_k"],
        "hybrid_mrr":  m_hybrid["mrr"],
        "hybrid_prec": m_hybrid["prec_k"],
        # hybrid vs dense delta（核心关注指标）
        "delta_hit_vs_dense": m_hybrid["hit_k"] - m_dense["hit_k"],
        "delta_mrr_vs_dense": m_hybrid["mrr"]   - m_dense["mrr"],
    }


async def run_all(top_k: int, concurrency: int, include_generated: bool = True) -> list[dict]:
    cases = load_all_cases(include_generated)
    sem = asyncio.Semaphore(concurrency)

    async def safe_eval(case: EvalCase):
        async with sem:
            try:
                return await evaluate_case(case, top_k)
            except Exception as exc:
                return {"id": case.id, "query": case.query, "source": case.source, "error": str(exc)}

    print(f"[info] 共 {len(cases)} 条，top_k={top_k}，并发={concurrency}")
    t0 = time.monotonic()
    results = await asyncio.gather(*[safe_eval(c) for c in cases])
    print(f"[info] 完成，总耗时 {time.monotonic() - t0:.1f}s")
    return list(results)


def _avg(vals):
    return sum(vals) / len(vals) if vals else 0.0


def print_report(results: list[dict], top_k: int) -> None:
    valid  = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    print("\n" + "=" * 70)
    print("  RAG 稀疏向量消融实验报告")
    print("=" * 70)
    print(f"  总案例: {len(results)}  有效: {len(valid)}  出错: {len(errors)}  top_k={top_k}")
    print(f"  固定变量: query 改写（统一用改写后）、关键词相关性判断")
    print(f"  不含 cross-encoder 重排（孤立检索阶段效果）")
    print()

    # 全量汇总
    print(f"{'指标':<12} {'dense_only':>12} {'sparse_only':>12} {'hybrid_rrf':>12}  {'Δ hybrid-dense':>14}")
    print("─" * 70)
    for metric, key in [("Hit@K", "hit"), ("MRR@K", "mrr"), ("Prec@K", "prec")]:
        d = _avg([r[f"dense_{key}"]  for r in valid])
        s = _avg([r[f"sparse_{key}"] for r in valid])
        h = _avg([r[f"hybrid_{key}"] for r in valid])
        print(f"  {metric:<10} {d:>12.3f} {s:>12.3f} {h:>12.3f}  {h-d:>+14.3f}")

    print()

    # 分类汇总
    groups = [
        ("手写-口语化",  [r for r in valid if r.get("source","handwritten")=="handwritten" and r["id"] <= 23]),
        ("手写-诊断",    [r for r in valid if r.get("source","handwritten")=="handwritten" and 24 <= r["id"] <= 30]),
        ("手写-规范",    [r for r in valid if r.get("source","handwritten")=="handwritten" and r["id"] >= 31]),
        ("自动生成",     [r for r in valid if r.get("source") == "generated"]),
    ]
    print(f"  {'类型':<12} {'n':>4} {'dense Hit':>10} {'sparse Hit':>11} {'hybrid Hit':>11} {'ΔMRR h-d':>10}")
    print("  " + "─" * 65)
    for label, group in groups:
        if not group:
            continue
        dh = _avg([r["dense_hit"]  for r in group])
        sh = _avg([r["sparse_hit"] for r in group])
        hh = _avg([r["hybrid_hit"] for r in group])
        dm = _avg([r["dense_mrr"]  for r in group])
        hm = _avg([r["hybrid_mrr"] for r in group])
        print(f"  {label:<12} {len(group):>4} {dh:>10.3f} {sh:>11.3f} {hh:>11.3f} {hm-dm:>+10.3f}")

    print()

    # hybrid 相对 dense 改善最多的案例
    improved = sorted(valid, key=lambda r: r["delta_mrr_vs_dense"], reverse=True)[:5]
    print("  hybrid 比 dense MRR 改善最多（Top 5）— 稀疏向量贡献显著的场景")
    print("  " + "─" * 65)
    for r in improved:
        print(f"  #{r['id']:02d} ΔMRR={r['delta_mrr_vs_dense']:+.3f} "
              f"dense={r['dense_hit']} sparse={r['sparse_hit']} hybrid={r['hybrid_hit']}"
              f"  {r['query'][:28]}")
        print(f"      改写: {r['rewritten'][:50]}")

    print()

    # hybrid 比 dense 更差的案例（RRF 引入噪声的情况）
    degraded = [r for r in valid if r["delta_mrr_vs_dense"] < -0.001]
    if degraded:
        print(f"  hybrid 比 dense MRR 下降的案例（{len(degraded)} 条）— 稀疏引入噪声")
        print("  " + "─" * 65)
        for r in sorted(degraded, key=lambda x: x["delta_mrr_vs_dense"])[:5]:
            print(f"  #{r['id']:02d} ΔMRR={r['delta_mrr_vs_dense']:+.3f}  {r['query'][:35]}")

    if errors:
        print(f"\n  出错: {[r['id'] for r in errors]}")

    print("\n" + "=" * 70)


def build_md_report(results: list[dict], top_k: int) -> str:
    valid = [r for r in results if "error" not in r]
    lines = []
    lines.append("# RAG 稀疏向量消融实验报告\n")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  top_k={top_k}\n")
    lines.append("**固定变量**：query 统一使用改写后版本，相关性判断用关键词匹配，不含 cross-encoder 重排。\n")

    lines.append("## 总体指标对比\n")
    lines.append("| 指标 | dense_only | sparse_only | hybrid_rrf | Δ hybrid-dense |")
    lines.append("|------|----------:|------------:|-----------:|---------------:|")
    for metric, key in [("Hit@K", "hit"), ("MRR@K", "mrr"), ("Prec@K", "prec")]:
        d = _avg([r[f"dense_{key}"]  for r in valid])
        s = _avg([r[f"sparse_{key}"] for r in valid])
        h = _avg([r[f"hybrid_{key}"] for r in valid])
        lines.append(f"| {metric} | {d:.3f} | {s:.3f} | {h:.3f} | **{h-d:+.3f}** |")
    lines.append("")

    lines.append("## 按查询类型细分\n")
    lines.append("| 类型 | n | dense Hit | sparse Hit | hybrid Hit | ΔMRR hybrid-dense |")
    lines.append("|------|--:|----------:|-----------:|-----------:|------------------:|")
    for label, lo, hi in [("口语化", 1, 23), ("疾病/诊断", 24, 30), ("已规范对照", 31, 40)]:
        g = [r for r in valid if lo <= r["id"] <= hi]
        if not g:
            continue
        lines.append(
            f"| {label} | {len(g)} "
            f"| {_avg([r['dense_hit'] for r in g]):.3f} "
            f"| {_avg([r['sparse_hit'] for r in g]):.3f} "
            f"| {_avg([r['hybrid_hit'] for r in g]):.3f} "
            f"| {_avg([r['hybrid_mrr'] for r in g]) - _avg([r['dense_mrr'] for r in g]):+.3f} |"
        )
    lines.append("")

    lines.append("## 逐条明细\n")
    lines.append("| # | 改写后 query | dense Hit | sparse Hit | hybrid Hit | ΔMRR |")
    lines.append("|---|------------|----------:|-----------:|-----------:|-----:|")
    for r in valid:
        rw = r["rewritten"][:32] + ("…" if len(r["rewritten"]) > 32 else "")
        lines.append(
            f"| {r['id']:02d} | {rw} "
            f"| {r['dense_hit']} | {r['sparse_hit']} | {r['hybrid_hit']} "
            f"| {r['delta_mrr_vs_dense']:+.3f} |"
        )
    return "\n".join(lines) + "\n"


async def main(top_k: int, concurrency: int, no_generated: bool) -> None:
    results = await run_all(top_k, concurrency, include_generated=not no_generated)
    print_report(results, top_k)

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"sparse_ablation_{ts}.json"
    md_path   = out_dir / f"sparse_ablation_{ts}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(build_md_report(results, top_k))

    print(f"[info] JSON → {json_path}")
    print(f"[info] 报告 → {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k",        type=int,  default=5,     help="检索返回条数（默认5）")
    parser.add_argument("--concurrency",  type=int,  default=1,     help="并发数（默认1，避免编码竞争）")
    parser.add_argument("--no-generated", action="store_true",      help="只跑手写40条，不加载生成测试集")
    args = parser.parse_args()
    asyncio.run(main(args.top_k, args.concurrency, args.no_generated))
