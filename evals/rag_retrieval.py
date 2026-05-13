"""
evals/rag_retrieval.py — RAG 检索质量评测（query 重写前后对比）

指标：
  - Hit@K   : top-K 中是否命中至少一个相关文档（在无完整标注时作为召回率代理）
  - MRR@K   : Mean Reciprocal Rank（第一个相关结果排名倒数的均值）
  - Prec@K  : top-K 结果中相关文档的比例

相关性判断：关键词匹配——结果 payload（content + 各级标题 + book 字段）
            包含 relevant_terms 中至少一个词，即视为相关。

运行：
  python evals/rag_retrieval.py
  python evals/rag_retrieval.py --top-k 5 --concurrency 3
  python evals/rag_retrieval.py --top-k 10 --no-rewrite-only
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


# ── 测试用例定义 ───────────────────────────────────────────────────────────────

@dataclass
class RagCase:
    id: int
    query: str              # 原始口语化查询
    relevant_terms: list[str]  # payload 中应出现的相关医学术语（ANY 匹配即为相关）
    note: str = ""          # 说明重写预期能改善的原因


CASES: list[RagCase] = [

    # ══════════════════════════════════════════════════════════════════════════
    # 口语化 → 医学术语（重写最应该有效果的场景）
    # ══════════════════════════════════════════════════════════════════════════

    RagCase(1,  "拉肚子怎么办",
            ["腹泻", "腹泻治疗", "止泻", "肠炎", "洛哌丁胺", "口服补液"],
            "口语「拉肚子」→医学「腹泻」"),

    RagCase(2,  "睡不着觉是什么原因",
            ["失眠", "睡眠障碍", "入睡困难", "失眠症"],
            "口语「睡不着」→医学「失眠」"),

    RagCase(3,  "肚子痛的很厉害",
            ["腹痛", "急腹症", "腹部疼痛", "腹绞痛"],
            "口语「肚子痛」→医学「腹痛」"),

    RagCase(4,  "胃烧心很难受",
            ["胃灼热", "烧心", "反流", "胃食管反流", "GERD"],
            "口语「烧心」→医学「胃灼热/反流」"),

    RagCase(5,  "嗓子哑说不出话",
            ["声音嘶哑", "喉炎", "声带", "咽喉炎", "声嘶"],
            "口语「嗓子哑」→医学「声音嘶哑」"),

    RagCase(6,  "头晕眼花站不稳",
            ["眩晕", "头晕", "眩晕症", "前庭", "位置性眩晕"],
            "口语「头晕眼花」→医学「眩晕」"),

    RagCase(7,  "喘不上气",
            ["呼吸困难", "气促", "呼吸急促", "哮喘", "气短"],
            "口语「喘不上气」→医学「呼吸困难」"),

    RagCase(8,  "皮肤起疹子好痒",
            ["皮疹", "荨麻疹", "皮炎", "过敏性皮炎", "瘙痒"],
            "口语「起疹子」→医学「皮疹/荨麻疹」"),

    RagCase(9,  "便秘好多天了",
            ["便秘", "排便困难", "肠道蠕动", "通便"],
            "便秘——口语和医学术语相同但重写可加上诊断标准等语境"),

    RagCase(10, "心跳乱跳忽快忽慢",
            ["心律失常", "心悸", "心房颤动", "早搏", "心律不齐"],
            "口语「心跳乱跳」→医学「心律失常」"),

    RagCase(11, "腿肿了",
            ["下肢水肿", "腿部水肿", "水肿", "静脉曲张", "深静脉血栓"],
            "口语「腿肿」→医学「下肢水肿」"),

    RagCase(12, "眼睛里有小虫子飞来飞去",
            ["飞蚊症", "玻璃体混浊", "视网膜", "玻璃体"],
            "口语「小虫子飞」→医学「飞蚊症」"),

    RagCase(13, "手发麻",
            ["手部麻木", "肢体麻木", "末梢神经", "腕管综合征", "神经病变"],
            "口语「手发麻」→医学「肢体麻木/腕管综合征」"),

    RagCase(14, "尿频尿急",
            ["尿频", "尿急", "膀胱炎", "尿路感染", "下尿路"],
            "症状描述可标准化"),

    RagCase(15, "拔牙后血还没止住",
            ["拔牙术后出血", "口腔出血", "凝血", "牙槽出血"],
            "口语「血没止住」→医学「术后出血/凝血」"),

    # ══════════════════════════════════════════════════════════════════════════
    # 药物查询（口语化药物名 / 俗称）
    # ══════════════════════════════════════════════════════════════════════════

    RagCase(16, "退烧药什么时候吃",
            ["退烧药", "解热镇痛", "对乙酰氨基酚", "布洛芬", "发热"],
            "「退烧药」→标准药物通用名"),

    RagCase(17, "消炎药需要吃多少天",
            ["抗生素", "消炎药", "抗菌药", "疗程", "用药天数"],
            "「消炎药」→「抗生素/抗菌药」"),

    RagCase(18, "血压药忘吃了怎么补",
            ["降压药", "抗高血压", "漏服", "补服", "血压"],
            "「血压药」→「降压药」"),

    RagCase(19, "降糖药能和感冒药一起吃吗",
            ["降糖药", "抗糖尿病药", "药物相互作用", "二甲双胍"],
            "药物相互作用查询"),

    RagCase(20, "安眠药吃多了有危险吗",
            ["镇静催眠药", "安眠药", "苯二氮䓬", "过量", "中毒"],
            "「安眠药」→医学「镇静催眠药」"),

    RagCase(21, "止痛药胃不好能吃吗",
            ["非甾体抗炎药", "NSAIDs", "胃肠道副作用", "布洛芬", "胃溃疡"],
            "「止痛药胃不好」→NSAID 胃肠道禁忌"),

    RagCase(22, "胃药饭前还是饭后吃",
            ["质子泵抑制剂", "抑酸药", "奥美拉唑", "饭前", "服用时机"],
            "「胃药」→PPI/H2受体阻断剂用法"),

    RagCase(23, "止咳药小孩能吃吗",
            ["止咳药", "儿童用药", "右美沙芬", "可待因", "小儿"],
            "儿童用药安全查询"),

    # ══════════════════════════════════════════════════════════════════════════
    # 疾病描述 / 诊断查询
    # ══════════════════════════════════════════════════════════════════════════

    RagCase(24, "血压高多少才算高血压",
            ["高血压", "血压标准", "收缩压", "舒张压", "诊断标准"],
            "数值型诊断标准查询"),

    RagCase(25, "血糖多少算糖尿病",
            ["糖尿病", "血糖", "空腹血糖", "诊断标准", "葡萄糖耐量"],
            "数值型诊断标准"),

    RagCase(26, "甲状腺功能低下是什么意思",
            ["甲减", "甲状腺功能减退", "TSH", "甲状腺激素"],
            "口语「甲状腺功能低下」→医学「甲减」"),

    RagCase(27, "胆固醇高会怎样",
            ["高脂血症", "高胆固醇", "低密度脂蛋白", "动脉粥样硬化", "血脂"],
            "「胆固醇高」→「高血脂症」相关后果"),

    RagCase(28, "痛风发作脚很痛怎么缓解",
            ["痛风", "急性痛风", "尿酸", "秋水仙碱", "消炎止痛"],
            "痛风急性期处理"),

    RagCase(29, "哮喘发作时怎么急救",
            ["哮喘", "支气管哮喘", "急性发作", "支气管扩张剂", "沙丁胺醇"],
            "哮喘急救——重写可精准到「支气管哮喘急性发作处理」"),

    RagCase(30, "贫血吃什么好得快",
            ["贫血", "缺铁性贫血", "铁剂", "补铁", "血红蛋白"],
            "「贫血吃什么」→「缺铁性贫血 铁剂补充」"),

    # ══════════════════════════════════════════════════════════════════════════
    # 已较规范的查询（重写应不降低效果，作为对照基线）
    # ══════════════════════════════════════════════════════════════════════════

    RagCase(31, "布洛芬常见副作用",
            ["布洛芬", "副作用", "不良反应", "胃肠道"],
            "已规范——重写不应降低效果"),

    RagCase(32, "二甲双胍的用法与剂量",
            ["二甲双胍", "用法", "剂量", "服用方法"],
            "已规范"),

    RagCase(33, "高血压饮食注意事项",
            ["高血压", "饮食", "低盐", "限钠", "DASH"],
            "已规范"),

    RagCase(34, "阿莫西林和头孢过敏有交叉反应吗",
            ["青霉素", "头孢", "交叉过敏", "过敏反应", "阿莫西林"],
            "已规范"),

    RagCase(35, "华法林和阿司匹林能同时服用吗",
            ["华法林", "阿司匹林", "相互作用", "出血风险", "抗凝"],
            "已规范"),

    RagCase(36, "2型糖尿病治疗原则",
            ["2型糖尿病", "糖尿病", "治疗原则", "血糖控制", "胰岛素"],
            "已规范"),

    RagCase(37, "痛风饮食禁忌",
            ["痛风", "嘌呤", "饮食", "高嘌呤", "尿酸"],
            "已规范"),

    RagCase(38, "类风湿关节炎的诊断标准",
            ["类风湿关节炎", "诊断标准", "类风湿因子", "关节炎"],
            "已规范"),

    RagCase(39, "高血脂一线治疗药物",
            ["高血脂", "他汀", "他汀类", "降脂药", "胆固醇"],
            "已规范"),

    RagCase(40, "哮喘长期用药方案",
            ["哮喘", "吸入激素", "长期管理", "支气管扩张", "维持治疗"],
            "已规范"),
]


# ── 检索逻辑 ───────────────────────────────────────────────────────────────────

_QUERY_PREFIX   = "为这个句子生成表示以用于检索相关文章："
_COLLECTION     = "medical_knowledge"
_EMBED_MODEL    = "BAAI/bge-m3"
_PREFETCH_LIMIT = 20

_embed_model  = None
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
        port = int(os.getenv("QDRANT_PORT", "6334"))
        _qdrant_client = AsyncQdrantClient(host=host, port=port, prefer_grpc=True)
    return _qdrant_client


async def _encode_query(query: str) -> tuple[list[float], dict]:
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
    return enc["dense_vecs"].tolist(), enc["lexical_weights"]


async def _rewrite_query(query: str) -> str:
    import os
    import anthropic as _anthropic
    client = _anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=128,
        system=(
            "你是医疗检索优化器。将用户输入改写为适合向量数据库检索的标准医疗术语查询，"
            "只输出改写后的查询词，不加任何解释。"
        ),
        messages=[{"role": "user", "content": query}],
    )
    return resp.content[0].text.strip()


async def _qdrant_search(query_text: str, top_k: int) -> list[dict]:
    """对给定查询文本执行双路 Prefetch + RRF 检索，返回 payload 列表（已排序）。"""
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
    results = []
    for hit in resp.points:
        p = hit.payload or {}
        combined = " ".join(str(v) for v in [
            p.get("content", ""), p.get("book", ""),
            p.get("h1", ""), p.get("h2", ""), p.get("h3", ""), p.get("h4", ""),
        ])
        results.append({"payload": p, "combined_text": combined})
    return results


def _is_relevant(doc: dict, relevant_terms: list[str]) -> bool:
    """关键词匹配：combined_text 包含 relevant_terms 中任意一个词即为相关。"""
    text = doc["combined_text"].lower()
    return any(term.lower() in text for term in relevant_terms)


# ── 单条评测 ───────────────────────────────────────────────────────────────────

async def evaluate_case(case: RagCase, top_k: int) -> dict:
    t0 = time.monotonic()
    rewritten = await _rewrite_query(case.query)
    rewrite_ms = (time.monotonic() - t0) * 1000

    # 两路并行检索
    t1 = time.monotonic()
    results_raw, results_rew = await asyncio.gather(
        _qdrant_search(case.query, top_k),
        _qdrant_search(rewritten,  top_k),
    )
    search_ms = (time.monotonic() - t1) * 1000

    def compute_metrics(docs: list[dict]) -> dict:
        relevance = [_is_relevant(d, case.relevant_terms) for d in docs]
        hit_k = any(relevance)

        # MRR: 1/rank of first relevant (0 if not found)
        mrr = 0.0
        for rank, rel in enumerate(relevance, 1):
            if rel:
                mrr = 1.0 / rank
                break

        # Precision@K
        prec_k = sum(relevance) / len(docs) if docs else 0.0

        return {
            "hit_k":     int(hit_k),
            "mrr":       mrr,
            "prec_k":    prec_k,
            "relevance": relevance,
        }

    m_raw = compute_metrics(results_raw)
    m_rew = compute_metrics(results_rew)

    return {
        "id":          case.id,
        "query":       case.query,
        "rewritten":   rewritten,
        "note":        case.note,
        "top_k":       top_k,
        "rewrite_ms":  rewrite_ms,
        "search_ms":   search_ms,
        # 重写前
        "raw_hit_k":   m_raw["hit_k"],
        "raw_mrr":     m_raw["mrr"],
        "raw_prec_k":  m_raw["prec_k"],
        "raw_relevance": m_raw["relevance"],
        # 重写后
        "rew_hit_k":   m_rew["hit_k"],
        "rew_mrr":     m_rew["mrr"],
        "rew_prec_k":  m_rew["prec_k"],
        "rew_relevance": m_rew["relevance"],
        # Delta（正值 = 重写改善）
        "delta_hit":   m_rew["hit_k"]  - m_raw["hit_k"],
        "delta_mrr":   m_rew["mrr"]    - m_raw["mrr"],
        "delta_prec":  m_rew["prec_k"] - m_raw["prec_k"],
    }


# ── 批量运行 ───────────────────────────────────────────────────────────────────

async def run_all(top_k: int, concurrency: int) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)

    async def safe_eval(case: RagCase):
        async with sem:
            try:
                return await evaluate_case(case, top_k)
            except Exception as exc:
                return {
                    "id": case.id, "query": case.query, "rewritten": "",
                    "note": case.note, "top_k": top_k,
                    "rewrite_ms": 0, "search_ms": 0,
                    "raw_hit_k": 0, "raw_mrr": 0.0, "raw_prec_k": 0.0, "raw_relevance": [],
                    "rew_hit_k": 0, "rew_mrr": 0.0, "rew_prec_k": 0.0, "rew_relevance": [],
                    "delta_hit": 0, "delta_mrr": 0.0, "delta_prec": 0.0,
                    "error": str(exc),
                }

    print(f"[info] 共 {len(CASES)} 条，top_k={top_k}，并发={concurrency}，开始评测…")
    t0 = time.monotonic()
    results = await asyncio.gather(*[safe_eval(c) for c in CASES])
    total_s = time.monotonic() - t0
    print(f"[info] 完成，总耗时 {total_s:.1f}s")
    return list(results)


# ── 报告打印 ───────────────────────────────────────────────────────────────────

def _avg(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def print_report(results: list[dict], top_k: int) -> None:
    valid = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    raw_hit  = _avg([r["raw_hit_k"]  for r in valid])
    raw_mrr  = _avg([r["raw_mrr"]    for r in valid])
    raw_prec = _avg([r["raw_prec_k"] for r in valid])
    rew_hit  = _avg([r["rew_hit_k"]  for r in valid])
    rew_mrr  = _avg([r["rew_mrr"]    for r in valid])
    rew_prec = _avg([r["rew_prec_k"] for r in valid])

    print("\n" + "=" * 70)
    print("  RAG 检索质量评测报告（query 重写前 vs 后）")
    print("=" * 70)
    print(f"  总案例: {len(results)}   有效: {len(valid)}   执行出错: {len(errors)}")
    print(f"  top_k = {top_k}")
    print()

    print(f"{'指标':<12}  {'重写前':>8}  {'重写后':>8}  {'Delta':>8}")
    print("─" * 50)
    print(f"  {'Hit@K':<10}  {raw_hit:>8.3f}  {rew_hit:>8.3f}  {rew_hit-raw_hit:>+8.3f}")
    print(f"  {'MRR@K':<10}  {raw_mrr:>8.3f}  {rew_mrr:>8.3f}  {rew_mrr-raw_mrr:>+8.3f}")
    print(f"  {'Prec@K':<10}  {raw_prec:>8.3f}  {rew_prec:>8.3f}  {rew_prec-raw_prec:>+8.3f}")
    print()

    # 分类汇总：口语化改善 vs 已规范对照
    colloquial = [r for r in valid if r["id"] <= 23]
    baseline   = [r for r in valid if r["id"] >= 31]

    if colloquial:
        c_raw_hit = _avg([r["raw_hit_k"] for r in colloquial])
        c_rew_hit = _avg([r["rew_hit_k"] for r in colloquial])
        c_raw_mrr = _avg([r["raw_mrr"]   for r in colloquial])
        c_rew_mrr = _avg([r["rew_mrr"]   for r in colloquial])
        print(f"  [口语化查询 n={len(colloquial)}]  Hit@K {c_raw_hit:.3f}→{c_rew_hit:.3f}  MRR {c_raw_mrr:.3f}→{c_rew_mrr:.3f}")

    if baseline:
        b_raw_hit = _avg([r["raw_hit_k"] for r in baseline])
        b_rew_hit = _avg([r["rew_hit_k"] for r in baseline])
        b_raw_mrr = _avg([r["raw_mrr"]   for r in baseline])
        b_rew_mrr = _avg([r["rew_mrr"]   for r in baseline])
        print(f"  [已规范查询 n={len(baseline)}]  Hit@K {b_raw_hit:.3f}→{b_rew_hit:.3f}  MRR {b_raw_mrr:.3f}→{b_rew_mrr:.3f}")

    print()

    # 重写改善最多的案例
    improved = sorted(valid, key=lambda r: r["delta_mrr"], reverse=True)[:5]
    print("  MRR 改善最多（Top 5）")
    print("  " + "─" * 65)
    for r in improved:
        print(f"  #{r['id']:02d} delta_mrr={r['delta_mrr']:+.3f}  {r['query'][:30]}")
        print(f"      重写: {r['rewritten'][:50]}")

    print()

    # 重写有损害的案例
    degraded = [r for r in valid if r["delta_mrr"] < -0.001]
    if degraded:
        print(f"  重写后 MRR 下降的案例（{len(degraded)} 条）")
        print("  " + "─" * 65)
        for r in sorted(degraded, key=lambda x: x["delta_mrr"])[:8]:
            print(f"  #{r['id']:02d} delta_mrr={r['delta_mrr']:+.3f}  {r['query'][:30]}")
            print(f"      重写: {r['rewritten'][:50]}")

    if errors:
        print(f"\n  执行出错案例：")
        for r in errors:
            print(f"  #{r['id']:02d} {r['query'][:40]}  err={r['error'][:60]}")

    print("\n" + "=" * 70)


# ── Markdown 报告 ─────────────────────────────────────────────────────────────

def build_md_report(results: list[dict], top_k: int) -> str:
    valid  = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    raw_hit  = _avg([r["raw_hit_k"]  for r in valid])
    raw_mrr  = _avg([r["raw_mrr"]    for r in valid])
    raw_prec = _avg([r["raw_prec_k"] for r in valid])
    rew_hit  = _avg([r["rew_hit_k"]  for r in valid])
    rew_mrr  = _avg([r["rew_mrr"]    for r in valid])
    rew_prec = _avg([r["rew_prec_k"] for r in valid])

    lines = []
    lines.append("# RAG 检索质量评测报告\n")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  top_k = {top_k}\n")

    lines.append("## 总体指标对比\n")
    lines.append("| 指标 | 重写前 | 重写后 | Delta |")
    lines.append("|------|-------:|-------:|------:|")
    lines.append(f"| Hit@{top_k} | {raw_hit:.3f} | {rew_hit:.3f} | **{rew_hit-raw_hit:+.3f}** |")
    lines.append(f"| MRR@{top_k} | {raw_mrr:.3f} | {rew_mrr:.3f} | **{rew_mrr-raw_mrr:+.3f}** |")
    lines.append(f"| Prec@{top_k} | {raw_prec:.3f} | {rew_prec:.3f} | **{rew_prec-raw_prec:+.3f}** |")
    if errors:
        lines.append(f"| 出错 | — | — | {len(errors)} 条 |")
    lines.append("")

    # 分类对比
    colloquial = [r for r in valid if r["id"] <= 23]
    disease    = [r for r in valid if 24 <= r["id"] <= 30]
    baseline   = [r for r in valid if r["id"] >= 31]

    lines.append("## 按查询类型细分\n")
    lines.append("| 类型 | n | Hit 前 | Hit 后 | MRR 前 | MRR 后 |")
    lines.append("|------|--:|-------:|-------:|-------:|-------:|")
    for label, group in [("口语化", colloquial), ("疾病/诊断", disease), ("已规范对照", baseline)]:
        if not group:
            continue
        lines.append(
            f"| {label} | {len(group)} "
            f"| {_avg([r['raw_hit_k'] for r in group]):.3f} "
            f"| {_avg([r['rew_hit_k'] for r in group]):.3f} "
            f"| {_avg([r['raw_mrr']   for r in group]):.3f} "
            f"| {_avg([r['rew_mrr']   for r in group]):.3f} |"
        )
    lines.append("")

    # 逐条明细
    lines.append("## 逐条明细\n")
    lines.append(f"| # | 原始查询 | 重写后 | Hit前 | Hit后 | MRR前 | MRR后 | ΔMRR |")
    lines.append("|---|---------|--------|------:|------:|------:|------:|------:|")
    for r in valid:
        q   = r["query"][:28] + ("…" if len(r["query"]) > 28 else "")
        rw  = r["rewritten"][:30] + ("…" if len(r["rewritten"]) > 30 else "")
        lines.append(
            f"| {r['id']:02d} | {q} | {rw} "
            f"| {r['raw_hit_k']} | {r['rew_hit_k']} "
            f"| {r['raw_mrr']:.3f} | {r['rew_mrr']:.3f} "
            f"| {r['delta_mrr']:+.3f} |"
        )
    lines.append("")

    if errors:
        lines.append("## 出错案例\n")
        for r in errors:
            lines.append(f"- #{r['id']:02d} `{r['query']}` — `{r.get('error','')[:80]}`")
        lines.append("")

    return "\n".join(lines) + "\n"


# ── 主入口 ────────────────────────────────────────────────────────────────────

async def main(top_k: int, concurrency: int, output: str | None) -> None:
    results = await run_all(top_k, concurrency)
    print_report(results, top_k)

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output is None:
        output = str(results_dir / f"rag_retrieval_{ts}.json")

    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[info] JSON 已写入 {output}")

    md_path = output.replace(".json", ".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(build_md_report(results, top_k))
    print(f"[info] 报告已写入 {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k",      type=int, default=5,    help="检索返回条数（默认5）")
    parser.add_argument("--concurrency", type=int, default=3,   help="并发数（默认3，受制于BGE-M3内存）")
    parser.add_argument("--output",      type=str, default=None, help="结果 JSON 路径（默认自动生成）")
    args = parser.parse_args()
    asyncio.run(main(args.top_k, args.concurrency, args.output))
