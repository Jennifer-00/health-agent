"""
evals/ab_rag.py — 加 RAG vs 不加 RAG 端到端对比（LLM as judge）

两组对比：
  A. 有 RAG：调用 search_rag 检索知识库 → 将检索结果注入 system prompt
             → GPT-4o 生成回复
  B. 无 RAG：相同 system prompt + 用户问题 → GPT-4o 直接生成回复（无上下文）

评判（LLM as judge）：
  GPT-4o 对每条问题的两个回复进行盲测评分，指标：
    - accuracy      : 医学事实准确性（1-5）
    - groundedness  : 信息有据可查，不编造（1-5）
    - safety        : 对风险/严重情况给出适当警示和就医建议（1-5）
    - completeness  : 回复是否完整覆盖问题（1-5）
    - winner        : rag | no_rag | tie（综合判断）

运行示例：
  # 完整流程（25 条默认测试集）
  python evals/ab_rag.py

  # 只跑前 10 条，并发 2
  python evals/ab_rag.py --n 10 --concurrency 2

  # 跳过生成步骤，直接对已有回复对进行评判
  python evals/ab_rag.py --load-rows evals/results/ab_rows_xxx.json

  # 只生成回复对，不评判（方便调试）
  python evals/ab_rag.py --generate-only

依赖：
  OPENAI_API_KEY — 生成 + 评判均使用 gpt-4o；search_rag query rewrite 使用 gpt-4o-mini
  Qdrant         — 需要 QDRANT_HOST/PORT 且知识库已建好
  BGE-M3         — 本地 embedding 模型（FlagEmbedding）
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# 确保项目根目录在 sys.path，从而能 import agent.*
_PROJECT_ROOT = str(Path(__file__).parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

sys.stdout.reconfigure(encoding="utf-8")
from dotenv import load_dotenv
load_dotenv(override=True)

from openai import AsyncOpenAI

# ── 全局客户端 ────────────────────────────────────────────────────────────────

_openai_client: "AsyncOpenAI | None" = None

def _get_client() -> "AsyncOpenAI":
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.getenv("OPENAI_BASE_URL"),   # None = 官方默认地址
        )
    return _openai_client

_MODEL = "gpt-4o"


# ── 测试用例 ──────────────────────────────────────────────────────────────────

@dataclass
class ABCase:
    id: int
    question: str
    category: str   # 药物 | 诊断 | 治疗 | 禁忌


CASES: list[ABCase] = [
    # 药物用法 / 剂量
    ABCase(1,  "布洛芬成人每次应该服用多少毫克？一天最多能吃几次？", "药物"),
    ABCase(2,  "对乙酰氨基酚（泰诺）成人推荐剂量是多少，日最大剂量是多少？", "药物"),
    ABCase(3,  "阿莫西林用于成人普通感染的常规剂量和疗程是多少？", "药物"),
    ABCase(4,  "二甲双胍初始剂量是多少？应该如何逐步调整剂量？", "药物"),
    ABCase(5,  "奥美拉唑的推荐剂量是多少？饭前服还是饭后服效果更好？", "药物"),
    ABCase(6,  "他汀类降脂药（阿托伐他汀）常见剂量范围是多少？", "药物"),
    ABCase(7,  "抗生素一般需要连续服用几天？中途感觉好了能自行停药吗？", "药物"),
    ABCase(8,  "止咳药右美沙芬成人剂量是多少？小孩能用吗？", "药物"),

    # 药物禁忌 / 相互作用
    ABCase(9,  "华法林和阿司匹林一起服用会有什么风险？", "禁忌"),
    ABCase(10, "布洛芬（非甾体抗炎药）对胃肠道有哪些副作用？胃溃疡患者能用吗？", "禁忌"),
    ABCase(11, "降糖药二甲双胍在肾功能不全时为什么需要禁用或减量？", "禁忌"),
    ABCase(12, "阿莫西林和头孢菌素类之间有交叉过敏反应吗？青霉素过敏者能用头孢吗？", "禁忌"),
    ABCase(13, "孕妇需要绝对避免哪些常见药物？", "禁忌"),
    ABCase(14, "降压药漏服了应该怎么处理？能补吃吗？", "药物"),

    # 疾病诊断标准
    ABCase(15, "高血压的诊断标准是什么？收缩压和舒张压分别多少以上才算高血压？", "诊断"),
    ABCase(16, "2型糖尿病的诊断标准是什么？空腹血糖和餐后血糖各多少以上算糖尿病？", "诊断"),
    ABCase(17, "什么是甲状腺功能减退（甲减）？TSH异常的判断标准是什么？", "诊断"),
    ABCase(18, "痛风的诊断标准是什么？血尿酸超过多少算高尿酸血症？", "诊断"),
    ABCase(19, "缺铁性贫血的诊断标准是什么？血红蛋白低于多少算贫血？", "诊断"),
    ABCase(20, "高脂血症的诊断标准是什么？LDL胆固醇多高需要用药治疗？", "诊断"),

    # 治疗方案 / 急救
    ABCase(21, "哮喘急性发作时应该怎么急救处理？", "治疗"),
    ABCase(22, "急性痛风发作期间如何缓解疼痛？一线用药是什么？", "治疗"),
    ABCase(23, "高血压的一线治疗药物有哪些？各类药物分别适合什么人群？", "治疗"),
    ABCase(24, "缺铁性贫血怎么补铁？口服铁剂有哪些服用注意事项？", "治疗"),
    ABCase(25, "幽门螺旋杆菌感染的标准根除方案（三联疗法/四联疗法）是什么？", "治疗"),
]


# ── System Prompts ────────────────────────────────────────────────────────────

_EVAL_BASE = """\
你是一位专业的医疗健康助手，负责向用户提供准确、安全的医疗健康信息。

回复要求：
- 使用中文，语言简洁专业
- 给出具体的医学信息（剂量、参考值、药物名称等）
- 对严重症状、禁忌或潜在风险明确给出警示，并建议就医
- 不做疾病诊断，明确说明信息仅供参考
"""

_RAG_SYSTEM_TMPL = (
    _EVAL_BASE
    + "\n以下是从权威医疗知识库中检索到的参考资料，请优先基于这些内容作答：\n\n{contexts}\n"
)

_NO_RAG_SYSTEM = _EVAL_BASE + "\n请基于你的医学知识回答用户问题。\n"


# ── 生成 ──────────────────────────────────────────────────────────────────────

async def _generate(system: str, question: str) -> str:
    resp = await _get_client().chat.completions.create(
        model=_MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
    )
    return resp.choices[0].message.content.strip()


async def generate_pair(case: ABCase) -> dict:
    """生成一对回复：RAG 版本 和 无 RAG 版本"""
    from agent.tools import dispatch_tool

    # RAG arm：先检索，再生成
    t0 = time.monotonic()
    contexts, is_error = await dispatch_tool(
        "search_rag", {"query": case.question}, user_id=""
    )
    retrieve_ms = (time.monotonic() - t0) * 1000

    if is_error or not contexts.strip():
        contexts = ""
        retrieve_ok = False
    else:
        retrieve_ok = True

    system_rag = _RAG_SYSTEM_TMPL.format(contexts=contexts) if contexts else _EVAL_BASE
    t1 = time.monotonic()
    rag_response = await _generate(system_rag, case.question)
    rag_gen_ms = (time.monotonic() - t1) * 1000

    # No-RAG arm：直接生成
    t2 = time.monotonic()
    no_rag_response = await _generate(_NO_RAG_SYSTEM, case.question)
    no_rag_gen_ms = (time.monotonic() - t2) * 1000

    return {
        "id": case.id,
        "category": case.category,
        "question": case.question,
        "retrieve_ms": retrieve_ms,
        "retrieve_ok": retrieve_ok,
        "rag_contexts": contexts,
        "rag_response": rag_response,
        "rag_gen_ms": rag_gen_ms,
        "no_rag_response": no_rag_response,
        "no_rag_gen_ms": no_rag_gen_ms,
    }


# ── LLM 评判 ──────────────────────────────────────────────────────────────────

_JUDGE_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_evaluation",
        "description": "提交对两个医疗回复的评估结果",
        "parameters": {
            "type": "object",
            "required": [
                "accuracy", "groundedness", "safety", "completeness",
                "winner", "reasoning",
            ],
            "properties": {
                "accuracy": {
                    "type": "object",
                    "required": ["score_a", "score_b", "reason"],
                    "properties": {
                        "score_a": {
                            "type": "integer",
                            "description": "回复A的准确性评分（1-5）：医学事实、数字、药名是否正确",
                        },
                        "score_b": {
                            "type": "integer",
                            "description": "回复B的准确性评分（1-5）",
                        },
                        "reason": {
                            "type": "string",
                            "description": "一到两句评分理由",
                        },
                    },
                },
                "groundedness": {
                    "type": "object",
                    "required": ["score_a", "score_b", "reason"],
                    "properties": {
                        "score_a": {
                            "type": "integer",
                            "description": "回复A的可信度评分（1-5）：5=有据可查无编造，1=大量不实内容",
                        },
                        "score_b": {"type": "integer"},
                        "reason": {"type": "string"},
                    },
                },
                "safety": {
                    "type": "object",
                    "required": ["score_a", "score_b", "reason"],
                    "properties": {
                        "score_a": {
                            "type": "integer",
                            "description": "回复A的安全性评分（1-5）：5=对风险/禁忌/严重情况有适当警示",
                        },
                        "score_b": {"type": "integer"},
                        "reason": {"type": "string"},
                    },
                },
                "completeness": {
                    "type": "object",
                    "required": ["score_a", "score_b", "reason"],
                    "properties": {
                        "score_a": {
                            "type": "integer",
                            "description": "回复A的完整性评分（1-5）：是否完整回答了用户问题",
                        },
                        "score_b": {"type": "integer"},
                        "reason": {"type": "string"},
                    },
                },
                "winner": {
                    "type": "string",
                    "enum": ["a", "b", "tie"],
                    "description": "综合四个维度，哪个回复更适合患者：a=回复A更好，b=回复B更好，tie=相当",
                },
                "reasoning": {
                    "type": "string",
                    "description": "综合判断的理由（两到三句）",
                },
            },
        },
    },
}

_JUDGE_SYSTEM = """\
你是医疗回复质量评审专家。你的任务是对两个医疗健康助手的回复质量进行严格评分。

评分维度（每项 1-5 分）：
- 准确性（accuracy）   ：医学事实是否准确，数字 / 标准 / 药名是否正确
- 可信度（groundedness）：信息是否有据可查，有无明显编造或过度自信
- 安全性（safety）     ：对潜在风险、禁忌、严重症状是否给出了适当警示和就医建议
- 完整性（completeness）：是否完整回答了用户的问题

评分参考：5=优秀 4=良好 3=一般 2=较差 1=很差

重要：请严格依据内容质量评分，不得因回复的排列位置（A 或 B）产生偏见。
"""


async def judge_pair(row: dict) -> dict:
    """用 LLM 对一对回复进行盲测评分，返回评分结果"""
    # 随机决定展示顺序，实现盲测
    a_is_rag = random.random() < 0.5
    if a_is_rag:
        resp_a, resp_b = row["rag_response"], row["no_rag_response"]
    else:
        resp_a, resp_b = row["no_rag_response"], row["rag_response"]

    user_msg = (
        f"请评价以下两个医疗健康回复的质量。\n\n"
        f"**用户问题：**\n{row['question']}\n\n"
        f"---\n**回复 A：**\n{resp_a}\n\n"
        f"---\n**回复 B：**\n{resp_b}\n\n"
        f"---\n请调用 submit_evaluation 工具提交你的评分。"
    )

    resp = await _get_client().chat.completions.create(
        model=_MODEL,
        max_tokens=1024,
        tools=[_JUDGE_TOOL],
        tool_choice={"type": "function", "function": {"name": "submit_evaluation"}},
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
    )

    # 提取工具调用结果
    eval_input = None
    for tool_call in (resp.choices[0].message.tool_calls or []):
        if tool_call.function.name == "submit_evaluation":
            eval_input = json.loads(tool_call.function.arguments)
            break

    if eval_input is None:
        return {"id": row["id"], "error": "judge returned no tool_use"}

    # 将 a/b 映射回 rag/no_rag
    raw_winner = eval_input.get("winner", "tie")
    if raw_winner == "tie":
        winner = "tie"
    elif (raw_winner == "a") == a_is_rag:
        winner = "rag"
    else:
        winner = "no_rag"

    def extract(dim: str) -> dict:
        d = eval_input.get(dim, {})
        rag_key    = "score_a" if a_is_rag else "score_b"
        no_rag_key = "score_b" if a_is_rag else "score_a"
        return {
            f"{dim}_rag":    d.get(rag_key, 0),
            f"{dim}_no_rag": d.get(no_rag_key, 0),
            f"{dim}_reason": d.get("reason", ""),
        }

    result = {
        "id":          row["id"],
        "category":    row["category"],
        "question":    row["question"],
        "retrieve_ok": row.get("retrieve_ok", False),
        "winner":      winner,
        "reasoning":   eval_input.get("reasoning", ""),
    }
    for dim in ("accuracy", "groundedness", "safety", "completeness"):
        result.update(extract(dim))
    return result


# ── 批量运行 ──────────────────────────────────────────────────────────────────

async def run_generation(cases: list[ABCase], concurrency: int) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)

    async def safe_gen(case: ABCase) -> dict:
        async with sem:
            try:
                return await generate_pair(case)
            except Exception as exc:
                return {
                    "id": case.id, "category": case.category,
                    "question": case.question,
                    "retrieve_ok": False, "rag_contexts": "",
                    "rag_response": "", "no_rag_response": "",
                    "error": str(exc),
                }

    print(f"[info] 生成对比回复，共 {len(cases)} 条，并发={concurrency}…")
    t0 = time.monotonic()
    rows = await asyncio.gather(*[safe_gen(c) for c in cases])
    print(f"[info] 生成完成，总耗时 {time.monotonic()-t0:.1f}s")
    return list(rows)


async def run_judging(rows: list[dict], concurrency: int) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)

    async def safe_judge(row: dict) -> dict:
        async with sem:
            if "error" in row:
                return {**row, "winner": "error"}
            if not row.get("rag_response") or not row.get("no_rag_response"):
                return {**row, "winner": "error", "error": "empty response"}
            try:
                return await judge_pair(row)
            except Exception as exc:
                return {"id": row["id"], "winner": "error", "error": str(exc)}

    print(f"[info] LLM 评判，共 {len(rows)} 条，并发={concurrency}…")
    t0 = time.monotonic()
    results = await asyncio.gather(*[safe_judge(r) for r in rows])
    print(f"[info] 评判完成，总耗时 {time.monotonic()-t0:.1f}s")
    return list(results)


# ── 报告 ──────────────────────────────────────────────────────────────────────

def _avg(vals: list) -> float:
    return sum(vals) / len(vals) if vals else 0.0


_DIM_LABELS = {
    "accuracy":     "准确性",
    "groundedness": "可信度",
    "safety":       "安全性",
    "completeness": "完整性",
}

_WINNER_LABELS = {"rag": "✓ 有RAG", "no_rag": "✓ 无RAG", "tie": "平局"}


def print_report(results: list[dict]) -> None:
    valid  = [r for r in results if r.get("winner") not in ("error", None)]
    errors = [r for r in results if r.get("winner") == "error"]

    rag_wins    = sum(1 for r in valid if r["winner"] == "rag")
    no_rag_wins = sum(1 for r in valid if r["winner"] == "no_rag")
    ties        = sum(1 for r in valid if r["winner"] == "tie")
    n = len(valid)

    print("\n" + "=" * 72)
    print("  A/B 测试报告：加 RAG vs 不加 RAG（LLM as judge）")
    print("=" * 72)
    print(
        f"  有效评测：{n} 条  |  "
        f"RAG 获胜：{rag_wins}（{rag_wins/n*100:.1f}%）  |  "
        f"无 RAG 获胜：{no_rag_wins}（{no_rag_wins/n*100:.1f}%）  |  "
        f"平局：{ties}"
    )
    if errors:
        print(f"  评测出错：{len(errors)} 条")
    print()

    # 各维度均值对比
    print(f"  {'维度':<12}  {'有 RAG':>8}  {'无 RAG':>8}  {'Delta':>8}")
    print("  " + "─" * 48)
    for dim, label in _DIM_LABELS.items():
        r_scores = [r[f"{dim}_rag"]    for r in valid if f"{dim}_rag" in r]
        n_scores = [r[f"{dim}_no_rag"] for r in valid if f"{dim}_no_rag" in r]
        avg_r = _avg(r_scores)
        avg_n = _avg(n_scores)
        print(f"  {label:<12}  {avg_r:>8.2f}  {avg_n:>8.2f}  {avg_r-avg_n:>+8.2f}")
    print()

    # 按类别细分
    categories = sorted({r.get("category", "?") for r in valid})
    if len(categories) > 1:
        print("  按类别细分：")
        for cat in categories:
            cat_rows = [r for r in valid if r.get("category") == cat]
            cat_rag = sum(1 for r in cat_rows if r["winner"] == "rag")
            rate = cat_rag / len(cat_rows) * 100 if cat_rows else 0
            print(f"    {cat}（n={len(cat_rows)}）：RAG 胜率 {rate:.0f}%")
        print()

    # RAG 获胜代表案例
    rag_cases = [r for r in valid if r["winner"] == "rag"][:3]
    if rag_cases:
        print("  RAG 获胜代表案例：")
        for r in rag_cases:
            print(f"    #{r['id']:02d} {r['question'][:42]}")
            print(f"         {r.get('reasoning','')[:80]}")
    print()

    # 无 RAG 获胜代表案例
    no_rag_cases = [r for r in valid if r["winner"] == "no_rag"][:3]
    if no_rag_cases:
        print("  无 RAG 获胜代表案例：")
        for r in no_rag_cases:
            print(f"    #{r['id']:02d} {r['question'][:42]}")
            print(f"         {r.get('reasoning','')[:80]}")
    print()

    print("=" * 72)


def build_md_report(rows: list[dict], results: list[dict]) -> str:
    valid  = [r for r in results if r.get("winner") not in ("error", None)]
    errors = [r for r in results if r.get("winner") == "error"]
    n = len(valid)

    rag_wins    = sum(1 for r in valid if r["winner"] == "rag")
    no_rag_wins = sum(1 for r in valid if r["winner"] == "no_rag")
    ties        = sum(1 for r in valid if r["winner"] == "tie")

    lines: list[str] = []
    lines.append("# A/B 测试报告：加 RAG vs 不加 RAG（LLM as judge）\n")
    lines.append(
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        f"  |  生成 + 评判模型：{_MODEL}\n"
    )

    # 总体结果
    lines.append("## 总体结果\n")
    lines.append("| 指标 | 值 |")
    lines.append("|------|--:|")
    lines.append(f"| 有效评测 | {n} 条 |")
    lines.append(
        f"| RAG 获胜 | {rag_wins} 条（{rag_wins/n*100:.1f}%） |" if n else "| RAG 获胜 | — |"
    )
    lines.append(
        f"| 无 RAG 获胜 | {no_rag_wins} 条（{no_rag_wins/n*100:.1f}%） |" if n else "| 无 RAG 获胜 | — |"
    )
    lines.append(f"| 平局 | {ties} 条 |")
    if errors:
        lines.append(f"| 评测出错 | {len(errors)} 条 |")
    lines.append("")

    # 维度对比
    lines.append("## 各维度得分对比（1-5 分）\n")
    lines.append("| 维度 | 有 RAG | 无 RAG | Delta |")
    lines.append("|------|-------:|-------:|------:|")
    for dim, label in _DIM_LABELS.items():
        r_s = _avg([r[f"{dim}_rag"]    for r in valid if f"{dim}_rag" in r])
        n_s = _avg([r[f"{dim}_no_rag"] for r in valid if f"{dim}_no_rag" in r])
        delta = r_s - n_s
        lines.append(f"| {label} | {r_s:.2f} | {n_s:.2f} | **{delta:+.2f}** |")
    lines.append("")

    # 按类别细分
    categories = sorted({r.get("category", "?") for r in valid})
    if len(categories) > 1:
        lines.append("## 按类别细分\n")
        lines.append("| 类别 | n | RAG 胜率 | avg accuracy_rag | avg accuracy_no_rag |")
        lines.append("|------|--:|---------:|----------------:|-------------------:|")
        for cat in categories:
            cat_rows = [r for r in valid if r.get("category") == cat]
            cat_rag  = sum(1 for r in cat_rows if r["winner"] == "rag")
            rate     = cat_rag / len(cat_rows) * 100 if cat_rows else 0
            acc_r    = _avg([r.get("accuracy_rag", 0) for r in cat_rows])
            acc_n    = _avg([r.get("accuracy_no_rag", 0) for r in cat_rows])
            lines.append(
                f"| {cat} | {len(cat_rows)} | {rate:.0f}% | {acc_r:.2f} | {acc_n:.2f} |"
            )
        lines.append("")

    # 逐条评审明细
    lines.append("## 逐条评审明细\n")
    lines.append("| # | 类别 | 问题 | 获胜方 | 理由摘要 |")
    lines.append("|---|------|------|--------|----------|")
    result_map = {r["id"]: r for r in valid}
    for row in rows:
        r = result_map.get(row["id"])
        if r is None:
            continue
        q        = row["question"][:35] + ("…" if len(row["question"]) > 35 else "")
        wlabel   = _WINNER_LABELS.get(r["winner"], "?")
        reason   = r.get("reasoning", "")
        short_r  = reason[:60] + ("…" if len(reason) > 60 else "")
        lines.append(f"| {row['id']:02d} | {row['category']} | {q} | {wlabel} | {short_r} |")
    lines.append("")

    # 详细对比（完整回复）
    lines.append("## 回复详情\n")
    for row in rows:
        if not row.get("rag_response") or not row.get("no_rag_response"):
            continue
        r = result_map.get(row["id"])
        lines.append(f"### #{row['id']:02d} {row['question']}\n")

        if r:
            wlabel = {"rag": "有 RAG 获胜", "no_rag": "无 RAG 获胜", "tie": "平局"}.get(
                r["winner"], "?"
            )
            lines.append(f"**评审结果**：{wlabel}")
            lines.append(f"\n**综合理由**：{r.get('reasoning', '')}\n")

            lines.append("| 维度 | 有 RAG | 无 RAG |")
            lines.append("|------|-------:|-------:|")
            for dim, label in _DIM_LABELS.items():
                lines.append(
                    f"| {label} | {r.get(f'{dim}_rag', 0)} | {r.get(f'{dim}_no_rag', 0)} |"
                )
            lines.append("")

        retrieve_status = "✓ 检索成功" if row.get("retrieve_ok") else "✗ 检索失败（降级）"
        lines.append(f"**有 RAG 回复** [{retrieve_status}]：\n")
        lines.append(f"> {row['rag_response'].replace(chr(10), chr(10) + '> ')}\n")
        lines.append(f"**无 RAG 回复**：\n")
        lines.append(f"> {row['no_rag_response'].replace(chr(10), chr(10) + '> ')}\n")
        lines.append("---\n")

    if errors:
        lines.append("## 出错案例\n")
        for r in errors:
            lines.append(f"- #{r.get('id','?'):02d} — `{r.get('error','')[:100]}`")
        lines.append("")

    return "\n".join(lines) + "\n"


# ── 主入口 ────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 获取回复对
    if args.load_rows:
        with open(args.load_rows, encoding="utf-8") as f:
            rows = json.load(f)
        print(f"[info] 已加载 {len(rows)} 条回复对 ← {args.load_rows}")
    else:
        cases = CASES[: args.n] if args.n else CASES
        rows  = await run_generation(cases, args.concurrency)

        rows_path = str(results_dir / f"ab_rows_{ts}.json")
        with open(rows_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"[info] 回复对已保存 → {rows_path}")

        if args.generate_only:
            print("[info] --generate-only 模式，跳过评判。")
            return

    # LLM 评判
    results = await run_judging(rows, args.concurrency)

    print_report(results)

    result_path = str(results_dir / f"ab_results_{ts}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[info] 评判结果已保存 → {result_path}")

    md_path = str(results_dir / f"ab_report_{ts}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(build_md_report(rows, results))
    print(f"[info] 报告已保存 → {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG A/B 测试（LLM as judge）")
    parser.add_argument(
        "--n", type=int, default=None,
        help="测试条数（默认全部 25 条）",
    )
    parser.add_argument(
        "--concurrency", type=int, default=3,
        help="并发数（默认 3，受限于 BGE-M3 内存）",
    )
    parser.add_argument(
        "--load-rows", type=str, default=None,
        help="跳过生成步骤，直接加载已有回复对 JSON 路径",
    )
    parser.add_argument(
        "--generate-only", action="store_true",
        help="只生成回复对并保存，不进行评判",
    )
    args = parser.parse_args()
    asyncio.run(main(args))
