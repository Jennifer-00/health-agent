"""
evals/gen_eval_cases.py — 从 Qdrant 采样 chunk，用 DeepSeek 生成评测问题集。

每个 chunk 生成 1 条问题 + 3-5 个相关词，输出格式与 rag_retrieval.CASES 兼容。
存入 evals/results/generated_cases.json，供 rag_sparse_ablation.py 加载。

Usage:
    python evals/gen_eval_cases.py
    python evals/gen_eval_cases.py --n 200
"""
import json
import random
import asyncio
import argparse
from pathlib import Path

from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from dotenv import load_dotenv
load_dotenv(override=True)

import os

COLLECTION = "medical_knowledge"
OUT_PATH   = Path(__file__).parent / "results" / "generated_cases.json"

# 各书采样比例权重（药物、内科、急诊覆盖消融测试的核心场景）
BOOK_WEIGHTS = {
    "临床药物治疗学":      0.18,
    "急诊内科学":          0.18,
    "内科疾病鉴别诊断学":  0.15,
    "内科治疗指南":        0.12,
    "药理学":              0.10,
    "ICU主治医师手册":     0.10,
    "精神病学":            0.07,
    "临床营养学":          0.05,
    "病理学":              0.05,
}

SKIP_PREFIXES = ("图", "表", "注", "参考", "附录")

SYSTEM_PROMPT = """\
你是一位医学教育专家。根据给定医学文本，生成评测数据。

输出严格遵守 JSON 格式，不要输出任何其他内容：
{
  "question": "模拟真实患者或医学生的口语化问题（能从文本中找到答案）",
  "terms": ["关键词1", "关键词2", "关键词3"]
}

要求：
- question：口语自然，不出现"根据文中"等字样，1句话
- terms：3-5个该段落的核心医学术语，用于相关性判断
"""


def fetch_all_points(qdrant: QdrantClient) -> list:
    all_points, next_offset = [], None
    while True:
        points, next_offset = qdrant.scroll(
            collection_name=COLLECTION, limit=1000,
            offset=next_offset, with_payload=True, with_vectors=False,
        )
        all_points.extend(points)
        if next_offset is None:
            break
    return all_points


def is_valid(point) -> bool:
    p = point.payload
    content = p.get("content", "")
    char_len = p.get("char_len", 0)
    if char_len < 150 or char_len > 700:
        return False
    if any(content.startswith(pf) for pf in SKIP_PREFIXES):
        return False
    return True


def sample_chunks(all_points: list, n: int) -> list:
    by_book: dict[str, list] = {}
    for pt in all_points:
        book = pt.payload.get("book", "")
        if book in BOOK_WEIGHTS:
            by_book.setdefault(book, []).append(pt)

    selected = []
    for book, weight in BOOK_WEIGHTS.items():
        pool = [pt for pt in by_book.get(book, []) if is_valid(pt)]
        quota = max(1, round(n * weight))
        chosen = random.sample(pool, min(quota, len(pool)))
        selected.extend(chosen)
        print(f"  {book:<22s}  quota={quota:>3d}  taken={len(chosen):>3d}  pool={len(pool):>4d}")

    # 补足或截断到目标 n
    random.shuffle(selected)
    return selected[:n]


async def generate_one(client: AsyncOpenAI, content: str, semaphore: asyncio.Semaphore) -> dict | None:
    async with semaphore:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model="deepseek-chat",
                    max_tokens=200,
                    temperature=0.7,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": f"医学文本：\n{content}"},
                    ],
                    response_format={"type": "json_object"},
                )
                data = json.loads(resp.choices[0].message.content)
                if "question" in data and "terms" in data:
                    return data
            except Exception as e:
                if attempt == 2:
                    print(f"  [ERROR] 生成失败: {e}")
        return None


async def run(n: int) -> None:
    random.seed(42)
    qdrant = QdrantClient(host="localhost", port=6333)
    client = AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    sem    = asyncio.Semaphore(8)   # DeepSeek 并发限制

    print(f"从 Qdrant 拉取全部 points...")
    all_points = fetch_all_points(qdrant)
    print(f"共 {len(all_points)} 个 point\n按权重采样 {n} 条：")

    selected = sample_chunks(all_points, n)
    print(f"\n实际采样 {len(selected)} 条，开始生成...\n")

    tasks = [generate_one(client, pt.payload["content"], sem) for pt in selected]
    outputs = await asyncio.gather(*tasks)

    results = []
    done = 0
    for point, output in zip(selected, outputs):
        done += 1
        if output is None:
            continue
        p = point.payload
        results.append({
            "id":                  len(results) + 41,   # 接续手写 40 条的编号
            "query":               output["question"],
            "relevant_terms":      output["terms"],
            "ground_truth_id":     str(point.id),
            "book":                p.get("book", ""),
            "h1":                  p.get("h1", ""),
            "source":              "generated",
            "note":                f"{p.get('book','')} / {p.get('h1','')}",
        })
        if done % 50 == 0:
            print(f"  进度 {done}/{len(selected)}，已生成 {len(results)} 条")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] 生成完成，共 {len(results)} 条 → {OUT_PATH}")

    book_dist: dict[str, int] = {}
    for r in results:
        book_dist[r["book"]] = book_dist.get(r["book"], 0) + 1
    for book, cnt in sorted(book_dist.items(), key=lambda x: -x[1]):
        print(f"  {book:<22s}  {cnt} 条")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="生成条数（默认200）")
    args = parser.parse_args()
    asyncio.run(run(args.n))
