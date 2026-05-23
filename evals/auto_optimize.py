"""
evals/auto_optimize.py — 自动化评测 → 诊断 → Prompt 优化闭环

Phase 1 EVAL   : 抽样运行 tool_selection 评测
Phase 2 ANALYZE: LLM 分析失败模式，定位 prompts.py 根因
Phase 3 PATCH  : 生成并展示 diff
Phase 4 APPLY  : 写回 prompts.py（--dry-run 跳过）
Phase 5 VERIFY : 重跑失败用例，对比修复前后

运行：
  python evals/auto_optimize.py
  python evals/auto_optimize.py --dry-run    # 只输出 patch，不写文件
  python evals/auto_optimize.py --no-verify  # 跳过验证步骤
"""

import asyncio
import json
import os
import re
import sys
import time
import argparse
from datetime import datetime, date
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(override=True)

PROMPTS_PATH = Path(__file__).parent.parent / "agent" / "prompts.py"

# 抽样用例 ID：覆盖所有工具 + 已知边界区域
SAMPLE_IDS = {
    # 时效性词（over-call 高发区：web_search + search_rag 同时触发）
    106, 107, 108, 113, 115, 197, 198, 199, 200,
    # web only
    109, 110, 116, 117, 123,
    # rag only
    51, 52, 83, 84, 96, 100,
    # memory only
    1, 2, 16, 17, 31,
    # memory + rag 双工具
    181, 182, 185,
    # no tool
    146, 162, 171, 179,
}


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _extract_system_prompt(prompts_path: Path) -> str:
    """从 prompts.py 文件中提取 SYSTEM_PROMPT 字符串值。"""
    content = prompts_path.read_text(encoding="utf-8")
    m = re.search(r'SYSTEM_PROMPT\s*=\s*"""(.*?)"""', content, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"SYSTEM_PROMPT\s*=\s*'''(.*?)'''", content, re.DOTALL)
    if m:
        return m.group(1)
    raise ValueError("在 prompts.py 中找不到 SYSTEM_PROMPT")


def _section(title: str) -> None:
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")


# ── Phase 1: EVAL ─────────────────────────────────────────────────────────────

async def run_eval(sample_ids: set[int], system_prompt: str) -> tuple[list[dict], list[dict]]:
    """用指定 prompt 文本跑评测，返回 (all_results, failures)。"""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from agent.tools import TOOL_SPECS
    from evals.tool_selection import CASES

    llm = ChatOpenAI(model="gpt-4o", streaming=False)
    llm_with_tools = llm.bind_tools(TOOL_SPECS)

    cases = [c for c in CASES if c.id in sample_ids]
    today = date.today().strftime("%Y年%m月%d日")
    system = SystemMessage(content=f"今天是 {today}。\n\n{system_prompt}")

    sem = asyncio.Semaphore(5)

    async def eval_one(case):
        async with sem:
            try:
                history_msgs = []
                for m in case.history:
                    if m["role"] == "user":
                        history_msgs.append(HumanMessage(content=m["content"]))
                    else:
                        history_msgs.append(AIMessage(content=m["content"]))

                t0 = time.monotonic()
                resp = await llm_with_tools.ainvoke([system] + history_msgs + [HumanMessage(content=case.message)])
                ms = (time.monotonic() - t0) * 1000

                actual_tools = [tc["name"] for tc in (resp.tool_calls or [])]
                expected_set = set(case.expected_tools)
                actual_set   = set(actual_tools)

                return {
                    "id":          case.id,
                    "message":     case.message,
                    "expected":    sorted(case.expected_tools),
                    "actual":      actual_tools,
                    "exact_match": expected_set == actual_set,
                    "over_call":   sorted(actual_set - expected_set),
                    "under_call":  sorted(expected_set - actual_set),
                    "wrong_tool":  bool(actual_set and expected_set and not (actual_set & expected_set)),
                    "ms":          ms,
                }
            except Exception as e:
                return {
                    "id": case.id, "message": case.message,
                    "expected": sorted(case.expected_tools), "actual": [],
                    "exact_match": False, "over_call": [], "wrong_tool": False,
                    "under_call": sorted(case.expected_tools),
                    "ms": 0, "error": str(e),
                }

    t0 = time.monotonic()
    results = list(await asyncio.gather(*[eval_one(c) for c in cases]))
    elapsed = time.monotonic() - t0

    exact    = sum(1 for r in results if r["exact_match"])
    failures = [r for r in results if not r["exact_match"]]

    print(f"用例数: {len(results)}   耗时: {elapsed:.1f}s   正确: {exact}/{len(results)}   失败: {len(failures)}")

    if failures:
        print("\n  失败详情：")
        for f in failures:
            tags = []
            if f.get("over_call"):  tags.append(f"误调 {f['over_call']}")
            if f.get("under_call"): tags.append(f"漏调 {f['under_call']}")
            if "error" in f:        tags.append(f"异常: {f['error'][:30]}")
            msg = f["message"][:42] + ("…" if len(f["message"]) > 42 else "")
            print(f"    #{f['id']:03d}  [{', '.join(tags)}]  {msg}")

    return results, failures


# ── Phase 2: ANALYZE ──────────────────────────────────────────────────────────

def analyze_failures(failures: list[dict], system_prompt: str) -> dict:
    """调用 Claude 分析失败模式，返回带 patch 的诊断结果。"""
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    failures_json = json.dumps([{
        "id":         f["id"],
        "message":    f["message"],
        "expected":   f["expected"],
        "actual":     f["actual"],
        "over_call":  f.get("over_call", []),
        "under_call": f.get("under_call", []),
    } for f in failures], ensure_ascii=False, indent=2)

    user_msg = f"""你是 LLM Agent 的 Prompt 优化专家。

下面是 agent 的 SYSTEM_PROMPT（工具规则部分）：
<prompt>
{system_prompt}
</prompt>

以下是工具选择评测中的失败用例：
<failures>
{failures_json}
</failures>

请分析失败模式并给出 prompt 修复方案，以 JSON 格式输出：
{{
  "has_issue": true,
  "pattern": "失败模式简述（1句话）",
  "root_cause": "prompt 中哪条规则导致了此问题（引用原文，控制在100字内）",
  "fix_strategy": "修复思路（1-2句话）",
  "patch": {{
    "old_text": "需要替换的 prompt 原文（精确匹配，包含换行）",
    "new_text": "替换后的文本"
  }}
}}

注意：old_text 必须能在上面的 prompt 原文中精确匹配到。只输出 JSON，不要任何解释。"""

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = resp.content[0].text.strip()

    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        print(f"  [warn] LLM 未返回有效 JSON:\n{raw[:200]}")
        return {"has_issue": False}

    result = json.loads(m.group())
    print(f"  失败模式 : {result.get('pattern', 'N/A')}")
    print(f"  根因     : {result.get('root_cause', 'N/A')[:100]}")
    print(f"  修复思路 : {result.get('fix_strategy', 'N/A')}")
    return result


# ── Phase 3+4: PATCH & APPLY ──────────────────────────────────────────────────

def apply_patch(patch: dict, prompts_content: str, dry_run: bool) -> tuple[bool, str]:
    """展示 diff 并写回文件，返回 (success, new_content)。"""
    old_text = patch.get("old_text", "").strip()
    new_text = patch.get("new_text", "").strip()

    if not old_text or not new_text or old_text == new_text:
        print("  [skip] 无有效 patch 或内容无变化")
        return False, prompts_content

    # prompts.py 里的 SYSTEM_PROMPT 内容
    system_prompt = _extract_system_prompt(PROMPTS_PATH)

    if old_text not in system_prompt:
        print(f"  [warn] old_text 未在 SYSTEM_PROMPT 中找到，跳过写入")
        print(f"         old_text 前50字: {old_text[:50]!r}")
        return False, prompts_content

    print("  diff：")
    for line in old_text.splitlines():
        print(f"  \033[31m- {line}\033[0m")
    for line in new_text.splitlines():
        print(f"  \033[32m+ {line}\033[0m")

    new_system_prompt = system_prompt.replace(old_text, new_text, 1)
    new_file_content  = prompts_content.replace(system_prompt, new_system_prompt, 1)

    if not dry_run:
        PROMPTS_PATH.write_text(new_file_content, encoding="utf-8")
        print(f"\n  [ok] 已写入 {PROMPTS_PATH.name}")
    else:
        print(f"\n  [dry-run] 未写入文件")

    return True, new_file_content


# ── Phase 5: VERIFY ───────────────────────────────────────────────────────────

async def verify(failures: list[dict], new_system_prompt: str) -> dict:
    """用新 prompt 重跑失败用例，返回修复统计。"""
    ids = {f["id"] for f in failures}
    _, new_failures = await run_eval(ids, new_system_prompt)
    fixed = len(failures) - len(new_failures)
    return {"fixed": fixed, "total": len(failures), "remaining": new_failures}


# ── 主流程 ────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    _section(f"自动化评测 → 优化闭环  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Phase 1 ──
    _section("Phase 1: EVAL")
    original_prompt  = _extract_system_prompt(PROMPTS_PATH)
    prompts_content  = PROMPTS_PATH.read_text(encoding="utf-8")
    results, failures = await run_eval(SAMPLE_IDS, original_prompt)

    if not failures:
        print("\n  所有用例通过，无需优化。")
        return

    # ── Phase 2 ──
    _section("Phase 2: ANALYZE")
    analysis = analyze_failures(failures, original_prompt)

    if not analysis.get("has_issue") or "patch" not in analysis:
        print("\n  LLM 未能生成有效诊断，流程终止。")
        return

    # ── Phase 3+4 ──
    _section(f"Phase 3+4: PATCH {'& APPLY' if not args.dry_run else '(dry-run)'}")
    patched, new_file_content = apply_patch(
        analysis["patch"], prompts_content, dry_run=args.dry_run
    )

    if not patched or args.dry_run or args.no_verify:
        return

    # ── Phase 5 ──
    _section("Phase 5: VERIFY")
    new_prompt = _extract_system_prompt(PROMPTS_PATH)
    verify_result = await verify(failures, new_prompt)

    # ── 汇总 ──
    _section("汇总")
    total = len(SAMPLE_IDS)
    before_pass = total - len(failures)
    after_pass  = total - (len(failures) - verify_result["fixed"])
    print(f"  抽样用例  : {total} 条")
    print(f"  修复前    : {before_pass}/{total} 通过  ({len(failures)} 失败)")
    print(f"  修复后    : {after_pass}/{total} 通过  ({len(failures) - verify_result['fixed']} 失败)")
    print(f"  本次修复  : {verify_result['fixed']}/{verify_result['total']} 条")

    if verify_result["remaining"]:
        print("\n  仍有失败（可再次运行继续优化）：")
        for r in verify_result["remaining"]:
            print(f"    #{r['id']:03d} {r['message'][:45]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自动化评测 → Prompt 优化闭环")
    parser.add_argument("--dry-run",   action="store_true", help="只展示 patch，不写入文件")
    parser.add_argument("--no-verify", action="store_true", help="跳过 Phase 5 验证步骤")
    asyncio.run(main(parser.parse_args()))
