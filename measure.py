"""
四项数据测量脚本。

1. Triage 准确率 + 平均延迟
2. Prompt cache 命中率（有 / 无 cache 对比 token 数）
3. fire-and-forget 延迟收益（同步 vs 后台写入）
4. 从已有 JSONL 日志提取真实 LLM 耗时
"""
import asyncio
import sys
import time
sys.stdout.reconfigure(encoding="utf-8")
from dotenv import load_dotenv
load_dotenv()

# ── 1. Triage 准确率 + 延迟 ───────────────────────────────────────────────────

EMERGENCY_CASES = [
    "我突然胸口剧烈疼痛，喘不上气",
    "我心脏疼，左臂也发麻",
    "我感觉心跳骤停了，很难受",
    "我呼吸困难，感觉快窒息了",
    "我朋友突然晕倒了，叫不醒",
    "我严重出血，血止不住",
    "我误吃了大量药物，感觉中毒",
    "我全身严重过敏，喉咙在肿",
    "我发烧41度，烧了两天了",
    "我突然说话不清楚，半边脸麻木，可能脑卒中",
    "我胸痛同时冒冷汗，心梗症状",
    "我呼吸越来越困难，嘴唇发紫",
    "我的孩子高烧40.5度，抽搐了",
    "我意识模糊，感觉要失去意识",
    "我严重烧伤，皮肤都起泡了",
]

NON_EMERGENCY_CASES = [
    "我最近睡眠不太好",
    "头痛了两天了",
    "我消化不太好，有点胀气",
    "我最近压力很大，情绪不好",
    "我喉咙有点痒，轻微咳嗽",
    "我肩膀有点酸痛",
    "我最近有点疲劳",
    "我眼睛干涩",
    "我轻微腹泻，今天两次",
    "我鼻子有点堵，轻微感冒",
    "我膝盖有点不舒服，走路时会疼",
    "我最近食欲不太好",
    "我有点头晕，站起来的时候",
    "我嘴里有点溃疡",
    "我手指关节有点僵硬",
]

async def measure_triage():
    from agent.triage import triage
    print("\n" + "="*60)
    print("1. TRIAGE 准确率 + 延迟")
    print("="*60)

    correct = 0
    total = 0
    latencies = []
    errors = []

    for msg in EMERGENCY_CASES:
        t0 = time.monotonic()
        is_em, _ = await triage(msg)
        ms = (time.monotonic() - t0) * 1000
        latencies.append(ms)
        total += 1
        if is_em:
            correct += 1
        else:
            errors.append(f"  漏报(应紧急): {msg}")

    for msg in NON_EMERGENCY_CASES:
        t0 = time.monotonic()
        is_em, _ = await triage(msg)
        ms = (time.monotonic() - t0) * 1000
        latencies.append(ms)
        total += 1
        if not is_em:
            correct += 1
        else:
            errors.append(f"  误报(应放行): {msg}")

    accuracy = correct / total * 100
    avg_ms = sum(latencies) / len(latencies)
    p50 = sorted(latencies)[len(latencies)//2]
    p95 = sorted(latencies)[int(len(latencies)*0.95)]

    print(f"准确率:  {correct}/{total} = {accuracy:.1f}%")
    print(f"延迟 avg={avg_ms:.0f}ms  p50={p50:.0f}ms  p95={p95:.0f}ms")
    if errors:
        print(f"错误案例 ({len(errors)} 条):")
        for e in errors:
            print(e)
    else:
        print("无错误案例")

    return accuracy, avg_ms


# ── 2. Prompt Cache 命中对比 ──────────────────────────────────────────────────

async def measure_cache():
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage, SystemMessage
    from agent.prompts import SYSTEM_PROMPT
    from datetime import date

    print("\n" + "="*60)
    print("2. PROMPT CACHE 命中率")
    print("="*60)

    llm = ChatAnthropic(model="claude-sonnet-4-6")
    today = date.today().strftime("%Y年%m月%d日")
    system = SystemMessage(content=[{
        "type": "text",
        "text": f"今天是 {today}。\n\n" + SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }])
    user_msg = HumanMessage(content="我最近头有点痛")

    results = []
    for i in range(3):
        label = "首次调用（写入 cache）" if i == 0 else f"第{i+1}次调用（应命中 cache）"
        t0 = time.monotonic()
        resp = await llm.ainvoke([system, user_msg])
        ms = (time.monotonic() - t0) * 1000
        meta = resp.usage_metadata or {}
        cache_read   = meta.get("cache_read_input_tokens", 0)
        cache_create = meta.get("cache_creation_input_tokens", 0)
        input_tokens = meta.get("input_tokens", 0)
        results.append((label, input_tokens, cache_read, cache_create, ms))
        print(f"  [{label}]")
        print(f"    input_tokens={input_tokens}  cache_read={cache_read}  cache_creation={cache_create}  耗时={ms:.0f}ms")

    # 计算节省比例
    first_input = results[0][1] + results[0][2] + results[0][3]
    if len(results) >= 2 and results[1][2] > 0:
        saved_pct = results[1][2] / first_input * 100
        print(f"\n  Cache 命中后节省输入 token: {saved_pct:.1f}%")
        ttft_improvement = (results[0][4] - results[1][4]) / results[0][4] * 100
        print(f"  TTFT 改善: {ttft_improvement:.1f}%（{results[0][4]:.0f}ms → {results[1][4]:.0f}ms）")
    else:
        print("\n  未检测到 cache_read_input_tokens，cache 可能未命中")

    return results


# ── 3. 从 JSONL 提取历史 LLM 耗时 ────────────────────────────────────────────

def measure_from_logs():
    import json
    from pathlib import Path

    print("\n" + "="*60)
    print("4. JSONL 日志中的真实 LLM 数据")
    print("="*60)

    log_file = Path("logs/agent_trace.jsonl")
    if not log_file.exists():
        print("  日志文件不存在，跳过")
        return

    lines = [json.loads(l) for l in log_file.read_text(encoding="utf-8").splitlines() if l.strip()]

    llm_calls = [e for e in lines if e["event"] == "llm_call" and e.get("ms", 0) > 0]
    tool_uses  = [e for e in lines if e["event"] == "tool_use"]
    pipelines  = [e for e in lines if e["event"] == "pipeline_end"]
    recurrences = [e for e in lines if e["event"] == "recurrence_detected"]

    print(f"  总 pipeline 次数:    {len(pipelines)}")
    print(f"  反复发作检测触发:    {len(recurrences)} 次")

    if llm_calls:
        avg_ms = sum(e["ms"] for e in llm_calls) / len(llm_calls)
        avg_in  = sum(e.get("tokens_input", 0)  for e in llm_calls) / len(llm_calls)
        avg_out = sum(e.get("tokens_output", 0) for e in llm_calls) / len(llm_calls)
        print(f"  LLM 调用次数:        {len(llm_calls)}")
        print(f"  平均耗时:            {avg_ms:.0f}ms")
        print(f"  平均 token 消耗:     input={avg_in:.0f}  output={avg_out:.0f}")
    else:
        print("  暂无真实 LLM 调用日志（ms>0）")

    if tool_uses:
        by_tool = {}
        for e in tool_uses:
            t = e.get("tool", "?")
            by_tool.setdefault(t, []).append(e.get("ms", 0))
        print("  工具调用统计:")
        for tool, mss in sorted(by_tool.items()):
            real = [m for m in mss if m > 0]
            if real:
                print(f"    {tool}: {len(mss)} 次，平均 {sum(real)/len(real):.0f}ms")
            else:
                print(f"    {tool}: {len(mss)} 次")


# ── 主入口 ────────────────────────────────────────────────────────────────────

async def main():
    await measure_triage()
    await measure_cache()
    measure_from_logs()
    print("\n" + "="*60)
    print("测量完成")
    print("="*60)

asyncio.run(main())
