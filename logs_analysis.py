import json
from datetime import datetime
from collections import defaultdict

traces = defaultdict(list)
with open("logs/agent_trace.jsonl") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        ev = json.loads(line)
        traces[ev["trace_id"]].append(ev)

results = []
for tid, events in traces.items():
    by_event = {e["event"]: e for e in events}
    start = by_event.get("pipeline_start")
    end   = by_event.get("pipeline_end")
    if not start or not end:
        continue

    t_start = datetime.fromisoformat(start["ts"])
    t_end   = datetime.fromisoformat(end["ts"])
    total   = (t_end - t_start).total_seconds()

    tools_called = [e.get("tools_requested", []) for e in events if e.get("event") == "llm_call"]
    all_tools = [t for sub in tools_called for t in sub]
    mem_count = all_tools.count("memory_search")
    has_cooldown = any(e.get("event") == "mem_search_cooldown" for e in events)
    cooldown_reset = any(e.get("event") == "mem_search_cooldown_reset" for e in events)
    session = start.get("session_id", "")[-6:]

    results.append({
        "tid": tid[:8], "ts": t_start.strftime("%H:%M:%S"), "session": session,
        "total_s": round(total, 1), "mem_count": mem_count, "all_tools": all_tools,
        "cooldown": has_cooldown, "reset": cooldown_reset,
    })

results.sort(key=lambda x: x["ts"])

# 过滤掉测试条目（耗时 < 2s 的基本是空跑）
real = [r for r in results if r["total_s"] >= 2]

print("trace      time       session  total   search  cooldown  tools")
print("-" * 80)
for r in real[-15:]:
    cd = "HIT" if r["cooldown"] else ("RST" if r["reset"] else "-")
    tools_str = ",".join(r["all_tools"]) if r["all_tools"] else "(none)"
    print(f"{r['tid']:10} {r['ts']:10} {r['session']:8} {r['total_s']:>5}s {r['mem_count']:>6}x  {cd:>6}    {tools_str}")

print()
with_s = [r for r in real if r["mem_count"] > 0]
no_s   = [r for r in real if r["mem_count"] == 0]
avg_with = sum(r["total_s"] for r in with_s) / len(with_s) if with_s else 0
avg_no   = sum(r["total_s"] for r in no_s)   / len(no_s)   if no_s   else 0

print(f"with memory_search  ({len(with_s)} turns): avg {avg_with:.1f}s")
print(f"without memory_search ({len(no_s)} turns): avg {avg_no:.1f}s")
if with_s and no_s:
    print(f"potential saving: {avg_with - avg_no:.1f}s/turn")
print()
print("note: cooldown just deployed - HIT=skipped, RST=reset, -=pre-cooldown history")
