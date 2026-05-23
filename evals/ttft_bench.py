"""
evals/ttft_bench.py — SSE 首字时延（TTFT）基准测试

从客户端侧测量：发出请求 → 收到第一个 type=text 事件的时间。
同时从服务端 agent_trace.jsonl 读取 first_token / 各阶段日志做对比。

运行前确保后端已启动：
    uvicorn backend.main:app --reload --port 8000

运行：
    python evals/ttft_bench.py
    python evals/ttft_bench.py --url http://localhost:8000 --runs 10
"""
import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(override=True)
sys.stdout.reconfigure(encoding="utf-8")

# ── 测试用消息（覆盖不调工具 / 调一个工具 / 调多个工具三种场景）────────────────

BENCH_CASES = [
    {"label": "no_tool",       "message": "你好，请问你能帮我做什么"},
    {"label": "memory_only",   "message": "我上次说的头痛有没有好转"},
    {"label": "rag_only",      "message": "布洛芬有哪些常见副作用"},
    {"label": "memory+rag",    "message": "我上次用的布洛芬，长期吃有什么副作用"},
    {"label": "web_only",      "message": "最新研究说阿司匹林预防心脏病还有用吗"},
]


def _get_jwt(base_url: str) -> str:
    """用 DEMO 账号登录拿 token。"""
    resp = httpx.post(
        f"{base_url}/auth/login",
        json={
            "email":    os.getenv("DEMO_EMAIL",    "demo@health.ai"),
            "password": os.getenv("DEMO_PASSWORD", "demo123"),
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["token"]


async def measure_ttft(base_url: str, token: str, message: str) -> dict:
    """
    发一条消息，返回：
      client_ttft_ms  : 客户端从发出请求到收到第一个 text token 的时间
      client_total_ms : 客户端从发出请求到 done 事件的总时间
      first_delta     : 第一个 token 的内容
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
    }
    payload = json.dumps({"message": message, "session_id": f"bench-{time.time_ns()}"})

    t0 = time.monotonic()
    client_ttft_ms   = None
    client_total_ms  = None
    first_delta      = ""

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", f"{base_url}/chat", headers=headers, content=payload) as resp:
            resp.raise_for_status()
            buffer = ""
            async for raw_chunk in resp.aiter_text():
                buffer += raw_chunk
                while "\n\n" in buffer:
                    event_str, buffer = buffer.split("\n\n", 1)
                    for line in event_str.splitlines():
                        if not line.startswith("data: "):
                            continue
                        try:
                            payload_obj = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue

                        if payload_obj.get("type") == "text" and client_ttft_ms is None:
                            client_ttft_ms = int((time.monotonic() - t0) * 1000)
                            first_delta    = payload_obj.get("delta", "")

                        if payload_obj.get("type") == "done":
                            client_total_ms = int((time.monotonic() - t0) * 1000)

    return {
        "client_ttft_ms":  client_ttft_ms,
        "client_total_ms": client_total_ms,
        "first_delta":     first_delta[:30],
    }


def read_server_ttft(log_path: Path, since_ts: float) -> list[dict]:
    """从 agent_trace.jsonl 读取本次 bench 产生的 first_token 和各阶段事件。"""
    if not log_path.exists():
        return []
    results = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        ts = r.get("ts", "")
        if not ts:
            continue
        try:
            record_ts = datetime.fromisoformat(ts).timestamp()
        except Exception:
            continue
        if record_ts >= since_ts:
            results.append(r)
    return results


async def run_bench(base_url: str, runs: int) -> None:
    print(f"\n[bench] 后端地址: {base_url}，每个场景重复 {runs} 次\n")

    log_path = Path(__file__).parent.parent / "logs" / "agent_trace.jsonl"

    token = _get_jwt(base_url)
    print(f"[bench] 登录成功，token 长度={len(token)}\n")

    all_results: dict[str, list[dict]] = {c["label"]: [] for c in BENCH_CASES}

    for case in BENCH_CASES:
        label   = case["label"]
        message = case["message"]
        print(f"▶ {label}: {message!r}")

        for i in range(runs):
            t_before = time.time()
            try:
                result = await measure_ttft(base_url, token, message)
            except Exception as exc:
                print(f"  run {i+1} 出错: {exc}")
                continue

            # 从日志读取服务端 first_token
            server_logs = read_server_ttft(log_path, t_before)
            server_ttft = next(
                (r["ms"] for r in server_logs if r.get("event") == "first_token"),
                None,
            )
            triage_ms = None
            agent_ms  = None
            for r in server_logs:
                if r.get("event") == "llm_call" and r.get("node") == "triage" and r.get("ms", 0) > 0:
                    triage_ms = r["ms"]
                if r.get("event") == "llm_call" and r.get("node") == "orchestrator" and r.get("ms", 0) > 0:
                    agent_ms = r["ms"]

            result["server_ttft_ms"] = server_ttft
            result["triage_ms"]      = triage_ms
            result["agent_ms"]       = agent_ms
            all_results[label].append(result)

            print(
                f"  run {i+1:2d}  client_ttft={result['client_ttft_ms']}ms"
                f"  total={result['client_total_ms']}ms"
                + (f"  server_ttft={server_ttft}ms" if server_ttft else "")
                + f"  first={result['first_delta']!r}"
            )

        await asyncio.sleep(1)  # 轮次间隔，避免 rate limit

    # ── 汇总报告 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TTFT 汇总报告")
    print("=" * 70)
    print(f"{'场景':<16} {'样本':>4} {'TTFT中位':>10} {'TTFT均值':>10} {'TTFT p95':>10} {'total中位':>10}")
    print("-" * 70)

    for case in BENCH_CASES:
        label = case["label"]
        vals  = [r["client_ttft_ms"] for r in all_results[label] if r.get("client_ttft_ms")]
        total = [r["client_total_ms"] for r in all_results[label] if r.get("client_total_ms")]
        if not vals:
            print(f"  {label:<14} {'—':>4}")
            continue
        vals.sort()
        p95 = vals[int(len(vals) * 0.95)]
        print(
            f"  {label:<14} {len(vals):>4}"
            f" {statistics.median(vals):>9.0f}ms"
            f" {statistics.mean(vals):>9.0f}ms"
            f" {p95:>9.0f}ms"
            f" {statistics.median(total) if total else 0:>9.0f}ms"
        )

    # ── 保存 JSON ─────────────────────────────────────────────────────────────
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"ttft_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n[bench] 结果已保存至 {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",  default="http://localhost:8000", help="后端地址")
    parser.add_argument("--runs", type=int, default=5,             help="每场景重复次数")
    args = parser.parse_args()
    asyncio.run(run_bench(args.url, args.runs))
