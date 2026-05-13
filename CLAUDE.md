# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Backend
```bash
# 启动后端（项目根目录）
uvicorn backend.main:app --reload --port 8000

# 运行测试
pytest
pytest tests/test_foo.py::test_bar   # 单个测试
```

### Frontend
```bash
cd frontend
npm install
npm run dev    # http://localhost:3000
npm run build
```

### 环境配置
复制 `.env.example` 为 `.env`，填入以下必填项：
- `ANTHROPIC_API_KEY` — 主模型调用
- `MEM0_API_KEY` — 长期记忆存储
- `REDIS_URL` — 会话缓冲（本地用 `redis://localhost:6379`）

`BRAVE_API_KEY` 缺失时 `web_search` 工具静默降级，不影响启动。`QDRANT_HOST/PORT` 仅 `search_rag` 工具需要。

---

## 架构

### 请求流水线（每条用户消息）

```
HTTP POST /chat
  → AuthMiddleware（JWT 验证，注入 request.state.user_id）
  → chat.py（读取 Redis session 历史）
      ├─ /consult 模式 → IntakeGraph（Haiku 多轮问诊，收集到 [INTAKE_DONE] 后转主流程）
      └─ 普通模式 → AgentHarness
           ├─ L1 Triage（Haiku，<500ms，紧急症状短路返回）
           ├─ L2 LangGraph orchestrator（Sonnet，tool loop 最多 5 轮）
           └─ L3 Critic（Haiku，合规审查，不通过则追加补充说明）
  → on_stop_hooks（fire-and-forget）
       ├─ session_hook：会话历史写回 Redis
       └─ memory_write_hook：对话写入 Mem0，刷新 prefetch 缓存
```

### 工具层（`agent/tools.py`）

四个工具通过 `dispatch_tool` 路由：

| 工具 | 实现 |
|------|------|
| `search_memory` | Mem0 云 API，prefetch 缓存优先 |
| `search_rag` | Qdrant gRPC，BGE-M3 dense+sparse 双路 Prefetch + RRF 融合 |
| `web_search` | Brave Search API |
| `generate_report` | 子 Agent（Haiku），读取 `agent/skills/report_gen/` |

`orchestrator_node` 用 `asyncio.gather` 并行执行同一轮次的多个工具调用。

### 记忆分层

| 层 | 实现 | TTL/范围 |
|----|------|----------|
| 短期会话 | Redis `session:{session_id}` | 1小时，最近10条 |
| 问诊状态 | Redis `consult:{session_id}` | 同上 |
| Mem0 预热缓存 | Redis `mem_prefetch:{user_id}` | 10分钟，跨 session |
| 长期记忆 | Mem0 云（向量检索） | 永久 |

Redis 不可用时所有 `SessionBuffer` 操作自动降级为内存字典，不中断服务。

### Critic 反馈回路

Critic 拦截的失败案例追加到 `logs/critic_failures.jsonl`。下次请求构建 system prompt 时，`orchestrator_node` 读取最近 3 条注入为反例，形成自我修正回路。

### 后台定时任务

`APScheduler` 每 6 小时执行 `alert_monitor.run_alert_monitor`：抓取健康预警 → Haiku 解析 → 匹配用户 Mem0 档案 → Redis Pub/Sub 推送。去重 key 为 `alert_sent:{user_id}:{md5[:12]}`，TTL 7 天。

### 模型分工

| 模型 | 用途 |
|------|------|
| `claude-sonnet-4-6` | 主 orchestrator（工具调用 + 最终回复） |
| `claude-haiku-4-5-20251001` | Triage、Critic、IntakeGraph、report subagent、query rewrite |
