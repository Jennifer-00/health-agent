# 健康 Agent 管理端 — 开发规格文档

> 状态：待开发 | 负责人：未分配 | 最后更新：2026-05-23

---

## 1. 背景

当前 Agent 系统已完成三层可观测性基建：

| 层 | 实现 | 数据位置 |
|----|------|---------|
| Traces (Span Tree) | OpenTelemetry SDK → `agent/telemetry.py` | 内存/OTLP 导出 + `logs/agent_trace.jsonl` |
| Metrics (聚合指标) | Prometheus → `agent/metrics.py` | `/metrics` 端点 |
| Logs (结构化日志) | structlog → `agent/logging_config.py` | `logs/agent.log`（JSON，按天轮换） |

但缺少可视化界面来消费这些数据。需要建一个管理端来展示监测指标、搜索日志、查看 Trace。

---

## 2. 目标

在现有 Next.js 前端（`frontend/`）中新增 `/admin` 路由组，提供运维和运营所需的全套监控界面。

**前端技术栈**：Next.js 14 + TypeScript + Tailwind CSS + Recharts（或 @ant-design/charts）

---

## 3. 后端 API 设计

所有管理端 API 放在 `backend/routers/admin.py`，挂载到 `/api/admin`。

### 3.1 实时概览

```yaml
GET /api/admin/overview
  返回:
    qps: float                    # 最近5分钟平均请求量/分钟
    p50_ms: int                   # Pipeline P50 耗时
    p99_ms: int                   # Pipeline P99 耗时
    p90_ttft_ms: int              # 首 Token P90
    error_rate: float             # 5xx 占比
    active_sessions: int          # 最近30分钟活跃 session 数
    triage_emergency_count: int   # 今天紧急判定数
    critic_intercept_rate: float  # 今天 Critic 拦截率
    dependencies:
      redis: healthy|unhealthy
      qdrant: healthy|unhealthy
      mem0: healthy|unhealthy
    tool_calls_today: int         # 今日工具调用总量
    tokens_today: {input: int, output: int}  # 今日 Token
```

**实现方式**：从 Prometheus `/metrics` 解析当前值 + `/health` 依赖状态。

### 3.2 Trace 查看器

```yaml
GET /api/admin/traces
  参数:
    user_id: str (可选)
    session_id: str (可选)
    start: ISO datetime (可选)
    end: ISO datetime (可选)
    min_duration_ms: int (可选, 过滤慢请求)
    limit: int (默认 50, 最大 200)
    offset: int (默认 0)
  返回:
    total: int
    items:
      - trace_id: str
        ts: ISO datetime
        user_id: str
        session_id: str
        intent: str
        exit: normal|triage_short_circuit|error
        duration_ms: int
        tool_rounds: int
        tokens_input: int
        tokens_output: int
        tools_used: [str]
        has_error: bool
        triage_emergency: bool
        critic_passed: bool|null

GET /api/admin/traces/{trace_id}
  返回:
    trace_id: str
    ts: ISO datetime
    user_id: str
    session_id: str
    spans:  # Span Tree
      - span_id: str
        parent_id: str|null
        name: str           # "pipeline" / "triage" / "orchestrator.round.0.llm" / "tool.search_memory"
        start_ms: int       # 相对 root 开始时间的偏移
        duration_ms: int
        attributes: dict
        status: ok|error
        children: [...]     # 递归嵌套（前端渲染瀑布图用）
    messages:  # 用户消息 + 助手回复（如果有）
      - role: user|assistant|tool
        content: str (截断到 500 字)
```

**实现方式**：读取 `logs/agent_trace.jsonl`，按 trace_id 聚合 + 从 OTEL Span 导出数据组装。生产环境应切到 Jaeger API 查询。

### 3.3 日志搜索

```yaml
GET /api/admin/logs
  参数:
    trace_id: str (可选)
    user_id: str (可选)
    level: debug|info|warning|error (可选)
    keyword: str (可选, 搜索 msg 字段)
    start: ISO datetime (可选)
    end: ISO datetime (可选)
    limit: int (默认 100)
    offset: int (默认 0)
  返回:
    total: int
    items:
      - ts: ISO datetime
        level: str
        logger: str
        msg: str
        trace_id: str|null
        user_id: str|null
        session_id: str|null
        tool: str|null
        ms: int|null
        exception: str|null
```

**实现方式**：读取 `logs/agent.log`（JSON 格式），按参数过滤。生产环境应切到 Loki API 查询。

### 3.4 Critic 审查记录

```yaml
GET /api/admin/critic-failures
  参数:
    type: grounding|compliance (可选)
    start: ISO datetime
    end: ISO datetime
    limit: int (默认 50)
  返回:
    total: int
    items:
      - ts: ISO datetime
        type: grounding|compliance
        user_message: str (截断 200 字)
        assistant_response: str (截断 200 字)
        note: str
    stats:
      total_today: int
      rate_today: float    # 拦截率
      by_type: {grounding: int, compliance: int}
      trend: [{date: str, count: int}]  # 近7天趋势
```

**实现方式**：读取 `logs/critic_failures.jsonl`。

### 3.5 Token 成本

```yaml
GET /api/admin/token-usage
  参数:
    period: today|7d|30d
  返回:
    total_input: int
    total_output: int
    estimated_cost_usd: float
    by_model: {model: {input: int, output: int}}
    by_day: [{date: str, input: int, output: int, cost: float}]
    top_users: [{user_id: str, tokens: int, cost: float}]  # Top 10
```

**实现方式**：聚合 Prometheus `agent_llm_tokens_total` counter 的增量。

### 3.6 用户会话

```yaml
GET /api/admin/users
  参数:
    search: str (可选, 搜索 user_id)
    sort: recent_active|total_tokens|total_sessions (默认 recent_active)
    limit: int (默认 50)
  返回:
    items:
      - user_id: str
        last_active: ISO datetime
        total_sessions: int
        total_tokens: int
        recent_intents: [str]

GET /api/admin/users/{user_id}/sessions
  返回:
    sessions:
      - session_id: str
        created: ISO datetime
        last_active: ISO datetime
        message_count: int
        intent_distribution: {intent: count}
        traces: [{trace_id, ts, duration_ms, exit}]  # 最近10条

GET /api/admin/sessions/{session_id}/messages
  返回:
    messages:
      - role: user|assistant
        content: str
        ts: ISO datetime
        trace_id: str
        tool_calls: [str]  (assistant 消息时)
```

**实现方式**：查询 `memory/turn_store.py` SQLite + 日志聚合。

### 3.7 意图分布

```yaml
GET /api/admin/intent-stats
  参数:
    period: today|7d|30d
  返回:
    distribution: {intent: count}
    rule_hit_rate: float    # 规则命中率
    trend: [{date, intent, count}]  # 按天趋势
```

**实现方式**：Prometheus `agent_pipeline_requests_total` counter 按 intent 聚合。

### 3.8 工具调用统计

```yaml
GET /api/admin/tool-stats
  参数:
    period: today|7d|30d
  返回:
    calls: {tool: count}
    avg_duration_ms: {tool: float}
    error_rate: {tool: float}
    trend: [{date, tool, count}]
```

---

## 4. 前端页面结构

```
frontend/src/app/admin/
├── layout.tsx              # 管理端共用布局（侧边栏导航）
├── page.tsx                # 重定向到 /admin/overview
│
├── overview/
│   └── page.tsx            # 实时概览 Dashboard
│
├── traces/
│   ├── page.tsx            # Trace 列表 + 搜索
│   └── [traceId]/
│       └── page.tsx        # Trace 详情（瀑布图）
│
├── logs/
│   └── page.tsx            # 日志搜索页
│
├── quality/
│   └── page.tsx            # Critic 审查 + 质量趋势
│
├── cost/
│   └── page.tsx            # Token 成本面板
│
├── users/
│   ├── page.tsx            # 用户列表
│   ├── [userId]/
│   │   └── page.tsx        # 用户详情（session 列表）
│   └── sessions/
│       └── [sessionId]/
│           └── page.tsx    # 会话回放
│
├── intent/
│   └── page.tsx            # 意图分布
│
└── tools/
    └── page.tsx            # 工具调用统计
```

### 4.1 共用组件

| 组件 | 用途 |
|------|------|
| `AdminLayout` | 左侧导航 + 顶栏（时间范围选择器） |
| `StatCard` | 概览页的数字卡片（标题、数值、趋势箭头） |
| `TimeRangeFilter` | 日期范围选择器（今天/7天/30天/自定义） |
| `TraceWaterfall` | Trace 瀑布图/火焰图组件 |
| `LogLine` | 单条日志行（级别颜色 + 展开详情） |
| `StatusBadge` | 状态徽章（healthy=绿, unhealthy=红, error=红） |

### 4.2 概览 Dashboard 布局

```
┌──────────────────────────────────────────────────────────────┐
│  📊 实时概览                             时间范围: [今天 ▼]   │
├────────┬────────┬────────┬────────┬────────┬─────────────────┤
│ 请求量  │ P50    │ P99    │ 错误率  │ TTFT   │ 在线 Session     │
│ 42/min │ 2.1s   │ 8.7s   │ 0.2%   │ 0.8s   │ 12              │
│  ↑12%  │ ↓5%    │ →持平   │ ↓0.1%  │ →持平   │                 │
├────────┴────────┴────────┴────────┴────────┴─────────────────┤
│                                                               │
│  📈 Pipeline 延迟趋势（折线图，P50/P90/P99 三条线）             │
│                                                               │
├──────────────────────┬────────────────────────────────────────┤
│  依赖健康             │  Critic 拦截                          │
│  🟢 Redis            │  ┌──────┬──────┬──────┐              │
│  🟢 Qdrant           │  │ 今日  │ 本周  │ 趋势  │              │
│  🟢 Mem0             │  │ 3 次  │ 12 次 │ →持平 │              │
│                      │  │ 0.5%  │ 0.3%  │       │              │
├──────────────────────┴────────────────────────────────────────┤
│  🔧 工具调用分布（今日）              💰 Token 消耗（今日）      │
│  search_memory  ████████ 38         Input  1.2M tokens        │
│  web_search     ████ 25             Output  320K tokens       │
│  search_rag     ███ 14              预估    $4.72              │
│  generate_report █ 3                                         │
├───────────────────────────────────────────────────────────────┤
│  ⚠️ 最近慢请求 (P99+)                                         │
│  trace_id | 时间 | 用户 | intent | 耗时 | 工具轮次            │
│  abc123   | 10:30 | u1  | symptom | 12.3s | 3 轮             │
│  def456   | 10:28 | u2  | medicat | 9.8s  | 2 轮             │
└───────────────────────────────────────────────────────────────┘
```

### 4.3 Trace 详情（瀑布图）

```
┌──────────────────────────────────────────────────────────────┐
│  Trace: abc123def456                    ← 返回列表            │
│  时间: 2026-05-22 10:30:01  用户: user_001  意图: symptom    │
│  总耗时: 4.2s  工具轮次: 2  退出: normal                      │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  瀑布图 (Waterfall):                                          │
│                                                               │
│  0ms    1000ms   2000ms   3000ms   4000ms                     │
│  ├──────┼────────┼────────┼────────┤                         │
│  pipeline                                 ██████████████████ │
│  ├─ triage                            ██                     │
│  ├─ agent                               █████████████████    │
│  │   ├─ round.0                         ██████████           │
│  │   │   ├─ llm.invoke               ████                    │
│  │   │   └─ tools                     ██████                 │
│  │   │       ├─ search_memory (//)  ██                       │
│  │   │       └─ web_search (//)     ██████ ← 瓶颈 950ms      │
│  │   └─ round.1                          ███                 │
│  │       ├─ llm.invoke                ██                     │
│  │       └─ search_rag                 █                     │
│  ├─ critic                                ██                 │
│  └─ hooks                                   ███              │
│      ├─ memory_write                     ██                  │
│      └─ session_persist                   █                   │
│                                                               │
│  点击任意 Span → 展开属性面板:                                  │
│  ┌─────────────────────────────────────┐                     │
│  │ Span: tool.web_search               │                     │
│  │ Duration: 950ms                     │                     │
│  │ tool.result_len: 3200               │                     │
│  │ tool.error: false                   │                     │
│  │ tool.query_hash: md5:abc123         │                     │
│  └─────────────────────────────────────┘                     │
└──────────────────────────────────────────────────────────────┘
```

---

## 5. 路由设计

```
/admin                  → 重定向到 /admin/overview
/admin/overview         → 实时概览 Dashboard
/admin/traces           → Trace 列表
/admin/traces/[id]      → Trace 详情（瀑布图）
/admin/logs             → 日志搜索
/admin/quality          → Critic 审查面板
/admin/cost             → Token 成本
/admin/users            → 用户列表
/admin/users/[id]       → 用户详情 + session 列表
/admin/sessions/[id]    → 会话消息回放
/admin/intent           → 意图分布
/admin/tools            → 工具调用统计
```

---

## 6. 文件清单

### 后端新增

| 文件 | 说明 |
|------|------|
| `backend/routers/admin.py` | 管理端所有 API 端点 |
| `backend/admin/trace_store.py` | JSONL trace 读取 + 聚合逻辑 |
| `backend/admin/log_store.py` | JSON 日志读取 + 搜索 |
| `backend/admin/metrics_reader.py` | Prometheus /metrics 解析 |

### 前端新增

| 文件 | 说明 |
|------|------|
| `src/app/admin/layout.tsx` | 管理端布局 + 侧边栏 |
| `src/app/admin/page.tsx` | 重定向 |
| `src/app/admin/overview/page.tsx` | 概览 Dashboard |
| `src/app/admin/traces/page.tsx` | Trace 列表 |
| `src/app/admin/traces/[traceId]/page.tsx` | Trace 详情 |
| `src/app/admin/logs/page.tsx` | 日志搜索 |
| `src/app/admin/quality/page.tsx` | Critic 审查 |
| `src/app/admin/cost/page.tsx` | Token 成本 |
| `src/app/admin/users/page.tsx` | 用户列表 |
| `src/app/admin/users/[userId]/page.tsx` | 用户详情 |
| `src/app/admin/sessions/[sessionId]/page.tsx` | 会话回放 |
| `src/app/admin/intent/page.tsx` | 意图分布 |
| `src/app/admin/tools/page.tsx` | 工具统计 |
| `src/components/admin/StatCard.tsx` | 数字卡片 |
| `src/components/admin/TraceWaterfall.tsx` | 瀑布图组件 |
| `src/components/admin/LogLine.tsx` | 日志行组件 |
| `src/components/admin/StatusBadge.tsx` | 状态徽章 |
| `src/components/admin/TimeRangeFilter.tsx` | 时间范围选择器 |
| `src/lib/admin-api.ts` | 前端 API 调用封装 |

---

## 7. 开发顺序

| 阶段 | 模块 | 预估工时 | 依赖 |
|------|------|---------|------|
| **P0** | 后端 admin API（overview + traces + logs） | 1.5 天 | 无 |
| **P0** | 概览 Dashboard 页 | 1 天 | 后端 API |
| **P0** | Trace 查看器（列表 + 详情 + 瀑布图） | 1.5 天 | 后端 API |
| **P0** | 日志搜索页 | 0.5 天 | 后端 API |
| **P1** | Critic 审查面板 | 0.5 天 | critic_failures.jsonl |
| **P1** | Token 成本面板 | 0.5 天 | Prometheus |
| **P1** | 工具调用统计 | 0.5 天 | Prometheus |
| **P2** | 用户会话浏览器 | 1 天 | turn_store |
| **P2** | 意图分布 | 0.5 天 | Prometheus |
| **P2** | 管理端权限（JWT + role check） | 0.5 天 | AuthMiddleware |

---

## 8. 注意事项

1. **日志文件读取**：`logs/agent_trace.jsonl` 和 `logs/agent.log` 是文件读取，不做索引。数据量大时切到 Elasticsearch/Loki。当前 JSONL 约 450KB，预计可撑到 10 万条 trace 之前不用换。

2. **管理端鉴权**：所有 `/api/admin/*` 端点必须加 admin role 检查。在 `AuthMiddleware` 或单独 middleware 中校验 JWT 中的 `role: admin`。

3. **Trace 瀑布图**：前端需要自己画瀑布图（HTML5 Canvas 或 SVG），没有特别好的 React 现成组件。建议用简单的 div + CSS 实现（每个 Span 是一个绝对定位的色块）。

4. **Prometheus 数据**：`/api/admin/overview` 里解析 `/metrics` 文本是临时方案。正式环境 Prometheus 提供 HTTP API (`/api/v1/query`) 可以直接查即时值。

5. **后端注册**：别忘了在 `backend/main.py` 里 `app.include_router(admin.router, prefix="/api/admin")`。
