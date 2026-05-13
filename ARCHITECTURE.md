# 健康 Agent 系统架构文档

## 目录

1. [系统概览](#1-系统概览)
2. [请求处理管线](#2-请求处理管线)
3. [Agent 三层结构](#3-agent-三层结构)
4. [工具层](#4-工具层)
5. [记忆架构](#5-记忆架构)
6. [会话生命周期与记忆写入策略](#6-会话生命周期与记忆写入策略)
7. [Mem0 OSS 配置详解](#7-mem0-oss-配置详解)
8. [后台任务](#8-后台任务)
9. [认证与中间件](#9-认证与中间件)
10. [前端架构](#10-前端架构)
11. [技术栈总览](#11-技术栈总览)

---

## 1. 系统概览

```
┌─────────────────────────────────────────────────────────┐
│                      前端 (Next.js)                      │
│   ChatWindow → useAgentChat → /api/chat (SSE stream)    │
└───────────────────────┬─────────────────────────────────┘
                        │ HTTP / SSE
┌───────────────────────▼─────────────────────────────────┐
│                   后端 (FastAPI)                         │
│   AuthMiddleware → chat.py → AgentHarness               │
│                                                         │
│   ┌─────────┐   ┌──────────────┐   ┌─────────────────┐ │
│   │ Triage  │   │  LangGraph   │   │     Critic      │ │
│   │ (Haiku) │→  │  (Sonnet)    │→  │    (Haiku)      │ │
│   └─────────┘   └──────┬───────┘   └─────────────────┘ │
│                        │ 工具调用                        │
│          ┌─────────────┼─────────────┐                  │
│          ▼             ▼             ▼                  │
│    search_memory   search_rag    web_search             │
└────────────────────────────────────────────────────────-┘
                        │
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼
    Mem0 OSS          Qdrant           Redis
  (长期记忆)         (RAG知识库)      (会话缓冲)
        │
   ┌────┴─────┐
   ▼          ▼
 Qdrant    Neo4j
(向量存储) (图存储)
```

---

## 2. 请求处理管线

### 普通对话模式

```
POST /chat
  │
  ▼
AuthMiddleware（JWT 验证 → request.state.user_id）
  │
  ▼
chat.py
  ├─ 从 Redis 读取 session 历史（最近 10 条）
  └─ AgentHarness.run()
       │
       ▼
  [L1] Triage Gate（Haiku，< 500ms）
       ├─ 紧急症状 → 直接返回急救提示，跳过 L2/L3
       └─ 正常 → 继续
       │
       ▼
  [L2] LangGraph Orchestrator（Sonnet，工具 loop 最多 5 轮）
       │
       ▼
  [L3] Critic Review（Haiku，合规审查）
       ├─ 通过 → 追加补充说明
       └─ 失败案例记录到 logs/critic_failures.jsonl
       │
       ▼
  SSE 流式推送回前端
       │
       ▼
  on_stop_hooks（fire-and-forget asyncio.Task）
       ├─ session_hook：会话历史写回 Redis
       └─ memory_write_hook：记忆写入（见第 6 节）
```

### 问诊模式（`/consult`）

```
用户发送 /consult
  │
  ▼
IntakeGraph（Haiku 多轮问诊）
  ├─ 每轮追问症状，状态存储在 Redis consult:{session_id}
  └─ 收集到 [INTAKE_DONE] 信号后
       │
       ▼
  生成问诊摘要 summary
       │
       ▼
  以 "[问诊摘要] {summary}" 作为 user_message 进入主流程
  （original_user_content = summary，用于记忆写入）
```

---

## 3. Agent 三层结构

### L1 Triage（输入守卫）

- **模型**：`claude-haiku-4-5-20251001`
- **目标**：< 500ms 内判断是否为紧急症状（胸痛、中风、大出血等）
- **短路条件**：检测到紧急症状时，跳过 L2/L3，直接返回急救建议
- **即使短路**，`on_stop_hooks` 仍然触发，紧急症状会被记录进 Mem0

### L2 LangGraph Orchestrator（主推理）

- **模型**：`claude-sonnet-4-6`
- **工具 loop**：最多 5 轮，`asyncio.gather` 并行执行同一轮次的多个工具调用
- **System prompt 增强**：注入最近 3 条 Critic 失败案例作为反例，形成自我修正回路
- **四个工具**：`search_memory`、`search_rag`、`web_search`、`generate_report`

### L3 Critic（输出守卫）

- **模型**：`claude-haiku-4-5-20251001`
- **职责**：合规性审查（不过度诊断、不替代医生、不给出不当建议）
- **失败处理**：审查不通过时，在回复末尾追加补充说明；失败案例写入 `logs/critic_failures.jsonl`

---

## 4. 工具层

所有工具通过 `agent/tools.py` 中的 `dispatch_tool` 统一路由。

### `search_memory` — 用户历史记忆检索

```
检查 Redis prefetch 缓存（TTL 10分钟）
  ├─ 命中 → 直接返回（零延迟）
  └─ 未命中 → Mem0 OSS search()
               embed(query) → Qdrant 向量搜索 + Neo4j 图搜索（并行）
               → bge-reranker-v2-m3 重排
               → 返回最相关的 5 条用户记忆
```

**冷却机制**：`mem_search_cooldown` Redis 计数器，避免同一 session 内频繁重复检索。

### `search_rag` — 医疗知识库检索

```
GPT-4o-mini 查询改写（医疗术语标准化）
  │
  ▼
BAAI/bge-m3 双路编码（dense + sparse）
  │
  ▼
Qdrant 双路 Prefetch（各取 20 条）
  + Qdrant 原生 RRF 融合（20 候选）
  │
  ▼
BAAI/bge-reranker-v2-m3 cross-encoder 重排
  │
  ▼
返回 Top-5 结果（含书名、章节路径、内容）
```

- **Qdrant 连接**：gRPC 端口 6334（`QDRANT_HOST/PORT`）
- **Collection**：`medical_knowledge`
- **所有模型调用通过 `loop.run_in_executor` 避免阻塞事件循环**

### `web_search` — 联网搜索

- Brave Search API（`BRAVE_API_KEY` 缺失时静默降级，不影响启动）
- 返回结果包含标题、摘要、来源链接

### `generate_report` — 健康报告生成

- 子 Agent（Haiku），读取 `agent/skills/report_gen/`
- 从 Mem0 `get_all()` 拉取该用户全部记忆
- 生成 Markdown 格式健康摘要，支持导出 PDF

---

## 5. 记忆架构

系统采用四层记忆结构：

```
┌──────────────────────────────────────────────────────────┐
│ 层级       │ 实现                    │ TTL / 范围         │
├──────────────────────────────────────────────────────────┤
│ L0 Prefetch│ Redis mem_prefetch:uid  │ 10分钟，跨 session │
│ L1 会话    │ Redis session:{sid}     │ 1小时，最近 10 条  │
│ L2 兜底    │ SQLite pending_turns    │ 永久，flushed 标记 │
│ L3 长期    │ Mem0 OSS（Qdrant+Neo4j）│ 永久，向量+图存储  │
└──────────────────────────────────────────────────────────┘
```

### L0：Prefetch 缓存

- Key：`mem_prefetch:{user_id}`（user 级别，跨 session 共享）
- 存储内容：`search("用户基本信息 症状 病史", limit=5)` 的结果
- 刷新时机：session 开始（warmup）、每次 Mem0 写入完成后

### L1：会话历史

- Key：`session:{session_id}`
- 存储最近 N 条对话（`MAX_HISTORY_MESSAGES=10`），作为 LLM 上下文
- 问诊状态独立存储：`consult:{session_id}`

### L2：SQLite 兜底

- 文件路径：`TURN_STORE_PATH`（默认 `data/turns.db`）
- 表结构：`pending_turns(id, user_id, session_id, user_content, asst_content, flushed, created_at)`
- 作用：防止 Redis pending 队列因崩溃/强杀丢失，确保记忆最终一致

### L3：Mem0 OSS 长期记忆

- 向量存储：Qdrant collection `health_memories`
- 图存储：Neo4j（实体关系图）
- 详见第 7 节

---

## 6. 会话生命周期与记忆写入策略

### 写入流程

```
每条消息结束（on_stop_hooks 触发）
  │
  ├─ Redis RPUSH mem_pending:session:{sid}   ← 快速追加
  └─ SQLite INSERT flushed=0                 ← 持久兜底
       │
       ▼
  turn_count % MEM_WRITE_EVERY_N_TURNS == 0？（默认 5 轮）
  ├─ 否 → 等待
  └─ 是 → _flush_pending_to_mem0()
            ├─ drain Redis pending 队列
            ├─ 拼成完整对话文本
            ├─ Mem0 add()（2次 Haiku + N次向量搜索）
            ├─ SQLite 标记 flushed=1
            └─ 刷新 Redis prefetch 缓存
```

### 会话关闭时（`/chat/end`）

```
前端 beforeunload / 组件卸载
  │ fetch keepalive: true
  ▼
POST /chat/end
  │
  ▼
drain 剩余 pending（可能 < 5 轮）→ Mem0 add()（fire-and-forget）
```

### 会话开始时（`/chat/warmup`）

```
前端页面加载
  │
  ▼
POST /chat/warmup
  │
  ▼
① 检查 SQLite flushed=0 的行（崩溃恢复）
   └─ 若有 → 批量 Mem0 add() → 标记 flushed=1
       │
       ▼
② Mem0 search("用户基本信息 症状 病史", limit=5)
   └─ 结果写入 Redis prefetch 缓存（TTL 10分钟）
```

### 为什么这样设计

| 目标 | 机制 |
|------|------|
| 减少 Mem0 API 调用频率 | 攒 5 轮批量写，而非每条写一次 |
| 防止浏览器强杀丢数据 | SQLite 每轮同步兜底，下次 warmup 恢复 |
| 首条消息无冷启动延迟 | warmup 预热 prefetch 缓存 |
| 写入失败不影响响应 | on_stop_hooks 全为 fire-and-forget asyncio.Task |

---

## 7. Mem0 OSS 配置详解

### 组件选型

```python
{
  "llm":         Anthropic claude-haiku-4-5   # 事实提取 + 记忆决策
  "embedder":    BAAI/bge-m3 (HuggingFace)   # 1024维 dense，与 RAG 一致
  "vector_store": Qdrant HTTP :6333           # collection: health_memories
  "reranker":    BAAI/bge-reranker-v2-m3     # search() 结果重排，与 RAG 一致
  "graph_store": Neo4j bolt://localhost:7687  # 实体关系图
  "custom_fact_extraction_prompt": ...        # 完整替换默认提取 prompt
}
```

### `add()` 内部流程（每次 flush 触发）

```
输入：5轮对话拼接的文本
  │
  ▼
① Haiku 提取事实列表
   ["最近三天眼睛酸涩", "盯屏幕后加重", ...]
  │
  ▼
② 每条事实 embed → Qdrant 搜 Top-5 相似旧记忆（去重合并）
  │
  ▼
③ Haiku 决策：对比新事实 vs 旧记忆
   → ADD（新建）/ UPDATE（覆盖）/ DELETE（移除）/ NONE（不动）
  │
  ▼
④ 执行 Qdrant 向量写操作
  │
  ▼
⑤ Haiku 提取实体和关系 → 写入 Neo4j 图
   （例：User -HAS_SYMPTOM→ 眼睛酸涩 -TRIGGERED_BY→ 长时间看屏幕）
```

**总计：每次 flush = 3次 Haiku 调用 + N次 Qdrant 操作 + 1次 Neo4j 写入**

### `search()` 内部流程（每次 search_memory 工具调用）

```
embed(query) → Qdrant 向量搜索
               Neo4j 图搜索（并行）
  │
  ▼
bge-reranker-v2-m3 对合并结果重排
  │
  ▼
返回 Top-5 记忆（含向量相关性 + 图关系上下文）
```

### 自定义提取 Prompt

完整替换 Mem0 默认 `USER_MEMORY_EXTRACTION_PROMPT`，专注健康信息：

- **提取**：症状（部位/程度/持续时间）、用药、诊断史、检查结果、过敏史、生活习惯、健康目标
- **忽略**：闲聊、用户的疑问句、模糊猜测、假设性问题
- **保留**：相对时间表达原文（"最近三天"不转换为绝对日期）
- **语言**：自动检测，中文输入存中文

---

## 8. 后台任务

### APScheduler 健康预警（每 6 小时）

```
run_alert_monitor()
  │
  ▼
Brave Search 抓取健康预警新闻
  │
  ▼
Haiku 解析：提取关键词、受影响人群
  │
  ▼
遍历所有用户（OSS 模式下暂跳过，list_users() 返回 []）
  └─ Mem0 search(keywords) 匹配用户档案
       └─ 匹配 → Redis Pub/Sub 推送实时通知
                  去重 key：alert_sent:{user_id}:{md5[:12]}，TTL 7天
```

> **注**：OSS 模式下 `list_users()` 返回空列表，健康预警的主动推送功能暂时停用。若需启用，可通过 Qdrant scroll API 自行实现用户枚举。

### 应用启动（lifespan）

```python
await init_db()          # SQLite 建表（幂等）
asyncio.create_task(_startup_warmup())  # OSS 模式跳过批量预热
scheduler.start()        # APScheduler 启动
```

### 关闭时

等待所有 `_background_tasks` 完成（超时 20 秒后强制取消），确保进行中的 Mem0 写入尽量完成。

---

## 9. 认证与中间件

### JWT 认证

- `AuthMiddleware`：拦截所有请求，验证 `Authorization: Bearer <token>`
- 验证通过后注入 `request.state.user_id`，下游代码直接使用
- 演示模式支持单用户（`DEMO_EMAIL` / `DEMO_PASSWORD`）

### 豁免路径

`/health`、`/auth/login`、`/auth/register` 不需要 JWT。

---

## 10. 前端架构

### 技术栈

- **框架**：Next.js App Router
- **状态管理**：`useAgentChat` hook（`useState` + `useCallback`）
- **通信**：SSE（Server-Sent Events）实时流式接收

### SSE 事件类型

| event.type | 含义 |
|------------|------|
| `text` | 回复文本增量（流式） |
| `status` | 状态提示（"Agent 正在思考..."） |
| `tool_call` | 工具调用摘要（显示在侧边栏） |
| `mode` | 模式切换（`chat` / `consult`） |
| `emergency` | 紧急症状，触发弹窗 |
| `done` | 本次回复结束 |

### 会话生命周期

```
页面加载
  → POST /api/chat/warmup（记忆预热 + SQLite 崩溃恢复）

用户发消息
  → POST /api/chat（SSE 流）

页面关闭 / 组件卸载
  → POST /api/chat/end（keepalive: true，flush 剩余记忆）
```

### API 代理层

所有 `/api/*` 请求通过 Next.js Server Component 代理到 `BACKEND_BASE_URL`（默认 `http://127.0.0.1:8000`），自动附加 JWT Token。

---

## 11. 技术栈总览

### 模型分工

| 模型 | 用途 |
|------|------|
| `claude-sonnet-4-6` | 主 Orchestrator（工具调用 + 最终回复） |
| `claude-haiku-4-5-20251001` | Triage、Critic、IntakeGraph、report subagent、Mem0 事实提取 + 记忆决策 + 图实体提取 |
| `gpt-4o-mini` | RAG 查询改写（医疗术语标准化） |
| `BAAI/bge-m3` | RAG 双路编码（dense+sparse）& Mem0 向量嵌入 |
| `BAAI/bge-reranker-v2-m3` | RAG cross-encoder 重排 & Mem0 search 重排 |

### 基础设施

| 服务 | 用途 | 端口 |
|------|------|------|
| Redis | 会话缓冲、prefetch 缓存、pending 队列、alert 去重 | 6379 |
| Qdrant | RAG 医疗知识库（`medical_knowledge`）& Mem0 用户记忆（`health_memories`） | 6333 HTTP / 6334 gRPC |
| Neo4j | Mem0 图存储（实体关系） | 7687 |
| SQLite | 会话轮次兜底（`data/turns.db`） | 本地文件 |

### 关键依赖

```
fastapi / uvicorn     — 后端框架
langgraph             — Agent 编排
mem0ai                — OSS 记忆管理
FlagEmbedding         — BGE-M3 / BGE-Reranker（RAG）
sentence-transformers — BGE-M3（Mem0 embedder）
qdrant-client         — 向量数据库客户端
redis                 — 会话缓冲
aiosqlite             — SQLite 异步驱动
apscheduler           — 定时任务
anthropic             — Claude API SDK
```

### 环境变量速查

| 变量 | 用途 | 必填 |
|------|------|------|
| `ANTHROPIC_API_KEY` | Claude 模型调用 & Mem0 LLM | ✅ |
| `OPENAI_API_KEY` | RAG 查询改写（gpt-4o-mini） | ✅ |
| `REDIS_URL` | Redis 连接地址 | ✅ |
| `QDRANT_URL` | Qdrant HTTP 地址（Mem0 用） | ✅ |
| `QDRANT_HOST/PORT` | Qdrant gRPC（RAG 用，默认 6334） | ✅ |
| `NEO4J_URI/USER/PASSWORD` | Neo4j 连接信息 | ✅ |
| `JWT_SECRET` | JWT 签名密钥 | ✅ |
| `BRAVE_API_KEY` | 联网搜索（缺失时静默降级） | ❌ |
| `MEM0_COLLECTION` | Mem0 Qdrant collection 名 | ❌ |
| `TURN_STORE_PATH` | SQLite 文件路径 | ❌ |
| `MEM_WRITE_EVERY_N_TURNS` | Mem0 批量写入触发轮数（默认 5） | ❌ |
| `MAX_HISTORY_MESSAGES` | Redis 会话保留条数（默认 10） | ❌ |
| `SESSION_TTL_SECONDS` | Redis session TTL（默认 3600） | ❌ |
