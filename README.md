# 健康管理 AI Agent

> 基于 Triage / Agent Graph / Critic 三层 Pipeline 的个人健康管理 AI Agent，支持语义记忆检索、医学知识库 RAG、联网搜索溯源、结构化问诊、健康预警广播与 PDF 健康报告导出

## 📝 项目简介

面向个人的健康管理 AI 助手，帮助用户以自然对话的方式记录、追踪和回顾自己的健康状况。

**解决的问题：**

- 健康信息散乱，难以系统记录和回溯
- 看病时记不清上次症状和用药情况
- 反复发作的症状容易被忽视，无法及时发现规律

## ✨ 核心功能

- [x] **症状记录**：通过自然对话记录症状、就医经历、用药情况
- [x] **记忆搜索**：描述症状时自动关联 Mem0 历史记录，主动提示相关信息
- [x] **医学知识库 RAG**：药物说明、疾病描述、临床指南从 Qdrant 知识库检索，不依赖模型训练知识
- [x] **联网搜索 + 来源链接**：Brave Search 接入，回复末尾自动追加可点击的参考来源，近期新药/最新研究强制走联网
- [x] **结构化问诊**：发送 `/consult` 启动问诊模式，逐步收集完整症状信息后交主 Agent 分析
- [x] **健康报告 PDF**：一键生成结构化健康报告并下载为 PDF（Playwright 渲染，中文完整支持）
- [x] **个性化建议**：发送 `/recommend` 基于历史记录生成作息、饮食、运动、就医建议
- [x] **健康预警广播**：APScheduler 定时抓取药品召回/疫情预警，与用户 Mem0 档案匹配后通过 Redis Pub/Sub + SSE 实时推送
- [x] **紧急症状识别**：胸痛、骤停、重度出血等危重症状自动短路，立即提示拨打 120
- [x] **合规质检**：Critic Agent 审查每条回复，防止明确诊断、不当用药建议等违规输出
- [x] **Token 级流式输出**：LangGraph `astream_events` 驱动，字符逐个打出，无等待感

## 🛠️ 技术栈

| 层级 | 技术 |
|------|------|
| Agent 框架 | [LangGraph](https://github.com/langchain-ai/langgraph)（单节点 Orchestrator + 工具循环） |
| LLM | Claude Sonnet 4.6（主 Agent）/ Claude Haiku 4.5（Triage、Critic、问诊） |
| 向量记忆 | [Mem0](https://mem0.ai/)（语义搜索 + 去重） |
| 医学知识库 | Qdrant + BGE-M3（dense + sparse 双路 RRF 融合检索） |
| 联网搜索 | [Brave Search API](https://api.search.brave.com) |
| 后端框架 | FastAPI + APScheduler |
| 前端框架 | Next.js 14 + Tailwind CSS |
| 会话缓存 | Redis（会话历史 / 记忆预热 / 健康预警 Pub/Sub） |
| PDF 渲染 | Playwright（headless Chromium） |
| 容器编排 | Docker Compose |

## 🏗️ 系统架构

### 对话 Pipeline

每条用户消息经过以下 Pipeline 处理：

```
用户消息
    │
    ▼
[L1] Triage Gate          Haiku 快速判定（< 500ms）
    │  紧急 → 短路返回 120 提示
    │  通过 ↓
[L2] Main Agent Graph     Sonnet 4.6，工具调用循环（最多 5 轮）
    │  ┌──────────────────────────────────────────────┐
    │  │  orchestrator_node                            │
    │  │    ↓ 按需调用                                │
    │  │  search_memory   ← Mem0 历史记录检索         │
    │  │  search_rag       ← Qdrant 医学知识库        │
    │  │  web_search       ← Brave Search 联网搜索    │
    │  │  generate_report  ← 健康报告生成             │
    │  └──────────────────────────────────────────────┘
    │  astream_events → token 逐个流出给前端
    │
    ▼
web_sources 追加          web_search 来源链接拼入回复末尾
    │
    ▼
[L3] Critic Review        Haiku 合规质检
    │  未通过 → 追加补充说明 + 记入 critic_store
    │  通过 ↓
    ▼
on_stop_hooks             Mem0 写入 + Redis 缓存刷新（异步 fire-and-forget）
    │
    ▼
SSE → 前端
```

### 健康预警广播

```
[APScheduler 每 6 小时]
    │
    ▼
alert_monitor.py
    ├── Brave Search 抓取药品召回 / 疫情预警 / 健康风险提示
    ├── Haiku 解析结构化列表 {title, summary, keywords}
    └── 遍历所有用户
            ├── keywords 检索 Mem0 档案（相关性匹配）
            ├── 命中 → Redis PUBLISH alert:{user_id}
            └── alert_sent:{user_id}:{hash} 7 天去重

[GET /api/notify/stream]  ← 前端 SSE 长连接
    └── Redis SUBSCRIBE alert:{user_id} → 推送弹窗
```

**Critic 反馈闭环：** Critic 拦截的违规案例写入 `logs/critic_failures.jsonl`，下次 L2 启动时读取最近 3 条注入 system prompt，让主 Agent 主动规避同类错误。

## 🚀 快速开始

### 环境要求

- Docker & Docker Compose
- Python 3.11+
- Node.js 18+
- API Key：
  - [Anthropic](https://console.anthropic.com/settings/keys)（必须）
  - [Mem0](https://app.mem0.ai/dashboard/api-keys)（必须）
  - [Brave Search](https://api.search.brave.com)（联网搜索 + 健康预警，有免费层）

### 1. 克隆项目

```bash
git clone https://github.com/Jennifer-00/health-agent.git
cd health-agent
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 API Key：

```env
ANTHROPIC_API_KEY=sk-ant-...
MEM0_API_KEY=...
BRAVE_API_KEY=...
JWT_SECRET=（用 openssl rand -hex 32 生成）
```

### 3. 启动所有服务

```bash
docker compose up -d
```

| 服务 | 地址 |
|------|------|
| 前端界面 | http://localhost:3000 |
| 后端 API | http://localhost:8000 |
| API 文档 | http://localhost:8000/docs |
| Qdrant 控制台 | http://localhost:6333/dashboard |

### 4. 本地开发启动

```bash
# 安装 Python 依赖
pip install -e .

# 安装 Playwright Chromium（PDF 导出需要）
playwright install chromium

# 启动后端
uvicorn backend.main:app --reload --port 8000

# 启动前端
cd frontend && npm install && npm run dev
```

> 本地开发只需启动 Redis 和 Qdrant：`docker compose up redis qdrant -d`

## 📖 使用说明

| 操作 | 方式 |
|------|------|
| 症状记录 | 直接描述，Agent 自动记录并关联历史 |
| 结构化问诊 | 发送 `/consult` |
| 个性化建议 | 发送 `/recommend` |
| 健康报告 PDF | 点击界面右上角「下载报告」按钮 |
| 健康预警 | 自动推送，无需操作（需保持页面打开） |

## 🎯 设计亮点

- **单节点 + 内部工具循环**：LangGraph 图只有一个 `orchestrator` 节点，工具循环在节点内部管理（最多 5 轮）。决策权始终在同一模型，避免多节点状态序列化开销。

- **来源链接确定性追加**：`web_search` 结果中的 URL 在 `nodes.py` 工具循环内用正则提取，存入 `AgentState.web_sources`，由 `harness.py` 在 Critic Review 之前以 SSE delta 推送给前端，不依赖 LLM 自行引用。

- **Token 级 SSE 流式输出**：使用 `graph.astream_events(version="v2")` 捕获 `on_chat_model_stream` 事件，每个 token 实时推送；工具调用事件从 `on_chat_model_end` 的 `tool_calls` 字段提取，在 LLM 决策阶段即发出，无需 LangGraph ToolNode。

- **Redis Pub/Sub 健康预警**：预警监控与对话 Pipeline 完全解耦。APScheduler 后台定时运行，通过 Redis Pub/Sub 推送到各用户的 SSE 长连接，7 天哈希去重避免重复打扰。

- **Critic 自学习**：违规案例持久化为 JSONL，下次对话自动注入 system prompt 作为负样本反例，形成闭环。

- **RAG 双路融合**：BGE-M3 同时生成 dense 向量和 sparse 词袋，Qdrant 原生 RRF（Reciprocal Rank Fusion）融合两路召回，比单路检索准确率更高。

- **XML 结构化 Prompt**：System prompt 采用 XML 标签划分角色、工具、触发条件与禁止行为，利用 Claude 对 XML 的边界感知提升指令遵循准确性。

- **Prompt Cache**：system prompt 使用 Anthropic `cache_control: ephemeral`，重复对话减少 token 消耗。

- **记忆预热**：后端启动和前端页面加载时双重触发 Mem0 检索，结果写入 Redis prefetch 缓存；后续 `search_memory` 直接命中缓存，消除冷启动延迟。

## 📊 性能数据

由 `measure.py` 及 `logs/agent_trace.jsonl` 实测（本地环境）：

| 指标 | 数值 | 说明 |
|------|------|------|
| Triage 准确率 | 96.7%（29/30） | 30 条测试用例 |
| Triage 延迟 avg / p95 | 1470ms / 2533ms | Haiku 快速判定 |
| Critic 拦截准确率 | 100%（13/13） | 7 违规 + 6 合规，无漏拦/误拦 |
| Critic 延迟 avg / p95 | 1487ms / 2638ms | 串行追加在主链路后 |
| 端到端延迟（预热后） | 9.6s – 14.5s | n=3 |

## 📁 项目结构

```
.
├── agent/
│   ├── harness.py          # Pipeline 编排（Triage → Agent → web_sources → Critic → hooks）
│   ├── graph.py            # LangGraph 图定义（单节点）
│   ├── nodes.py            # orchestrator_node 实现，收集 web_sources
│   ├── tools.py            # 工具定义与分发（search_memory / search_rag / web_search / generate_report）
│   ├── alert_monitor.py    # 健康预警监控（定时抓取 → 匹配用户 → Pub/Sub 推送）
│   ├── triage.py           # L1 紧急症状分诊
│   ├── critic.py           # L3 输出合规质检
│   ├── critic_store.py     # 违规案例持久化与注入
│   ├── prompts.py          # XML 结构化系统提示词
│   ├── state.py            # 图状态定义（含 web_sources）
│   ├── intake_graph.py     # /consult 问诊子图
│   └── skills/
│       └── report_gen/     # 健康报告生成 Skill
│           ├── SKILL.md
│           ├── references/
│           └── scripts/    # run.py / formatter.py / pdf_export.py
├── backend/
│   ├── main.py             # FastAPI 入口 + 启动预热 + APScheduler（含预警定时任务）
│   ├── routers/
│   │   ├── chat.py         # SSE 流式对话端点
│   │   ├── notify.py       # SSE 健康预警推送端点
│   │   ├── memory.py       # 记忆查询/管理端点
│   │   └── report.py       # PDF 报告下载端点
│   ├── schemas/
│   └── middleware/auth.py  # JWT 鉴权
├── frontend/
│   ├── app/
│   │   ├── page.tsx
│   │   └── api/            # Next.js Route Handlers（反向代理后端）
│   │       ├── chat/
│   │       ├── memory/
│   │       └── report/pdf/
│   ├── components/
│   │   ├── ChatWindow.tsx  # Markdown 渲染 + 超链接新标签打开
│   │   ├── EmergencyModal.tsx
│   │   └── MemoryPanel.tsx
│   └── lib/
│       ├── api.ts
│       ├── serverProxy.ts
│       └── serverAuth.ts
├── memory/
│   ├── mem0_client.py      # Mem0 向量记忆（读写 + 去重）
│   ├── session_buffer.py   # Redis 会话缓存 + prefetch
│   └── pubsub.py           # Redis Pub/Sub（健康预警发布/订阅）
├── tests/
├── logs/                   # agent_trace.jsonl / critic_failures.jsonl
├── pyproject.toml
├── docker-compose.yml
└── .env.example
```

## 🔮 未来计划

- [ ] 支持上传检查报告图片，OCR 解析后自动入库
- [ ] Redis prefetch 改为基于查询语义的本地向量过滤，提升检索精度
- [ ] 支持多用户隔离的 RAG 知识库（当前为全局共享）
- [ ] 健康预警前端弹窗 UI 组件

## 📄 许可证

MIT License

## 👤 作者

- GitHub：[@Jennifer-00](https://github.com/Jennifer-00)
- Email：1172988148@qq.com

## 🙏 致谢

- [LangGraph](https://github.com/langchain-ai/langgraph) — Agent 编排框架
- [Mem0](https://mem0.ai/) — 向量记忆层
- [Anthropic](https://anthropic.com/) — Claude 模型
- [Qdrant](https://qdrant.tech/) — 向量数据库
- [Brave Search](https://search.brave.com/) — 联网搜索
- [Playwright](https://playwright.dev/) — PDF 渲染
