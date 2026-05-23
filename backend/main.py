import posthog
posthog.disabled = True
posthog.api_key = "disabled"  # 即使有队列事件也无法上报

import logging
import os
import time
from dotenv import load_dotenv
load_dotenv(override=True)  # 必须在所有业务模块 import 之前执行

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from backend.routers import auth, chat, memory, report, notify
from backend.middleware.auth import AuthMiddleware
from memory.mem0_client import list_users
from backend.routers.chat import _warmup_task

logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()


async def _startup_warmup():
    try:
        users = await list_users()
        for user in users:
            user_id = user.get("name", "")
            if not user_id or user_id.isdigit():
                continue
            await _warmup_task(user_id)
        if users:
            logger.info("[startup] 预热完成，共处理 %d 个用户", len(users))
        else:
            logger.info("[startup] OSS 模式，跳过批量预热（用户首次请求时按需预热）")
    except Exception as e:
        logger.warning("[startup] 预热失败（不影响正常启动）: %s", e)


async def _warmup_models() -> None:
    try:
        from memory.mem0_client import _get_client
        import asyncio
        await asyncio.to_thread(_get_client)
        logger.info("[startup] Mem0 OSS 模型预热完成")
    except Exception as exc:
        logger.warning("[startup] 模型预热失败（不影响启动）: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

    # ── 结构化日志 + OpenTelemetry 初始化 ──────────────────────────────────
    from agent.logging_config import setup_logging
    setup_logging()
    logger.info("[startup] 结构化日志已初始化")

    try:
        from agent.telemetry import init_telemetry
        init_telemetry()
        logger.info("[startup] OpenTelemetry 已初始化")
    except Exception as exc:
        logger.warning("[startup] OpenTelemetry 初始化失败: %s", exc)

    from memory.turn_store import init_db
    from memory.profile_graph import init_schema as init_profile_schema
    await init_db()
    asyncio.create_task(init_profile_schema())
    asyncio.create_task(_warmup_models())
    asyncio.create_task(_startup_warmup())

    from agent.alert_monitor import run_alert_monitor
    scheduler.add_job(run_alert_monitor, "interval", hours=6, id="health_alert_monitor")

    scheduler.start()

    yield

    scheduler.shutdown(wait=False)

    from agent.harness import _background_tasks
    tasks = list(_background_tasks)
    if tasks:
        logger.info(f"[shutdown] 等待 {len(tasks)} 个后台 task 完成...")
        done, pending = await asyncio.wait(tasks, timeout=20)
        for t in pending:
            t.cancel()
        if pending:
            logger.warning(f"[shutdown] {len(pending)} 个 task 超时被取消")


app = FastAPI(title="健康 Agent API", version="0.1.0", lifespan=lifespan)

_cors_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(AuthMiddleware)


# ── HTTP 指标中间件 ────────────────────────────────────────────────────────────

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """
    为每个 HTTP 请求记录:
      - 请求计数 (method, endpoint, status_code)
      - 请求耗时 (method, endpoint)
    """
    t0 = time.monotonic()
    response = await call_next(request)
    duration = time.monotonic() - t0

    try:
        from agent.metrics import http_requests, http_request_duration
        http_requests.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=str(response.status_code),
        ).inc()
        http_request_duration.labels(
            method=request.method,
            endpoint=request.url.path,
        ).observe(duration)
    except Exception:
        pass  # metrics 不应中断请求

    return response


# ── 路由注册 ────────────────────────────────────────────────────────────────────

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(memory.router)
app.include_router(report.router)
app.include_router(notify.router)


# ── 健康检查 + 指标端点 ────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """
    扩展健康检查：除服务状态外，还探测关键依赖。

    面试要点:
      /health 不应只是 return ok——它应该是依赖拓扑的快照。
      负载均衡器和 k8s 用这个端点决定是否路由流量。
    """
    status = {
        "status": "ok",
        "service": "health-agent",
        "version": "0.1.0",
        "checks": {
            "redis": await _check_redis(),
            "qdrant": await _check_qdrant(),
            "mem0": await _check_mem0(),
        },
    }
    # 任一依赖不健康时返回 503，但不影响响应体
    if any(v == "unhealthy" for v in status["checks"].values()):
        from fastapi.responses import JSONResponse
        return JSONResponse(status, status_code=503)
    return status


@app.get("/metrics")
async def metrics():
    """
    Prometheus 指标端点。

    面试要点:
      /metrics 是 Prometheus scrape 的标准入口。
      所有 Counter/Histogram/Gauge 在这里序列化为文本格式。
      不存磁盘——Prometheus 定期拉取并写入自己的 TSDB。
    """
    try:
        from agent.metrics import get_metrics_text, CONTENT_TYPE_LATEST
        return Response(get_metrics_text(), media_type=CONTENT_TYPE_LATEST)
    except ImportError:
        return Response("# prometheus_client not installed\n", media_type="text/plain")


# ── 依赖健康探测 ────────────────────────────────────────────────────────────────

async def _check_redis() -> str:
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
        await r.ping()
        await r.aclose()
        return "healthy"
    except Exception:
        return "unhealthy"


async def _check_qdrant() -> str:
    try:
        from qdrant_client import AsyncQdrantClient
        client = AsyncQdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6334")),
            prefer_grpc=True,
        )
        # Qdrant gRPC 没有 ping，用 collections list 替代
        await client.get_collections()
        await client.close()
        return "healthy"
    except Exception:
        return "unhealthy"


async def _check_mem0() -> str:
    try:
        from memory.mem0_client import _get_client
        # 只检查客户端能否创建，不实际搜索
        import asyncio
        await asyncio.to_thread(_get_client)
        return "healthy"
    except Exception:
        return "unhealthy"
