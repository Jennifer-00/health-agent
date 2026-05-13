import posthog
posthog.disabled = True
posthog.api_key = "disabled"  # 即使有队列事件也无法上报

import logging
import os
from dotenv import load_dotenv
load_dotenv(override=True)  # 必须在所有业务模块 import 之前执行

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from backend.routers import auth, chat, memory, report, notify
from backend.middleware.auth import AuthMiddleware
from memory.mem0_client import list_users
from backend.routers.chat import _warmup_task

logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()


async def _startup_warmup():
    """后端启动时对所有已有用户做记忆预热。OSS 模式下 list_users 返回空列表，跳过批量预热。"""
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
    """预加载 BGE-M3 / reranker，消除首次 search_memory 的冷启动延迟。"""
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

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(memory.router)
app.include_router(report.router)
app.include_router(notify.router)


@app.get("/health")
async def health_check():
    return {"status": "ok"}
