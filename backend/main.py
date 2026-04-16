import posthog
posthog.disabled = True
posthog.api_key = "disabled"  # 即使有队列事件也无法上报

import logging
from dotenv import load_dotenv
load_dotenv()  # 必须在所有业务模块 import 之前执行

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from backend.routers import chat, memory
from backend.middleware.auth import AuthMiddleware
from agent.skills.memory_consolidate import _consolidate_mem0
from memory.mem0_client import Mem0Client
from backend.routers.chat import _warmup_task

logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()


async def _scheduled_consolidate():
    """定时任务：合并所有用户的 Mem0 重复记忆，完成后写入系统通知。"""
    from datetime import datetime
    print("[scheduler] 开始定时记忆合并...")
    try:
        mem0 = Mem0Client(user_id="")
        users_resp = await mem0._mem.users()
        users = users_resp if isinstance(users_resp, list) else users_resp.get("results", [])
        for user in users:
            user_id = user.get("name", "")
            # 跳过空值和 Mem0 内部自动生成的纯数字用户
            if not user_id or user_id.isdigit():
                continue
            result = await _consolidate_mem0(user_id)
            print(f"[scheduler] user={user_id!r}: {result}")

            # 写入系统通知，前端记忆档案可见
            ts = datetime.now().strftime('%Y-%m-%d %H:%M')
            note = f"[系统通知] {ts} 自动整理完成：{result}"
            client = Mem0Client(user_id=user_id)
            await client._mem.add(
                note,
                user_id=user_id,
                metadata={"category": "系统"},
            )
            print(f"[scheduler] 系统通知已写入 user={user_id!r}")

    except Exception as e:
        print(f"[scheduler] 定时合并失败: {e}")
    print("[scheduler] 定时记忆合并完成。")


async def _startup_warmup():
    """后端启动时对所有已有用户做记忆预热，消除首条消息的冷启动延迟。"""
    try:
        mem0 = Mem0Client(user_id="")
        users_resp = await mem0._mem.users()
        users = users_resp if isinstance(users_resp, list) else users_resp.get("results", [])
        for user in users:
            user_id = user.get("name", "")
            if not user_id or user_id.isdigit():
                continue
            await _warmup_task(user_id)
        print(f"[startup] 预热完成，共处理 {len(users)} 个用户")
    except Exception as e:
        print(f"[startup] 预热失败（不影响正常启动）: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时预热所有用户的记忆缓存
    import asyncio
    asyncio.create_task(_startup_warmup())

    # 启动时开启调度器
    scheduler.add_job(
        _scheduled_consolidate,
        trigger=CronTrigger(minute="*/10"),  # 每天凌晨 3:00 触发
        id="daily_consolidate",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("[scheduler] APScheduler 已启动，每天 03:00 执行记忆合并。")
    yield
    # 关闭时停止调度器
    scheduler.shutdown(wait=False)
    logger.info("[scheduler] APScheduler 已停止。")


app = FastAPI(title="健康 Agent API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(AuthMiddleware)

app.include_router(chat.router)
app.include_router(memory.router)


@app.get("/health")
async def health_check():
    return {"status": "ok"}
