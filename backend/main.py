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

from backend.routers import chat, memory
from backend.middleware.auth import AuthMiddleware
from memory.mem0_client import Mem0Client
from backend.routers.chat import _warmup_task

logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()


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

    scheduler.start()
    yield
    scheduler.shutdown(wait=False)


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
