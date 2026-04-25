import asyncio
import json
import logging

from fastapi import APIRouter, Request
from starlette.responses import StreamingResponse

from memory.pubsub import subscribe_alerts

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/notify", tags=["notify"])


@router.get("/stream")
async def notification_stream(request: Request):
    """SSE 长连接，实时接收健康预警广播。"""
    user_id: str = request.state.user_id

    async def event_generator():
        try:
            async for alert in subscribe_alerts(user_id):
                if await request.is_disconnected():
                    break
                yield f"data: {json.dumps(alert, ensure_ascii=False)}\n\n"
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("[notify] stream error user=%r: %s", user_id, exc)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
