import json
import logging
import os
from collections.abc import AsyncGenerator

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


async def publish_alert(user_id: str, alert: dict) -> None:
    r = aioredis.from_url(_REDIS_URL, decode_responses=True)
    try:
        channel = f"alert:{user_id}"
        await r.publish(channel, json.dumps(alert, ensure_ascii=False))
        logger.info("[pubsub] published to %s", channel)
    except Exception as exc:
        logger.error("[pubsub] publish failed user=%r: %s", user_id, exc)
    finally:
        await r.aclose()


async def subscribe_alerts(user_id: str) -> AsyncGenerator[dict, None]:
    r = aioredis.from_url(_REDIS_URL, decode_responses=True)
    try:
        async with r.pubsub() as pubsub:
            await pubsub.subscribe(f"alert:{user_id}")
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        yield json.loads(message["data"])
                    except Exception:
                        pass
    except Exception as exc:
        logger.error("[pubsub] subscribe error user=%r: %s", user_id, exc)
    finally:
        await r.aclose()
