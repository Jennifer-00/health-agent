import logging
import os

import jwt
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

SECRET = os.getenv("JWT_SECRET", "change-me")
_SKIP_PATHS = {"/health", "/docs", "/openapi.json", "/auth/login"}


class AuthMiddleware:
    """Pure ASGI auth middleware — avoids BaseHTTPMiddleware streaming buffering."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in _SKIP_PATHS:
            await self.app(scope, receive, send)
            return

        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        auth = headers.get(b"authorization", b"").decode("latin-1")

        if not auth.startswith("Bearer "):
            host = scope["client"][0] if scope.get("client") else "unknown"
            logger.warning("[auth] missing token path=%s ip=%s", path, host)
            await JSONResponse({"detail": "Unauthorized"}, status_code=401)(scope, receive, send)
            return

        token = auth[7:].strip()
        try:
            payload = jwt.decode(token, SECRET, algorithms=["HS256"])
            scope.setdefault("state", {})["user_id"] = payload["sub"]
        except jwt.PyJWTError as exc:
            host = scope["client"][0] if scope.get("client") else "unknown"
            logger.warning("[auth] invalid token path=%s ip=%s: %s", path, host, exc)
            await JSONResponse({"detail": "Invalid token"}, status_code=401)(scope, receive, send)
            return

        await self.app(scope, receive, send)
