from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
import jwt
import os

SECRET = os.getenv("JWT_SECRET", "change-me")


class AuthMiddleware(BaseHTTPMiddleware):
    """JWT 验证，提取 user_id 注入 request state，隔离多用户记忆"""

    SKIP_PATHS = {"/health", "/docs", "/openapi.json"}

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)

        token = auth.removeprefix("Bearer ").strip()
        try:
            payload = jwt.decode(token, SECRET, algorithms=["HS256"])
            request.state.user_id = payload["sub"]
        except jwt.PyJWTError:
            return JSONResponse({"detail": "Invalid token"}, status_code=401)

        return await call_next(request)
