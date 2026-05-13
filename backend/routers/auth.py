import logging
import os

import jwt
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])

SECRET = os.getenv("JWT_SECRET", "change-me")


class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    token: str
    user_id: str


@router.post("/login", response_model=LoginResponse)
async def login(body: LoginRequest):
    demo_email = os.getenv("DEMO_EMAIL", "demo@health.ai")
    demo_password = os.getenv("DEMO_PASSWORD", "demo123")

    if body.email != demo_email or body.password != demo_password:
        logger.warning("[auth] login failed email=%r", body.email)
        raise HTTPException(status_code=401, detail="邮箱或密码错误")

    token = jwt.encode({"sub": body.email}, SECRET, algorithm="HS256")
    logger.info("[auth] login ok user=%r", body.email)
    return LoginResponse(token=token, user_id=body.email)
