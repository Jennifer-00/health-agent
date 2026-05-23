from pydantic import BaseModel
from typing import Literal, Optional
from datetime import date


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class MemoryPatchRequest(BaseModel):
    action: Literal["delete", "update"]
    content: Optional[str] = None


class MemoryItem(BaseModel):
    """统一格式的记忆条目，供前端展示用。"""
    id: str
    content: str
    category: Optional[str] = None
    record_date: Optional[str] = None
    source: Literal["mem0", "db"]


class MemoryImportRequest(BaseModel):
    text: str
