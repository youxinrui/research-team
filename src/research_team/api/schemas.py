from __future__ import annotations

from pydantic import BaseModel, Field


class StartRequest(BaseModel):
    topic: str = Field(min_length=1, description="研究主题")


class StartResponse(BaseModel):
    session_id: str
    status: str


class SessionResponse(BaseModel):
    session_id: str
    status: str
    topic: str | None = None
    report_path: str | None = None
    error: str | None = None
    events: list[str] = Field(default_factory=list)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class AskResponse(BaseModel):
    answer: str


class EditRequest(BaseModel):
    instruction: str = Field(min_length=1)


class EditResponse(BaseModel):
    message: str
    content: str


class ReportResponse(BaseModel):
    session_id: str
    content: str
    report_path: str


class SessionListResponse(BaseModel):
    sessions: list[str]


class MessageItem(BaseModel):
    role: str
    content: str
    created_at: str = ""


class MessagesResponse(BaseModel):
    session_id: str
    messages: list[MessageItem] = Field(default_factory=list)
