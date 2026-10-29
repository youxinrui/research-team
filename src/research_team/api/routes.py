from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from .schemas import (
    AskRequest,
    AskResponse,
    EditRequest,
    EditResponse,
    MessageItem,
    MessagesResponse,
    ReportResponse,
    SessionListResponse,
    SessionResponse,
    StartRequest,
    StartResponse,
)
from .sessions import SessionBusy, SessionManager, SessionNotFound

router = APIRouter(prefix="/api", tags=["research"])


def manager_of(request: Request) -> SessionManager:
    return request.app.state.sessions


def require_ready(request: Request, session_id: str):
    manager = manager_of(request)
    try:
        return manager.require_ready(session_id)
    except SessionNotFound:
        raise HTTPException(status_code=404, detail="会话不存在")
    except SessionBusy as exc:
        raise HTTPException(status_code=409, detail=f"会话尚未完成: {exc}")


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/sessions", response_model=StartResponse, status_code=202)
def start_session(payload: StartRequest, request: Request) -> StartResponse:
    session_id = manager_of(request).start(payload.topic)
    return StartResponse(session_id=session_id, status="running")


@router.get("/sessions", response_model=SessionListResponse)
def list_sessions(request: Request) -> SessionListResponse:
    return SessionListResponse(sessions=manager_of(request).saved_sessions())


@router.get("/sessions/{session_id}", response_model=SessionResponse)
def get_session(session_id: str, request: Request) -> SessionResponse:
    record = manager_of(request).get(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="会话不存在")

    chat = record.chat
    return SessionResponse(
        session_id=session_id,
        status=record.status,
        topic=chat.state.get("topic"),
        report_path=chat.report_path,
        error=record.error,
        events=record.events[-80:],
    )


@router.get("/sessions/{session_id}/messages", response_model=MessagesResponse)
def get_messages(session_id: str, request: Request) -> MessagesResponse:
    manager = manager_of(request)
    record = manager.get(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="会话不存在")

    rows = manager.messages(session_id, record)
    return MessagesResponse(
        session_id=session_id,
        messages=[MessageItem(**row) for row in rows],
    )


@router.get("/sessions/{session_id}/report", response_model=ReportResponse)
def get_report(session_id: str, request: Request) -> ReportResponse:
    record = require_ready(request, session_id)
    path = Path(record.chat.report_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="报告文件不存在")

    return ReportResponse(
        session_id=session_id,
        content=path.read_text(encoding="utf-8"),
        report_path=str(path),
    )


@router.post("/sessions/{session_id}/ask", response_model=AskResponse)
def ask(session_id: str, payload: AskRequest, request: Request) -> AskResponse:
    record = require_ready(request, session_id)
    with record.lock:
        answer = record.chat.answer_question(payload.question)
    return AskResponse(answer=answer)


@router.post("/sessions/{session_id}/edit", response_model=EditResponse)
def edit(session_id: str, payload: EditRequest, request: Request) -> EditResponse:
    record = require_ready(request, session_id)
    with record.lock:
        message = record.chat.revise_report(payload.instruction)

    path = Path(record.chat.report_path)
    content = path.read_text(encoding="utf-8") if path.exists() else ""
    return EditResponse(message=message, content=content)
