from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..chat import ResearchChat
from ..config import Settings
from ..storage import ChatStore


class SessionNotFound(Exception):
    pass


class SessionBusy(Exception):
    pass


@dataclass
class SessionRecord:
    chat: object
    status: str = "running"
    error: str | None = None
    events: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


class SessionManager:
    def __init__(
        self,
        settings: Settings,
        *,
        background: bool = True,
        chat_factory=None,
        store: ChatStore | None = None,
    ):
        self.settings = settings
        self.background = background
        self.store = store or ChatStore(settings.chat_db_path)
        self.chat_factory = chat_factory or (
            lambda events: ResearchChat.from_settings(
                settings, event_callback=events.append
            )
        )
        self._sessions: dict[str, SessionRecord] = {}
        self._lock = threading.Lock()

    def start(self, topic: str) -> str:
        session_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        events: list[str] = []
        chat = self.chat_factory(events)
        record = SessionRecord(chat=chat, events=events)

        with self._lock:
            self._sessions[session_id] = record

        def run() -> None:
            try:
                chat.start_research(topic, run_id=session_id)
                record.status = "ready"
            except Exception as exc:  # noqa: BLE001
                record.status = "failed"
                record.error = f"{type(exc).__name__}: {exc}"

        if self.background:
            threading.Thread(target=run, daemon=True).start()
        else:
            run()
        return session_id

    def get(self, session_id: str) -> SessionRecord | None:
        with self._lock:
            record = self._sessions.get(session_id)
        if record is not None:
            return record
        return self._load_from_disk(session_id)

    def _load_from_disk(self, session_id: str) -> SessionRecord | None:
        events: list[str] = []
        try:
            chat = self.chat_factory(events)
            message = chat.load_session(session_id)
        except Exception:  # noqa: BLE001
            return None
        if not getattr(chat, "report_path", None):
            return None

        record = SessionRecord(chat=chat, status="ready", events=[message])
        with self._lock:
            self._sessions[session_id] = record
        return record

    def require_ready(self, session_id: str) -> SessionRecord:
        record = self.get(session_id)
        if record is None:
            raise SessionNotFound(session_id)
        if record.status != "ready":
            raise SessionBusy(record.status)
        return record

    def saved_sessions(self) -> list[str]:
        ids = set(self.store.list_session_ids())
        directory = self.settings.output_dir / "sessions"
        if directory.exists():
            ids.update(path.stem for path in directory.glob("*.json"))
        return sorted(ids, reverse=True)

    def messages(
        self, session_id: str, record: SessionRecord | None = None
    ) -> list[dict]:
        rows = self.store.get_messages(session_id)
        if rows:
            return rows
        if record is not None:
            return [
                {
                    "role": item["role"],
                    "content": item["content"],
                    "created_at": "",
                }
                for item in getattr(record.chat, "history", [])
            ]
        return []
