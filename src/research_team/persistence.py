from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver


def make_checkpointer(backend: str = "sqlite", path: Path | None = None):
    backend = backend.lower()
    if backend == "memory":
        return InMemorySaver()
    if backend == "sqlite":
        if path is None:
            raise ValueError("sqlite checkpointer 需要提供 path")
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(path), check_same_thread=False)
        return SqliteSaver(connection)
    raise ValueError(f"未知的 checkpointer 后端: {backend}")
