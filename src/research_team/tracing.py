from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

SUMMARY_LIMIT = 120


def summarize(value, limit: int = SUMMARY_LIMIT) -> str:
    if isinstance(value, str):
        text = value.replace("\n", " ").strip()
        return text[:limit] + ("..." if len(text) > limit else "")
    if isinstance(value, list):
        return f"list[{len(value)}]"
    if isinstance(value, dict):
        return f"dict({', '.join(sorted(value))})"
    return repr(value)


class Tracer:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, entry: dict) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def event(self, kind: str, message: str) -> None:
        self._write(
            {
                "time": datetime.now().isoformat(timespec="seconds"),
                "event": kind,
                "message": message,
            }
        )
        print(f"[{kind}] {message}")

    def record(self, node: str, update: dict) -> None:
        summaries = {key: summarize(value) for key, value in update.items()}
        self._write(
            {
                "time": datetime.now().isoformat(timespec="seconds"),
                "node": node,
                "updates": summaries,
            }
        )
        detail = ", ".join(f"{key}: {value}" for key, value in summaries.items())
        print(f"[{node}] {detail or '无字段更新'}")
