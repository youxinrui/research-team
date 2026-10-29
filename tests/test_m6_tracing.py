from __future__ import annotations

import json

from research_team.tracing import Tracer, summarize


def test_summarize_string_truncates():
    result = summarize("a" * 200, limit=10)

    assert result == "a" * 10 + "..."


def test_summarize_collections():
    assert summarize([1, 2, 3]) == "list[3]"
    assert summarize({"b": 1, "a": 2}) == "dict(a, b)"


def test_tracer_writes_jsonl(tmp_path):
    log_path = tmp_path / "run.jsonl"
    tracer = Tracer(log_path)

    tracer.event("start", "run_id=abc")
    tracer.record("writer", {"draft": "hello", "revisions": 1})

    lines = log_path.read_text(encoding="utf-8").splitlines()
    entries = [json.loads(line) for line in lines]

    assert entries[0]["event"] == "start"
    assert entries[1]["node"] == "writer"
    assert entries[1]["updates"] == {"draft": "hello", "revisions": "1"}
