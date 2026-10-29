from __future__ import annotations

import json

from research_team.storage import ChatStore


def test_create_session_and_messages(tmp_path):
    store = ChatStore(tmp_path / "chat.db")

    store.create_session("run1", "测试主题")
    store.add_message("run1", "user", "问题")
    store.add_message("run1", "assistant", "回答")

    session = store.get_session("run1")
    assert session["topic"] == "测试主题"
    assert session["status"] == "running"

    messages = store.get_messages("run1")
    assert [item["role"] for item in messages] == ["user", "assistant"]
    assert [item["content"] for item in messages] == ["问题", "回答"]
    assert store.list_session_ids() == ["run1"]


def test_update_session_status_path_and_sources(tmp_path):
    store = ChatStore(tmp_path / "chat.db")
    store.create_session("run1", "测试主题")

    store.update_session(
        "run1",
        status="ready",
        report_path=str(tmp_path / "report.md"),
        sources=[{"url": "https://example.com"}],
    )

    session = store.get_session("run1")
    assert session["status"] == "ready"
    assert session["report_path"].endswith("report.md")
    assert json.loads(session["sources_json"]) == [{"url": "https://example.com"}]


def test_data_persists_across_instances(tmp_path):
    path = tmp_path / "chat.db"
    ChatStore(path).create_session("run1", "测试主题")

    reopened = ChatStore(path)
    assert reopened.get_session("run1")["topic"] == "测试主题"
