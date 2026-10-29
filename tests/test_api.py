from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi.testclient import TestClient

from research_team.api.app import create_app
from research_team.api.sessions import SessionManager


@dataclass
class FakeSettings:
    output_dir: Path
    chat_db_path: Path


class FakeChat:
    def __init__(self, events, output_dir):
        self.events = events
        self.output_dir = Path(output_dir)
        self.state = {}
        self.report_path = None
        self.run_id = None
        self.history = []

    def start_research(self, topic, run_id=None):
        self.run_id = run_id
        self.state = {"topic": topic, "draft": "# 报告", "sources": []}
        path = self.output_dir / f"{run_id}.md"
        path.write_text("# 报告", encoding="utf-8")
        self.report_path = str(path)
        self.events.append("[done]")
        return "ok"

    def answer_question(self, question):
        return f"回答: {question}"

    def revise_report(self, instruction):
        content = f"# 修改后的报告\n\n{instruction}"
        Path(self.report_path).write_text(content, encoding="utf-8")
        return "报告已更新"

    def load_session(self, session_id):
        return "已载入"


def make_client(tmp_path):
    settings = FakeSettings(tmp_path, tmp_path / "chat.db")
    manager = SessionManager(
        settings,
        background=False,
        chat_factory=lambda events: FakeChat(events, tmp_path),
    )
    return TestClient(create_app(settings, manager=manager))


def test_health(tmp_path):
    client = make_client(tmp_path)

    assert client.get("/api/health").json() == {"status": "ok"}


def test_root_returns_service_info(tmp_path):
    client = make_client(tmp_path)

    body = client.get("/").json()

    assert body["docs"] == "/docs"
    assert body["health"] == "/api/health"


def test_full_session_flow(tmp_path):
    client = make_client(tmp_path)

    start = client.post("/api/sessions", json={"topic": "测试主题"})
    assert start.status_code == 202
    session_id = start.json()["session_id"]

    status = client.get(f"/api/sessions/{session_id}").json()
    assert status["status"] == "ready"
    assert status["topic"] == "测试主题"

    report = client.get(f"/api/sessions/{session_id}/report").json()
    assert report["content"] == "# 报告"

    messages = client.get(f"/api/sessions/{session_id}/messages").json()
    assert messages["messages"] == []

    answer = client.post(
        f"/api/sessions/{session_id}/ask", json={"question": "讲了什么"}
    ).json()
    assert answer["answer"] == "回答: 讲了什么"

    edited = client.post(
        f"/api/sessions/{session_id}/edit", json={"instruction": "改短一点"}
    ).json()
    assert "修改后的报告" in edited["content"]
    assert edited["message"] == "报告已更新"


def test_unknown_session_returns_404(tmp_path):
    client = make_client(tmp_path)

    assert client.get("/api/sessions/nope").status_code == 404
    assert (
        client.post("/api/sessions/nope/ask", json={"question": "hi"}).status_code
        == 404
    )
