from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from research_team.chat import ResearchChat, is_modification
from research_team.storage import ChatStore

OUTLINE = "## 大纲\n- 第一章\n- 第二章\n- 第三章"
PASS = '{"passed": true, "reasons": []}'


@dataclass
class FakeSettings:
    output_dir: Path
    checkpoint_backend: str = "memory"
    checkpoint_path: Path = Path("unused.sqlite")
    max_steps: int = 20
    max_revisions: int = 2
    kb_top_k: int = 3
    kb_vec_min: float = 0.4
    kb_bm25_min: float = 2.0
    kb_vector_weight: float = 0.1


class FakeSearch:
    def __init__(self):
        self.calls = 0

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        self.calls += 1
        return [{"title": "资料", "url": "https://example.com", "snippet": "摘要"}]


def make_chat(tmp_path, responses):
    return ResearchChat(
        FakeSettings(tmp_path),
        FakeListChatModel(responses=responses),
        FakeSearch(),
        store=ChatStore(tmp_path / "chat.db"),
    )


def preset(chat, tmp_path, draft="# 旧报告"):
    chat.run_id = "test-run"
    report = tmp_path / "report.md"
    report.write_text(draft, encoding="utf-8")
    chat.report_path = str(report)
    chat.state = {"topic": "测试主题", "sources": [], "draft": draft}
    chat.store.create_session("test-run", "测试主题", status="ready")
    return report


def test_is_modification():
    assert is_modification("请把报告精简一下") is True
    assert is_modification("补充一节关于成本的讨论") is True
    assert is_modification("报告里讲了什么？") is False


def test_answer_question_appends_history(tmp_path):
    chat = make_chat(tmp_path, ["这是回答"])
    preset(chat, tmp_path)

    reply = chat.ask("报告的核心结论是什么")

    assert reply == "这是回答"
    assert len(chat.history) == 2
    assert len(chat.store.get_messages("test-run")) == 2


def test_revise_report_updates_file(tmp_path):
    chat = make_chat(tmp_path, ["# 修订后的报告", PASS])
    report = preset(chat, tmp_path)

    reply = chat.ask("请把报告精简一下")

    assert "报告已更新" in reply
    assert "修订后的报告" in report.read_text(encoding="utf-8")
    assert chat.state["draft"] == "# 修订后的报告"


def test_ask_requires_a_report(tmp_path):
    chat = make_chat(tmp_path, ["unused"])

    reply = chat.ask("/ask 这个项目是什么")

    assert "还没有报告" in reply


def test_load_session_then_ask(tmp_path):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    report = tmp_path / "existing.md"
    report.write_text("# 已存在的报告", encoding="utf-8")
    (sessions / "run1.json").write_text(
        json.dumps(
            {
                "run_id": "run1",
                "topic": "已有主题",
                "report": str(report),
                "sources": [],
                "history": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    chat = make_chat(tmp_path, ["这是基于已有报告的回答"])

    loaded = chat.load_session("run1")
    assert "已载入会话" in loaded
    assert chat.state["draft"] == "# 已存在的报告"

    answer = chat.ask("/ask 报告讲了什么")
    assert answer == "这是基于已有报告的回答"


def test_start_research_creates_report(tmp_path):
    chat = make_chat(tmp_path, [OUTLINE, "# 报告正文", PASS])

    reply = chat.start_research("测试主题")

    assert chat.run_id
    assert Path(chat.report_path).exists()
    assert "报告已完成" in reply
    assert len(chat.history) == 2
    session = chat.store.get_session(chat.run_id)
    assert session["status"] == "ready"
    assert len(chat.store.get_messages(chat.run_id)) == 2
