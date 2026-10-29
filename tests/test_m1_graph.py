from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from research_team.graph import build_graph

FAKE_SOURCES = [
    {"title": "示例资料", "url": "https://example.com/a", "snippet": "摘要 A"},
    {"title": "示例资料 B", "url": "https://example.com/b", "snippet": "摘要 B"},
]


class FakeSearch:
    def __init__(self, results=None, error: Exception | None = None):
        self.results = results if results is not None else FAKE_SOURCES
        self.error = error
        self.calls = 0

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        self.calls += 1
        if self.error:
            raise self.error
        return self.results


def run_graph(tmp_path, *, search, run_id="test-run", max_steps=20):
    llm = FakeListChatModel(
        responses=[
            "## 大纲\n- 第一章\n- 第二章\n- 第三章",
            "# 测试报告\n\n这是正文。",
            '{"passed": true, "reasons": []}',
        ]
    )
    graph = build_graph(
        llm, search, output_dir=tmp_path, run_id=run_id, max_steps=max_steps
    )
    state = {"topic": "测试主题", "steps": 0, "revisions": 0, "research_attempts": 0}
    for update in graph.stream(state, stream_mode="updates"):
        for _, payload in update.items():
            state.update(payload or {})
    return state


def test_end_to_end_writes_report(tmp_path):
    search = FakeSearch()
    run_graph(tmp_path, search=search)

    report = tmp_path / "test-run.md"
    assert report.exists()
    assert "测试报告" in report.read_text(encoding="utf-8")
    assert search.calls == 1


def test_respects_max_steps(tmp_path):
    run_graph(tmp_path, search=FakeSearch(), run_id="guard-run", max_steps=1)

    assert (tmp_path / "guard-run.md").exists()


def test_empty_search_still_finishes(tmp_path):
    search = FakeSearch(results=[], error=RuntimeError("网络不可用"))
    run_graph(tmp_path, search=search, run_id="empty-run")

    report = (tmp_path / "empty-run.md").read_text(encoding="utf-8")
    assert "未获取到有效资料" in report
    assert search.calls == 1
