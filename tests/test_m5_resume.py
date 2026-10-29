from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.checkpoint.memory import InMemorySaver

from research_team.graph import build_graph

OUTLINE = "## 大纲\n- 第一章\n- 第二章\n- 第三章"
PASS = '{"passed": true, "reasons": []}'
CONFIG = {"configurable": {"thread_id": "test-thread"}}


class FakeSearch:
    def __init__(self):
        self.calls = 0

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        self.calls += 1
        return [{"title": "资料", "url": "https://example.com", "snippet": "摘要"}]


class CrashOnceLLM:
    """Delegates to an inner model, but raises on the first call to simulate Ctrl+C."""

    def __init__(self, inner):
        self.inner = inner
        self.crashed = False

    def invoke(self, *args, **kwargs):
        if not self.crashed:
            self.crashed = True
            raise RuntimeError("simulated crash")
        return self.inner.invoke(*args, **kwargs)


def initial_state():
    return {"topic": "测试主题", "steps": 0, "revisions": 0, "research_attempts": 0}


def test_resume_continues_without_rerunning_research(tmp_path):
    search = FakeSearch()
    llm = CrashOnceLLM(
        FakeListChatModel(responses=[OUTLINE, "# 报告正文", PASS])
    )
    graph = build_graph(
        llm,
        search,
        output_dir=tmp_path,
        run_id="resume-run",
        checkpointer=InMemorySaver(),
    )

    try:
        list(graph.stream(initial_state(), CONFIG, stream_mode="updates"))
    except RuntimeError as exc:
        assert "simulated crash" in str(exc)
    else:
        raise AssertionError("expected the simulated crash")

    snapshot = graph.get_state(CONFIG)
    assert snapshot.values["sources"]
    assert snapshot.next

    list(graph.stream(None, CONFIG, stream_mode="updates"))

    assert (tmp_path / "resume-run.md").exists()
    assert search.calls == 1


def test_resume_completed_run_does_nothing(tmp_path):
    llm = FakeListChatModel(responses=[OUTLINE, "# 报告正文", PASS])
    graph = build_graph(
        llm,
        FakeSearch(),
        output_dir=tmp_path,
        run_id="done-run",
        checkpointer=InMemorySaver(),
    )
    list(graph.stream(initial_state(), CONFIG, stream_mode="updates"))

    snapshot = graph.get_state(CONFIG)
    assert not snapshot.next

    updates = list(graph.stream(None, CONFIG, stream_mode="updates"))
    assert updates == []
