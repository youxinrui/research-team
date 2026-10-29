from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from research_team.agents.reviewer import parse_review
from research_team.graph import build_graph

OUTLINE = "## 大纲\n- 第一章\n- 第二章\n- 第三章"
PASS = '{"passed": true, "reasons": []}'
FAIL = '{"passed": false, "reasons": ["引用不足", "结构不完整"]}'


class FakeSearch:
    def search(self, query: str, max_results: int = 5) -> list[dict]:
        return [{"title": "资料", "url": "https://example.com", "snippet": "摘要"}]


def run_pipeline(tmp_path, responses, *, max_revisions=2):
    llm = FakeListChatModel(responses=responses)
    graph = build_graph(
        llm,
        FakeSearch(),
        output_dir=tmp_path,
        run_id="m4-run",
        max_revisions=max_revisions,
    )
    state = {"topic": "测试主题", "steps": 0, "revisions": 0, "research_attempts": 0}
    for update in graph.stream(state, stream_mode="updates"):
        for _, payload in update.items():
            state.update(payload or {})
    return state


def test_parse_review_passes():
    assert parse_review('{"passed": true, "reasons": []}') == {
        "passed": True,
        "reasons": [],
    }


def test_parse_review_fails_with_reasons():
    assert parse_review(FAIL) == {
        "passed": False,
        "reasons": ["引用不足", "结构不完整"],
    }


def test_parse_review_handles_invalid_output():
    result = parse_review("这不是 JSON")

    assert result["passed"] is False
    assert result["reasons"]


def test_revision_loop_rewrites_once(tmp_path):
    state = run_pipeline(
        tmp_path,
        [OUTLINE, "# 初稿", FAIL, "# 修订稿", PASS],
    )

    assert state["revisions"] == 1
    assert state["review"]["passed"] is True
    assert "修订稿" in (tmp_path / "m4-run.md").read_text(encoding="utf-8")


def test_revision_loop_stops_at_max_revisions(tmp_path):
    # analyst + 3 writer calls + 3 reviewer calls
    state = run_pipeline(
        tmp_path,
        [OUTLINE, "# 初稿", FAIL, "# 修订一", FAIL, "# 修订二", FAIL],
        max_revisions=2,
    )

    assert state["revisions"] == 2
    assert state["review"]["passed"] is False
    report = (tmp_path / "m4-run.md").read_text(encoding="utf-8")
    assert "审校意见" in report
    assert "引用不足" in report
