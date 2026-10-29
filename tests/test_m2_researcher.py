from __future__ import annotations

from research_team.agents.researcher import make_researcher


class FakeSearch:
    def __init__(self, results=None, error: Exception | None = None):
        self.results = results or []
        self.error = error
        self.queries: list[str] = []

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        self.queries.append(query)
        if self.error:
            raise self.error
        return self.results


def test_researcher_populates_sources():
    results = [{"title": "T", "url": "https://example.com", "snippet": "S"}]
    search = FakeSearch(results=results)
    researcher = make_researcher(search)

    update = researcher({"topic": "测试主题", "research_attempts": 0})

    assert update["sources"] == results
    assert update["research_attempts"] == 1
    assert search.queries == ["测试主题"]


def test_researcher_handles_search_error():
    search = FakeSearch(error=RuntimeError("boom"))
    researcher = make_researcher(search)

    update = researcher({"topic": "测试主题", "research_attempts": 0})

    assert update["sources"] == []
    assert update["research_attempts"] == 1
