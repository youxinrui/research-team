from __future__ import annotations

from research_team.agents.researcher import make_researcher


class FakeKnowledge:
    def __init__(self, results=None, error: Exception | None = None):
        self.results = results or []
        self.error = error
        self.queries: list[str] = []

    def search(self, query: str, top_k: int = 3, **kwargs) -> list[dict]:
        self.queries.append(query)
        if self.error:
            raise self.error
        return self.results


class FakeWeb:
    def __init__(self):
        self.calls = 0

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        self.calls += 1
        return [{"title": "网页结果", "url": "https://example.com", "snippet": "摘要"}]


def test_knowledge_hit_skips_web_search():
    knowledge = FakeKnowledge(
        results=[{"title": "本地片段", "url": "doc.md#chunk-0", "snippet": "内容", "source_type": "kb"}]
    )
    web = FakeWeb()
    researcher = make_researcher(web, knowledge=knowledge)

    update = researcher({"topic": "测试主题", "research_attempts": 0})

    assert knowledge.queries == ["测试主题"]
    assert update["sources"][0]["source_type"] == "kb"
    assert web.calls == 0


def test_knowledge_miss_falls_back_to_web():
    knowledge = FakeKnowledge(results=[])
    web = FakeWeb()
    researcher = make_researcher(web, knowledge=knowledge)

    update = researcher({"topic": "测试主题", "research_attempts": 0})

    assert web.calls == 1
    assert update["sources"][0]["source_type"] == "web"


def test_knowledge_error_falls_back_to_web():
    knowledge = FakeKnowledge(error=RuntimeError("索引损坏"))
    web = FakeWeb()
    researcher = make_researcher(web, knowledge=knowledge)

    update = researcher({"topic": "测试主题", "research_attempts": 0})

    assert web.calls == 1
    assert update["sources"]
