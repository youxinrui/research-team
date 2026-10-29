from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from research_team.agents.analyst import make_analyst
from research_team.agents.common import format_sources, strip_code_fence


def test_format_sources_numbers_and_includes_urls():
    text = format_sources(
        [
            {"title": "资料一", "url": "https://a.example", "snippet": "摘要一"},
            {"title": "资料二", "url": "https://b.example", "snippet": "摘要二"},
        ]
    )

    assert "[1] [网络] 资料一 (https://a.example)" in text
    assert "[2] [网络] 资料二 (https://b.example)" in text


def test_format_sources_handles_empty():
    assert format_sources([]) == "（暂无资料）"


def test_strip_code_fence_removes_wrapper():
    assert strip_code_fence("```markdown\n# 标题\n正文\n```") == "# 标题\n正文"


def test_strip_code_fence_keeps_plain_text():
    assert strip_code_fence("正文内容") == "正文内容"


def test_analyst_produces_outline():
    llm = FakeListChatModel(responses=["## 大纲\n- 要点一\n- 要点二\n- 要点三"])
    analyst = make_analyst(llm)

    update = analyst(
        {
            "topic": "测试主题",
            "sources": [{"title": "资料", "url": "https://a.example", "snippet": "摘要"}],
        }
    )

    assert "要点一" in update["outline"]
