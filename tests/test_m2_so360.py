from __future__ import annotations

from research_team.tools.search import parse_so360

SAMPLE_HTML = """
<ul>
  <li class="res-list">
    <h3 class="res-title">
      <a href="/link?m=abc" data-mdurl="https://example.com/one">结果一</a>
    </h3>
    <span class="res-list-summary">第一段摘要</span>
  </li>
  <li class="res-list">
    <h3 class="res-title">
      <a href="https://example.com/two">结果二</a>
    </h3>
    <span class="res-list-summary">第二段摘要</span>
  </li>
  <li class="res-list">
    <div>没有标题的块</div>
  </li>
</ul>
"""


def test_parse_so360_extracts_results():
    results = parse_so360(SAMPLE_HTML, max_results=5)

    assert results == [
        {"title": "结果一", "url": "https://example.com/one", "snippet": "第一段摘要"},
        {"title": "结果二", "url": "https://example.com/two", "snippet": "第二段摘要"},
    ]


def test_parse_so360_respects_max_results():
    results = parse_so360(SAMPLE_HTML, max_results=1)

    assert len(results) == 1
    assert results[0]["title"] == "结果一"
