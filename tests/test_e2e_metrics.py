from __future__ import annotations

from research_team.evaluation.e2e import (
    invalid_citations,
    parse_citations,
    percentile,
    summarize_runs,
)


def test_parse_citations():
    assert parse_citations("结论一[1]，结论二[2][3]。") == [1, 2, 3]
    assert parse_citations("没有引用") == []


def test_invalid_citations():
    assert invalid_citations([1, 2, 3], 3) == []
    assert invalid_citations([0, 1, 4], 3) == [0, 4]


def test_percentile_handles_empty():
    assert percentile([], 50) == 0.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5


def test_summarize_runs():
    runs = [
        {
            "success": True,
            "latency_s": 10.0,
            "revisions": 1,
            "first_review_passed": False,
            "citation_count": 4,
            "invalid_citation_count": 1,
            "tokens": {"input": 100, "output": 50, "total": 150},
        },
        {
            "success": True,
            "latency_s": 20.0,
            "revisions": 0,
            "first_review_passed": True,
            "citation_count": 2,
            "invalid_citation_count": 0,
            "tokens": {"input": 200, "output": 100, "total": 300},
        },
        {"success": False, "latency_s": 5.0},
    ]

    summary = summarize_runs(runs)

    assert summary["total"] == 3
    assert summary["success"] == 2
    assert summary["success_rate"] == 2 / 3
    assert summary["first_pass_rate"] == 0.5
    assert summary["avg_revisions"] == 0.5
    assert summary["avg_latency_s"] == 15.0
    assert summary["invalid_citation_rate"] == 1 / 6
    assert summary["avg_total_tokens"] == 225
