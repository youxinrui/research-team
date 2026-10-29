from __future__ import annotations

import re

import numpy as np

CITATION_RE = re.compile(r"\[(\d+)\]")


def parse_citations(text: str) -> list[int]:
    return [int(match) for match in CITATION_RE.findall(text)]


def invalid_citations(citations: list[int], source_count: int) -> list[int]:
    return [value for value in citations if value < 1 or value > source_count]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_runs(runs: list[dict]) -> dict:
    total = len(runs)
    succeeded = [run for run in runs if run.get("success")]
    reviewed = [
        run for run in succeeded if run.get("first_review_passed") is not None
    ]
    latencies = [run.get("latency_s", 0.0) for run in succeeded]
    citation_total = sum(run.get("citation_count", 0) for run in succeeded)
    invalid_total = sum(
        run.get("invalid_citation_count", 0) for run in succeeded
    )

    return {
        "total": total,
        "success": len(succeeded),
        "success_rate": len(succeeded) / total if total else 0.0,
        "first_pass_reviewed": len(reviewed),
        "first_pass_rate": (
            sum(1 for run in reviewed if run["first_review_passed"])
            / len(reviewed)
            if reviewed
            else 0.0
        ),
        "avg_revisions": mean([run.get("revisions", 0) for run in succeeded]),
        "avg_latency_s": mean(latencies),
        "p50_latency_s": percentile(latencies, 50),
        "p95_latency_s": percentile(latencies, 95),
        "avg_input_tokens": mean(
            [run.get("tokens", {}).get("input", 0) for run in succeeded]
        ),
        "avg_output_tokens": mean(
            [run.get("tokens", {}).get("output", 0) for run in succeeded]
        ),
        "avg_total_tokens": mean(
            [run.get("tokens", {}).get("total", 0) for run in succeeded]
        ),
        "citation_count": citation_total,
        "invalid_citation_count": invalid_total,
        "invalid_citation_rate": (
            invalid_total / citation_total if citation_total else 0.0
        ),
    }


def summarize_by_category(runs: list[dict]) -> dict[str, dict]:
    groups: dict[str, list[dict]] = {}
    for run in runs:
        groups.setdefault(run.get("category", "unknown"), []).append(run)
    return {category: summarize_runs(items) for category, items in groups.items()}
