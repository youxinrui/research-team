"""Shared state passed between agents.

All fields must stay JSON-serializable so checkpointing works across processes.
"""

from __future__ import annotations

from typing import TypedDict


class ReviewResult(TypedDict, total=False):
    passed: bool
    reasons: list[str]


class ResearchState(TypedDict, total=False):
    topic: str
    sources: list[dict]
    outline: str
    draft: str
    review: ReviewResult | None
    final_report: str
    steps: int
    revisions: int
    research_attempts: int
