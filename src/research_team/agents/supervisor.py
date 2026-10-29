from __future__ import annotations

from ..state import ResearchState


def supervisor_node(state: ResearchState) -> dict:
    return {"steps": state.get("steps", 0) + 1}


def make_rule_router(max_steps: int, max_revisions: int):
    def rule_router(state: ResearchState) -> str:
        if state.get("steps", 0) >= max_steps:
            return "finalize"
        if not state.get("sources") and state.get("research_attempts", 0) < 1:
            return "researcher"
        if not state.get("outline"):
            return "analyst"
        if not state.get("draft"):
            return "writer"
        if state.get("review") is None:
            return "reviewer"
        review = state["review"]
        if not review.get("passed") and state.get("revisions", 0) < max_revisions:
            return "writer"
        return "finalize"

    return rule_router
