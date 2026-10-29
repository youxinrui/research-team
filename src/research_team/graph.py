from __future__ import annotations

from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from .agents.analyst import make_analyst
from .agents.researcher import make_researcher
from .agents.reviewer import make_reviewer
from .agents.supervisor import make_rule_router, supervisor_node
from .agents.writer import make_writer
from .state import ResearchState
from .tools.search import SearchProvider


def make_finalize(output_dir: Path, run_id: str):
    def finalize(state: ResearchState) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{run_id}.md"
        content = state.get("draft") or "（无内容）"
        if not state.get("sources"):
            content = (
                "> 注意：本次未获取到有效资料，报告内容可能不准确。\n\n" + content
            )
        review = state.get("review")
        if review and not review.get("passed") and review.get("reasons"):
            content += "\n\n## 审校意见\n" + "\n".join(
                f"- {reason}" for reason in review["reasons"]
            )
        path.write_text(content, encoding="utf-8")
        return {"final_report": str(path)}

    return finalize


def build_graph(
    llm: BaseChatModel,
    search: SearchProvider,
    *,
    output_dir: Path,
    run_id: str,
    max_steps: int = 20,
    max_revisions: int = 2,
    checkpointer=None,
    knowledge=None,
    kb_top_k: int = 3,
    kb_vec_min: float = 0.4,
    kb_bm25_min: float = 2.0,
    kb_vector_weight: float = 0.5,
):
    builder = StateGraph(ResearchState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node(
        "researcher",
        make_researcher(
            search,
            knowledge=knowledge,
            kb_top_k=kb_top_k,
            kb_vec_min=kb_vec_min,
            kb_bm25_min=kb_bm25_min,
            kb_vector_weight=kb_vector_weight,
        ),
    )
    builder.add_node("analyst", make_analyst(llm))
    builder.add_node("writer", make_writer(llm))
    builder.add_node("reviewer", make_reviewer(llm))
    builder.add_node("finalize", make_finalize(output_dir, run_id))

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        make_rule_router(max_steps, max_revisions),
        {"researcher": "researcher", "analyst": "analyst", "writer": "writer", "reviewer": "reviewer", "finalize": "finalize"},
    )
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("analyst", "supervisor")
    builder.add_edge("writer", "supervisor")
    builder.add_edge("reviewer", "supervisor")
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)
