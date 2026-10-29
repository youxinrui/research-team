from __future__ import annotations

from ..state import ResearchState
from ..tools.search import SearchProvider


def make_researcher(
    search: SearchProvider,
    max_results: int = 5,
    knowledge=None,
    kb_top_k: int = 3,
    kb_vec_min: float = 0.4,
    kb_bm25_min: float = 2.0,
    kb_vector_weight: float = 0.5,
):
    def researcher(state: ResearchState) -> dict:
        attempts = state.get("research_attempts", 0)
        sources: list[dict] = []

        if knowledge is not None:
            try:
                sources = knowledge.search(
                    state["topic"],
                    top_k=kb_top_k,
                    vec_min=kb_vec_min,
                    bm25_min=kb_bm25_min,
                    vector_weight=kb_vector_weight,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[researcher] 知识库检索失败: {exc}")
                sources = []
            if sources:
                print(f"[researcher] 知识库命中 {len(sources)} 条，跳过网络搜索")

        if not sources:
            try:
                sources = search.search(state["topic"], max_results=max_results)
            except Exception as exc:  # noqa: BLE001
                print(f"[researcher] 搜索失败: {exc}")
                sources = []
            for item in sources:
                item.setdefault("source_type", "web")

        return {"sources": sources, "research_attempts": attempts + 1}

    return researcher
