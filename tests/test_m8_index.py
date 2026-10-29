from __future__ import annotations

from research_team.rag.index import build_index, load_index


class FakeEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0] if "LangGraph" in text else [0.0, 1.0]
            for text in texts
        ]


def test_build_and_load_index(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("LangGraph 使用 StateGraph 编排节点。", encoding="utf-8")
    (docs / "b.txt").write_text("RAG 需要向量检索和 BM25 混合。", encoding="utf-8")

    index_dir = tmp_path / "index"
    count = build_index(docs, FakeEmbedder(), index_dir, "fake-model")

    assert count == 2
    assert (index_dir / "index.json").exists()
    assert (index_dir / "vectors.npy").exists()

    index = load_index(index_dir, "fake-model")
    assert index is not None
    assert index.model == "fake-model"
    assert index.dim == 2
    assert len(index.chunks) == 2


def test_load_index_returns_none_when_missing(tmp_path):
    assert load_index(tmp_path / "missing") is None


def test_load_index_rejects_model_mismatch(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("内容", encoding="utf-8")
    index_dir = tmp_path / "index"
    build_index(docs, FakeEmbedder(), index_dir, "model-a")

    try:
        load_index(index_dir, "model-b")
    except ValueError as exc:
        assert "重新执行 --ingest" in str(exc)
    else:
        raise AssertionError("expected ValueError for model mismatch")
