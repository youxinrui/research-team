from __future__ import annotations

import numpy as np

from research_team.rag.chunker import Chunk
from research_team.rag.index import KnowledgeIndex, normalize_rows
from research_team.rag.retriever import KnowledgeRetriever


class FakeEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            if "断点续跑" in text or "checkpoint" in text.lower():
                vectors.append([1.0, 0.0])
            elif "天气" in text:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.0, 0.0])
        return vectors


class ZeroEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0] for _ in texts]


class FixedEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def make_index() -> KnowledgeIndex:
    chunks = [
        Chunk(
            id="langgraph.md#chunk-0",
            text="LangGraph 的断点续跑依赖 checkpoint 保存状态。",
            source="langgraph.md",
            index=0,
        ),
        Chunk(
            id="weather.md#chunk-0",
            text="今天天气晴朗，适合出行。",
            source="weather.md",
            index=0,
        ),
        Chunk(
            id="cooking.md#chunk-0",
            text="番茄炒蛋需要先炒鸡蛋再放番茄。",
            source="cooking.md",
            index=0,
        ),
        Chunk(
            id="sports.md#chunk-0",
            text="足球比赛每队上场十一名球员。",
            source="sports.md",
            index=0,
        ),
    ]
    vectors = normalize_rows(
        np.array(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
            dtype=np.float32,
        )
    )
    return KnowledgeIndex(model="fake", dim=2, chunks=chunks, vectors=vectors)


def test_vector_retrieval_hits_relevant_chunk():
    retriever = KnowledgeRetriever(make_index(), FakeEmbedder())

    hits = retriever.search("断点续跑怎么实现", top_k=2)

    assert hits
    assert hits[0]["url"] == "langgraph.md#chunk-0"
    assert hits[0]["source_type"] == "kb"


def test_bm25_only_retrieval_hits_without_vector_signal():
    retriever = KnowledgeRetriever(make_index(), ZeroEmbedder())

    hits = retriever.search("断点续跑怎么实现", top_k=2, bm25_min=0.1)

    assert hits
    assert hits[0]["url"] == "langgraph.md#chunk-0"


def test_returns_empty_when_nothing_matches():
    retriever = KnowledgeRetriever(make_index(), FakeEmbedder())

    hits = retriever.search("量子计算的退相干时间", top_k=2)

    assert hits == []


def test_vector_weight_shifts_ranking():
    chunks = [
        Chunk(id="a.md#chunk-0", text="完全无关的内容甲", source="a.md", index=0),
        Chunk(id="b.md#chunk-0", text="断点续跑机制说明", source="b.md", index=0),
        Chunk(id="c.md#chunk-0", text="无关内容乙", source="c.md", index=0),
        Chunk(id="d.md#chunk-0", text="无关内容丙", source="d.md", index=0),
    ]
    vectors = normalize_rows(
        np.array(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            dtype=np.float32,
        )
    )
    index = KnowledgeIndex(model="fake", dim=2, chunks=chunks, vectors=vectors)
    retriever = KnowledgeRetriever(index, FixedEmbedder())

    vector_heavy = retriever.search(
        "断点续跑", top_k=2, vec_min=0.0, bm25_min=0.0, vector_weight=0.9
    )
    bm25_heavy = retriever.search(
        "断点续跑", top_k=2, vec_min=0.0, bm25_min=0.0, vector_weight=0.1
    )

    assert vector_heavy[0]["url"] == "a.md#chunk-0"
    assert bm25_heavy[0]["url"] == "b.md#chunk-0"
