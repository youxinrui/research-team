from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from .embedder import Embedder
from .index import KnowledgeIndex

STOPWORDS = {
    "的", "了", "和", "是", "在", "与", "及", "或", "对", "为", "中", "上", "下",
    "有", "也", "就", "都", "而", "等", "把", "被", "让", "从", "到", "一个",
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "is", "are",
    "with", "how", "what", "why", "this", "that",
}


def tokenize(text: str) -> list[str]:
    import jieba

    words = jieba.lcut(text.lower())
    return [
        word
        for word in words
        if word.strip()
        and word not in STOPWORDS
        and re.fullmatch(r"\w+", word) is not None
    ]


class KnowledgeRetriever:
    def __init__(self, index: KnowledgeIndex, embedder: Embedder):
        self.index = index
        self.embedder = embedder
        self.bm25 = BM25Okapi([tokenize(chunk.text) for chunk in index.chunks])

    def search(
        self,
        query: str,
        top_k: int = 3,
        *,
        vec_min: float = 0.4,
        bm25_min: float = 2.0,
        vector_weight: float = 0.5,
    ) -> list[dict]:
        if not self.index.chunks:
            return []

        query_vector = np.asarray(self.embedder.embed([query])[0], dtype=np.float32)
        if query_vector.shape[0] != self.index.dim:
            raise ValueError(
                f"查询向量维度 {query_vector.shape[0]} 与索引维度 {self.index.dim} 不一致"
            )
        norm = np.linalg.norm(query_vector)
        if norm:
            query_vector = query_vector / norm

        cosines = self.index.vectors @ query_vector
        bm25_scores = np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float32)

        candidates = [
            i
            for i in range(len(self.index.chunks))
            if cosines[i] >= vec_min or bm25_scores[i] >= bm25_min
        ]
        if not candidates:
            return []

        rrf = np.zeros(len(self.index.chunks), dtype=np.float32)
        for ranking, weight in (
            (np.argsort(-cosines), vector_weight),
            (np.argsort(-bm25_scores), 1.0 - vector_weight),
        ):
            for rank, idx in enumerate(ranking):
                rrf[idx] += weight / (60 + rank + 1)

        candidates.sort(key=lambda i: rrf[i], reverse=True)
        results = []
        for i in candidates[:top_k]:
            chunk = self.index.chunks[i]
            results.append(
                {
                    "title": f"知识库片段：{chunk.source}",
                    "url": chunk.id,
                    "snippet": chunk.text,
                    "source_type": "kb",
                    "score": float(rrf[i]),
                }
            )
        return results
