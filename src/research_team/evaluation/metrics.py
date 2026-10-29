from __future__ import annotations

import numpy as np


def recall_at_k(ranking: list[int], expected: set[int], k: int) -> float:
    return 1.0 if set(ranking[:k]) & expected else 0.0


def mrr(ranking: list[int], expected: set[int]) -> float:
    for rank, item in enumerate(ranking, start=1):
        if item in expected:
            return 1.0 / rank
    return 0.0


def rrf_ranking(
    vector_ranking: list[int],
    bm25_ranking: list[int],
    vector_weight: float = 0.5,
    k: int = 60,
) -> list[int]:
    size = len(vector_ranking)
    scores = np.zeros(size, dtype=np.float64)

    for rank, index in enumerate(vector_ranking):
        scores[index] += vector_weight / (k + rank + 1)
    for rank, index in enumerate(bm25_ranking):
        scores[index] += (1.0 - vector_weight) / (k + rank + 1)

    return list(np.argsort(-scores))
