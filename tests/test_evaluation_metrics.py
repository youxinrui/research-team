from __future__ import annotations

from research_team.evaluation.metrics import mrr, recall_at_k, rrf_ranking


def test_recall_at_k_counts_expected_in_top_k():
    ranking = [3, 1, 4, 2]

    assert recall_at_k(ranking, {1}, 2) == 1.0
    assert recall_at_k(ranking, {2}, 2) == 0.0
    assert recall_at_k(ranking, {2}, 4) == 1.0


def test_mrr_uses_first_relevant_rank():
    ranking = [5, 3, 1]

    assert mrr(ranking, {5}) == 1.0
    assert mrr(ranking, {3}) == 0.5
    assert mrr(ranking, {1}) == 1 / 3
    assert mrr(ranking, {9}) == 0.0


def test_rrf_prefers_items_ranked_high_by_both():
    vector_ranking = [0, 1, 2]
    bm25_ranking = [0, 2, 1]

    assert rrf_ranking(vector_ranking, bm25_ranking)[0] == 0


def test_rrf_weight_shifts_result():
    assert rrf_ranking([0, 1], [1, 0], vector_weight=0.9)[0] == 0
    assert rrf_ranking([0, 1], [1, 0], vector_weight=0.1)[0] == 1
