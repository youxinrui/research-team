from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from research_team.config import load_settings
from research_team.evaluation.metrics import mrr, recall_at_k, rrf_ranking
from research_team.rag.index import load_index
from research_team.rag.retriever import tokenize
from research_team.runtime import make_embedder

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "eval" / "retrieval.jsonl"
DEFAULT_OUT = ROOT / "eval"

VEC_MIN_GRID = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
BM25_MIN_GRID = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0]
RRF_WEIGHT_GRID = [round(0.1 * i, 1) for i in range(1, 10)]


def load_dataset(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def rank_vector(index, query_vector: np.ndarray):
    cosines = index.vectors @ query_vector
    return list(np.argsort(-cosines)), cosines


def rank_bm25(bm25: BM25Okapi, query: str):
    scores = np.asarray(bm25.get_scores(tokenize(query)), dtype=np.float32)
    return list(np.argsort(-scores)), scores


def evaluate(rows: list[dict], index, embedder) -> dict:
    settings = load_settings()
    chunk_ids = [chunk.id for chunk in index.chunks]
    chunk_index = {chunk_id: i for i, chunk_id in enumerate(chunk_ids)}
    bm25 = BM25Okapi([tokenize(chunk.text) for chunk in index.chunks])

    query_vectors = embedder.embed([row["query"] for row in rows])

    methods = ("vector", "bm25", "hybrid")
    per_query = []
    for row, raw_vector in zip(rows, query_vectors):
        expected = {chunk_index[cid] for cid in row["expected"] if cid in chunk_index}
        query_vector = np.asarray(raw_vector, dtype=np.float32)
        norm = np.linalg.norm(query_vector)
        if norm:
            query_vector = query_vector / norm

        vector_rank, cosines = rank_vector(index, query_vector)
        bm25_rank, bm25_scores = rank_bm25(bm25, row["query"])
        hybrid_rank = rrf_ranking(
            vector_rank, bm25_rank, vector_weight=settings.kb_vector_weight
        )

        per_query.append(
            {
                "row": row,
                "expected": expected,
                "cosines": cosines,
                "bm25_scores": bm25_scores,
                "rankings": {
                    "vector": vector_rank,
                    "bm25": bm25_rank,
                    "hybrid": hybrid_rank,
                },
            }
        )

    positive = [item for item in per_query if item["row"]["type"] == "positive"]
    negative = [item for item in per_query if item["row"]["type"] == "negative"]

    retrieval = {}
    for method in methods:
        retrieval[method] = {
            "recall@3": mean(
                recall_at_k(item["rankings"][method], item["expected"], 3)
                for item in positive
            ),
            "recall@5": mean(
                recall_at_k(item["rankings"][method], item["expected"], 5)
                for item in positive
            ),
            "mrr": mean(mrr(item["rankings"][method], item["expected"]) for item in positive),
        }

    best_weight = {"vector_weight": None, "recall@3": -1.0, "mrr": -1.0}
    for weight in RRF_WEIGHT_GRID:
        recall = mean(
            recall_at_k(
                rrf_ranking(
                    item["rankings"]["vector"],
                    item["rankings"]["bm25"],
                    vector_weight=weight,
                ),
                item["expected"],
                3,
            )
            for item in positive
        )
        reciprocal = mean(
            mrr(
                rrf_ranking(
                    item["rankings"]["vector"],
                    item["rankings"]["bm25"],
                    vector_weight=weight,
                ),
                item["expected"],
            )
            for item in positive
        )
        if (reciprocal, recall) > (best_weight["mrr"], best_weight["recall@3"]):
            best_weight = {
                "vector_weight": weight,
                "recall@3": recall,
                "mrr": reciprocal,
            }

    overlap = {"both": [], "vector_only": [], "bm25_only": [], "neither": []}
    for item in positive:
        vector_hit = recall_at_k(
            item["rankings"]["vector"], item["expected"], 3
        ) == 1.0
        bm25_hit = recall_at_k(
            item["rankings"]["bm25"], item["expected"], 3
        ) == 1.0
        if vector_hit and bm25_hit:
            overlap["both"].append(item["row"]["id"])
        elif vector_hit:
            overlap["vector_only"].append(item["row"]["id"])
        elif bm25_hit:
            overlap["bm25_only"].append(item["row"]["id"])
        else:
            overlap["neither"].append(item["row"]["id"])

    threshold_grid = []
    top_k = settings.kb_top_k
    vector_weight = settings.kb_vector_weight
    for vec_min in VEC_MIN_GRID:
        for bm25_min in BM25_MIN_GRID:
            positive_success = mean(
                any_expected_returned(
                    item, vec_min, bm25_min, top_k, vector_weight
                )
                for item in positive
            )
            negative_false = mean(
                bool(
                    returned_indices(
                        item, vec_min, bm25_min, top_k, vector_weight
                    )
                )
                for item in negative
            )
            threshold_grid.append(
                {
                    "vec_min": vec_min,
                    "bm25_min": bm25_min,
                    "positive_success": positive_success,
                    "negative_false_positive": negative_false,
                    "score": positive_success - negative_false,
                }
            )

    best_threshold = max(
        threshold_grid,
        key=lambda item: (
            item["score"],
            item["positive_success"],
            -item["negative_false_positive"],
            item["vec_min"],
            item["bm25_min"],
        ),
    )

    current = next(
        (
            item
            for item in threshold_grid
            if abs(item["vec_min"] - settings.kb_vec_min) < 1e-9
            and abs(item["bm25_min"] - settings.kb_bm25_min) < 1e-9
        ),
        None,
    )

    diagnostics = {
        "current": threshold_diagnostics(
            positive,
            negative,
            chunk_ids,
            settings.kb_vec_min,
            settings.kb_bm25_min,
            top_k,
            vector_weight,
        ),
        "best": threshold_diagnostics(
            positive,
            negative,
            chunk_ids,
            best_threshold["vec_min"],
            best_threshold["bm25_min"],
            top_k,
            vector_weight,
        ),
    }

    return {
        "dataset": {
            "positive": len(positive),
            "negative": len(negative),
            "chunks": len(index.chunks),
            "embedding_model": index.model,
            "top_k": top_k,
        },
        "retrieval": retrieval,
        "overlap": overlap,
        "best_rrf": best_weight,
        "threshold_grid": threshold_grid,
        "current_thresholds": current,
        "best_thresholds": best_threshold,
        "diagnostics": diagnostics,
    }


def returned_indices(
    item: dict,
    vec_min: float,
    bm25_min: float,
    top_k: int,
    vector_weight: float,
) -> list[int]:
    passed = {
        i
        for i in range(len(item["cosines"]))
        if item["cosines"][i] >= vec_min or item["bm25_scores"][i] >= bm25_min
    }
    if not passed:
        return []
    ranked = rrf_ranking(
        item["rankings"]["vector"],
        item["rankings"]["bm25"],
        vector_weight=vector_weight,
    )
    return [i for i in ranked if i in passed][:top_k]


def any_expected_returned(
    item: dict,
    vec_min: float,
    bm25_min: float,
    top_k: int,
    vector_weight: float,
) -> bool:
    returned = returned_indices(
        item, vec_min, bm25_min, top_k, vector_weight
    )
    return bool(set(returned) & item["expected"])


def threshold_diagnostics(
    positive: list[dict],
    negative: list[dict],
    chunk_ids: list[str],
    vec_min: float,
    bm25_min: float,
    top_k: int,
    vector_weight: float,
) -> list[dict]:
    details = []
    for item in positive + negative:
        returned = returned_indices(
            item, vec_min, bm25_min, top_k, vector_weight
        )
        expected = item["expected"]
        if expected:
            success = bool(set(returned) & expected)
        else:
            success = not returned
        details.append(
            {
                "id": item["row"]["id"],
                "query": item["row"]["query"],
                "type": item["row"]["type"],
                "expected": [chunk_ids[i] for i in sorted(expected)],
                "returned": [chunk_ids[i] for i in returned],
                "success": success,
            }
        )
    return details


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def render_markdown(result: dict) -> str:
    lines = [
        "# 检索评测结果",
        "",
        f"- 正例：{result['dataset']['positive']} 条",
        f"- 反例（应回退网络）：{result['dataset']['negative']} 条",
        f"- 知识库片段：{result['dataset']['chunks']} 个",
        f"- Embedding 模型：`{result['dataset']['embedding_model']}`",
        "",
        "## 三种检索方式对比（正例）",
        "",
        "| 方式 | Recall@3 | Recall@5 | MRR |",
        "|---|---|---|---|",
    ]
    for method, metrics in result["retrieval"].items():
        lines.append(
            f"| {method} | {metrics['recall@3']:.3f} | {metrics['recall@5']:.3f} | {metrics['mrr']:.3f} |"
        )

    best = result["best_rrf"]
    overlap = result["overlap"]
    lines += [
        "",
        "## 互补性分析（Recall@3）",
        "",
        f"- 两者都命中：{len(overlap['both'])} 条",
        f"- 仅向量命中：{len(overlap['vector_only'])} 条"
        f"（{', '.join(overlap['vector_only']) or '无'}）",
        f"- 仅 BM25 命中：{len(overlap['bm25_only'])} 条"
        f"（{', '.join(overlap['bm25_only']) or '无'}）",
        f"- 两者都未命中：{len(overlap['neither'])} 条"
        f"（{', '.join(overlap['neither']) or '无'}）",
        "",
        "## RRF 权重网格搜索",
        "",
        f"- 最优向量权重：**{best['vector_weight']}**"
        f"（BM25 权重 {round(1 - best['vector_weight'], 1)}）",
        f"- 对应 Recall@3：{best['recall@3']:.3f}，MRR：{best['mrr']:.3f}",
        "",
        "## 命中阈值",
        "",
    ]
    current = result["current_thresholds"]
    best_threshold = result["best_thresholds"]
    if current:
        lines += [
            "| 配置 | vec_min | bm25_min | 正例命中率 | 反例误命中率 |",
            "|---|---|---|---|---|",
            f"| 当前 | {current['vec_min']} | {current['bm25_min']} "
            f"| {current['positive_success']:.3f} | {current['negative_false_positive']:.3f} |",
        ]
    lines.append(
        f"| 最优 | {best_threshold['vec_min']} | {best_threshold['bm25_min']} "
        f"| {best_threshold['positive_success']:.3f} | {best_threshold['negative_false_positive']:.3f} |"
    )
    lines += [
        "",
        "> 说明：正例命中率 = 返回结果中包含标注片段的比例；",
        "> 反例误命中率 = 无相关知识时仍返回片段的比例，越低越好。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="运行知识库检索评测")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    settings = load_settings()
    index = load_index(settings.knowledge_index_dir, settings.embedding_model)
    if index is None:
        print("知识库索引不存在，请先执行 python -m research_team --ingest knowledge")
        return 1

    rows = load_dataset(args.dataset)
    result = evaluate(rows, index, make_embedder(settings))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out / "results.md").write_text(render_markdown(result), encoding="utf-8")

    print(render_markdown(result))
    print(f"结果已写入 {args.out / 'results.md'} 和 results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
