from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from research_team.config import load_settings
from research_team.evaluation.e2e import (
    invalid_citations,
    parse_citations,
    summarize_by_category,
    summarize_runs,
)
from research_team.graph import build_graph
from research_team.llm import build_llm
from research_team.runtime import load_knowledge
from research_team.tools.search import get_search_provider

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOPICS = ROOT / "eval" / "topics.jsonl"
DEFAULT_OUT = ROOT / "eval"


class UsageTracker:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    def snapshot(self) -> dict:
        return {
            "input": self.input_tokens,
            "output": self.output_tokens,
            "total": self.input_tokens + self.output_tokens,
            "calls": self.calls,
        }


class TrackingLLM:
    def __init__(self, llm, tracker: UsageTracker):
        self.llm = llm
        self.tracker = tracker

    def invoke(self, *args, **kwargs):
        response = self.llm.invoke(*args, **kwargs)
        self.tracker.calls += 1

        usage = getattr(response, "usage_metadata", None)
        if usage:
            self.tracker.input_tokens += int(usage.get("input_tokens") or 0)
            self.tracker.output_tokens += int(usage.get("output_tokens") or 0)
        else:
            metadata = getattr(response, "response_metadata", {}) or {}
            token_usage = metadata.get("token_usage") or {}
            self.tracker.input_tokens += int(token_usage.get("prompt_tokens") or 0)
            self.tracker.output_tokens += int(
                token_usage.get("completion_tokens") or 0
            )
        return response


def load_topics(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_topic(
    item: dict,
    settings,
    llm,
    tracker: UsageTracker,
    search,
    knowledge,
    output_dir: Path,
) -> dict:
    run_id = f"e2e-{item['id']}"
    graph = build_graph(
        llm,
        search,
        output_dir=output_dir,
        run_id=run_id,
        max_steps=settings.max_steps,
        max_revisions=settings.max_revisions,
        knowledge=knowledge,
        kb_top_k=settings.kb_top_k,
        kb_vec_min=settings.kb_vec_min,
        kb_bm25_min=settings.kb_bm25_min,
        kb_vector_weight=settings.kb_vector_weight,
    )
    tracker.reset()
    state = {
        "topic": item["topic"],
        "steps": 0,
        "revisions": 0,
        "research_attempts": 0,
    }
    reviews: list[dict] = []
    start = time.perf_counter()
    error = None
    success = False

    try:
        for update in graph.stream(state, stream_mode="updates"):
            for node, payload in update.items():
                if not payload:
                    continue
                state.update(payload)
                if node == "reviewer" and payload.get("review"):
                    reviews.append(payload["review"])
        success = bool(state.get("final_report"))
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"

    latency = time.perf_counter() - start
    tokens = tracker.snapshot()

    report = ""
    report_path = state.get("final_report")
    if report_path and Path(report_path).exists():
        report = Path(report_path).read_text(encoding="utf-8")
    else:
        success = False

    sources = state.get("sources", [])
    citations = parse_citations(report)
    invalid = invalid_citations(citations, len(sources))
    kb_sources = sum(1 for s in sources if s.get("source_type") == "kb")
    web_sources = sum(1 for s in sources if s.get("source_type") == "web")

    return {
        "id": item["id"],
        "topic": item["topic"],
        "category": item.get("category", "unknown"),
        "success": success,
        "error": error,
        "latency_s": round(latency, 2),
        "tokens": tokens,
        "revisions": state.get("revisions", 0),
        "steps": state.get("steps", 0),
        "review_calls": len(reviews),
        "first_review_passed": reviews[0].get("passed") if reviews else None,
        "final_review_passed": reviews[-1].get("passed") if reviews else None,
        "sources": {"kb": kb_sources, "web": web_sources},
        "citation_count": len(citations),
        "invalid_citation_count": len(invalid),
        "report_path": report_path,
    }


def estimate_cost(tokens: dict) -> float:
    input_price = float(os.getenv("EVAL_INPUT_PRICE_PER_M", "0") or 0)
    output_price = float(os.getenv("EVAL_OUTPUT_PRICE_PER_M", "0") or 0)
    return round(
        tokens["input"] / 1_000_000 * input_price
        + tokens["output"] / 1_000_000 * output_price,
        6,
    )


def render_markdown(result: dict) -> str:
    summary = result["summary"]
    lines = [
        "# 端到端评测结果",
        "",
        f"- 评测时间：{result['run_at']}",
        f"- 模型：`{result['model']}`",
        f"- 主题数：{summary['total']}（成功 {summary['success']}）",
        "",
        "## 汇总",
        "",
        "| 指标 | 数值 |",
        "|---|---|",
        f"| 任务完成率 | {summary['success_rate']:.3f} |",
        f"| 首次审校通过率 | {summary['first_pass_rate']:.3f} |",
        f"| 平均返工次数 | {summary['avg_revisions']:.2f} |",
        f"| 引用有效率 | {1 - summary['invalid_citation_rate']:.3f} |",
        f"| 平均耗时 | {summary['avg_latency_s']:.1f}s |",
        f"| P50 耗时 | {summary['p50_latency_s']:.1f}s |",
        f"| P95 耗时 | {summary['p95_latency_s']:.1f}s |",
        f"| 平均输入 tokens | {summary['avg_input_tokens']:.0f} |",
        f"| 平均输出 tokens | {summary['avg_output_tokens']:.0f} |",
        f"| 平均总 tokens | {summary['avg_total_tokens']:.0f} |",
        "",
        "## 分类统计",
        "",
        "| 分类 | 数量 | 完成率 | 首次通过率 | 平均返工 | 平均耗时 |",
        "|---|---|---|---|---|---|",
    ]
    for category, item in result["by_category"].items():
        lines.append(
            f"| {category} | {item['total']} | {item['success_rate']:.3f} "
            f"| {item['first_pass_rate']:.3f} | {item['avg_revisions']:.2f} "
            f"| {item['avg_latency_s']:.1f}s |"
        )

    lines += [
        "",
        "## 逐条结果",
        "",
        "| ID | 主题 | 分类 | 完成 | 耗时 | 返工 | 首次审校 | 引用 | 来源(KB/Web) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for run in result["runs"]:
        first = (
            "通过"
            if run["first_review_passed"]
            else ("未通过" if run["first_review_passed"] is False else "-")
        )
        lines.append(
            f"| {run['id']} | {run['topic']} | {run['category']} "
            f"| {'✓' if run['success'] else '✗'} | {run['latency_s']:.1f}s "
            f"| {run['revisions']} | {first} "
            f"| {run['citation_count'] - run['invalid_citation_count']}"
            f"/{run['citation_count']} "
            f"| {run['sources']['kb']}/{run['sources']['web']} |"
        )

    lines += [
        "",
        "> 引用有效率 = 合法引用数 / 总引用数；合法指编号落在资料范围内。",
        "> token 统计来自模型返回的 usage_metadata。",
        "",
    ]
    return "\n".join(lines)


def save(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "e2e_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "e2e_results.md").write_text(
        render_markdown(result), encoding="utf-8"
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="端到端研究任务评测")
    parser.add_argument("--topics", type=Path, default=DEFAULT_TOPICS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--ids", help="只跑指定主题 ID，逗号分隔，例如 t04,t13"
    )
    args = parser.parse_args()

    settings = load_settings()
    topics = load_topics(args.topics)
    if args.ids:
        wanted = {item.strip() for item in args.ids.split(",") if item.strip()}
        topics = [item for item in topics if item["id"] in wanted]
    if args.limit:
        topics = topics[: args.limit]

    output_dir = args.out / "e2e_reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    tracker = UsageTracker()
    llm = TrackingLLM(build_llm(settings), tracker)
    search = get_search_provider(settings.search_provider)
    knowledge = load_knowledge(settings)

    result = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "model": settings.model,
        "pricing": {
            "input_per_m": float(os.getenv("EVAL_INPUT_PRICE_PER_M", "0") or 0),
            "output_per_m": float(os.getenv("EVAL_OUTPUT_PRICE_PER_M", "0") or 0),
        },
        "runs": [],
    }

    consecutive_failures = 0
    for index, item in enumerate(topics, start=1):
        print(f"[{index}/{len(topics)}] {item['id']} {item['topic']}", flush=True)
        run = run_topic(
            item, settings, llm, tracker, search, knowledge, output_dir
        )
        run["cost"] = estimate_cost(run["tokens"])
        result["runs"].append(run)
        print(
            f"  success={run['success']} latency={run['latency_s']}s "
            f"revisions={run['revisions']} tokens={run['tokens']['total']}",
            flush=True,
        )

        result["summary"] = summarize_runs(result["runs"])
        result["by_category"] = summarize_by_category(result["runs"])
        save(result, args.out)

        consecutive_failures = 0 if run["success"] else consecutive_failures + 1
        if consecutive_failures >= 3:
            print("连续失败 3 次，停止评测", flush=True)
            break

    print(render_markdown(result))
    print(f"结果已写入 {args.out / 'e2e_results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
