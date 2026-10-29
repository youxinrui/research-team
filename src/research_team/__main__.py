from __future__ import annotations

import argparse
import sys
import uuid
from datetime import datetime
from pathlib import Path

from .config import load_settings
from .graph import build_graph
from .llm import build_llm
from .persistence import make_checkpointer
from .rag.index import build_index
from .runtime import load_knowledge, make_embedder
from .tracing import Tracer
from .tools.search import get_search_provider


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="research_team")
    parser.add_argument("topic", nargs="?", help="研究主题（新建任务时必填）")
    parser.add_argument("--resume", metavar="RUN_ID", help="从中断的 run 继续")
    parser.add_argument(
        "--ingest", metavar="DIR", help="把指定目录下的 .md/.txt 文档写入知识库索引"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args(argv)
    try:
        settings = load_settings()
    except ValueError as exc:
        print(f"配置错误: {exc}")
        return 1

    if args.ingest:
        try:
            count = build_index(
                Path(args.ingest),
                make_embedder(settings),
                settings.knowledge_index_dir,
                settings.embedding_model,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"入库失败: {type(exc).__name__}: {exc}")
            return 1
        print(f"索引完成: {count} 个片段 -> {settings.knowledge_index_dir}")
        return 0

    knowledge = load_knowledge(settings)

    run_id = args.resume or (
        f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    )
    checkpointer = make_checkpointer(
        settings.checkpoint_backend, settings.checkpoint_path
    )
    graph = build_graph(
        build_llm(settings),
        get_search_provider(settings.search_provider),
        output_dir=settings.output_dir,
        run_id=run_id,
        max_steps=settings.max_steps,
        max_revisions=settings.max_revisions,
        checkpointer=checkpointer,
        knowledge=knowledge,
        kb_top_k=settings.kb_top_k,
        kb_vec_min=settings.kb_vec_min,
        kb_bm25_min=settings.kb_bm25_min,
        kb_vector_weight=settings.kb_vector_weight,
    )
    config = {"configurable": {"thread_id": run_id}}
    tracer = Tracer(settings.output_dir / "logs" / f"{run_id}.jsonl")

    if args.resume:
        snapshot = graph.get_state(config)
        if not snapshot.values:
            print(f"找不到 run: {run_id}")
            return 1
        if not snapshot.next:
            print(f"任务已完成: {snapshot.values.get('final_report')}")
            return 0
        print(f"恢复 run: {run_id}")
        print(f"主题: {snapshot.values.get('topic', '')}")
        stream = graph.stream(None, config, stream_mode="updates")
    else:
        if not args.topic:
            print('用法: python -m research_team "研究主题"')
            print("      python -m research_team --resume <run_id>")
            return 2
        print(f"run_id: {run_id}")
        print(f"主题: {args.topic}")
        initial = {
            "topic": args.topic,
            "steps": 0,
            "revisions": 0,
            "research_attempts": 0,
        }
        stream = graph.stream(initial, config, stream_mode="updates")

    final_report = ""
    tracer.event("start", f"run_id={run_id}")
    try:
        for update in stream:
            for node, payload in update.items():
                payload = payload or {}
                tracer.record(node, payload)
                if "final_report" in payload:
                    final_report = payload["final_report"]
    except KeyboardInterrupt:
        print(f"\n已中断。恢复命令: python -m research_team --resume {run_id}")
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"运行失败: {type(exc).__name__}: {exc}")
        print(f"可从最近检查点继续: python -m research_team --resume {run_id}")
        return 1

    tracer.event("done", f"report={final_report}")
    print(f"报告已生成: {final_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
