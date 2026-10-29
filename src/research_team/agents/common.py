from __future__ import annotations


SNIPPET_LIMIT = 300


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def format_sources(sources: list[dict]) -> str:
    if not sources:
        return "（暂无资料）"

    blocks = []
    for index, source in enumerate(sources, start=1):
        snippet = source.get("snippet", "")
        if len(snippet) > SNIPPET_LIMIT:
            snippet = snippet[:SNIPPET_LIMIT] + "..."
        label = "知识库" if source.get("source_type") == "kb" else "网络"
        blocks.append(
            f"[{index}] [{label}] {source.get('title', '')} ({source.get('url', '')})\n{snippet}"
        )
    return "\n\n".join(blocks)
