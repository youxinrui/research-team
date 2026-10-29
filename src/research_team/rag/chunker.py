from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    index: int


def _split_long(text: str, chunk_size: int, overlap: int) -> list[str]:
    pieces = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(text), step):
        piece = text[start : start + chunk_size].strip()
        if piece:
            pieces.append(piece)
        if start + chunk_size >= len(text):
            break
    return pieces


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 500,
    overlap: int = 80,
) -> list[Chunk]:
    paragraphs: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) > chunk_size:
            paragraphs.extend(_split_long(paragraph, chunk_size, overlap))
        else:
            paragraphs.append(paragraph)

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if not current:
            current = paragraph
        elif len(current) + len(paragraph) + 2 <= chunk_size:
            current = f"{current}\n\n{paragraph}"
        else:
            chunks.append(current)
            tail_len = max(0, min(overlap, chunk_size - len(paragraph) - 2))
            tail = current[-tail_len:] if tail_len else ""
            current = f"{tail}\n\n{paragraph}".strip() if tail else paragraph
    if current:
        chunks.append(current)

    return [
        Chunk(id=f"{source}#chunk-{index}", text=text, source=source, index=index)
        for index, text in enumerate(chunks)
    ]


def load_documents(root: Path) -> list[tuple[str, str]]:
    root = Path(root)
    documents: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue
        documents.append((path.relative_to(root).as_posix(), path.read_text(encoding="utf-8")))
    return documents
