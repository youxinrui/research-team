from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .chunker import Chunk, chunk_text, load_documents
from .embedder import Embedder


@dataclass(frozen=True)
class KnowledgeIndex:
    model: str
    dim: int
    chunks: list[Chunk]
    vectors: np.ndarray


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def build_index(
    docs_dir: Path,
    embedder: Embedder,
    index_dir: Path,
    model: str,
    *,
    chunk_size: int = 500,
    overlap: int = 80,
) -> int:
    chunks: list[Chunk] = []
    for source, text in load_documents(docs_dir):
        chunks.extend(chunk_text(text, source, chunk_size=chunk_size, overlap=overlap))

    if not chunks:
        raise ValueError(f"{docs_dir} 下没有找到 .md 或 .txt 文档")

    vectors = np.asarray(embedder.embed([chunk.text for chunk in chunks]), dtype=np.float32)
    vectors = normalize_rows(vectors)

    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "vectors.npy", vectors)
    payload = {
        "model": model,
        "dim": int(vectors.shape[1]),
        "chunks": [asdict(chunk) for chunk in chunks],
    }
    (index_dir / "index.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    return len(chunks)


def load_index(index_dir: Path, expected_model: str | None = None) -> KnowledgeIndex | None:
    index_file = index_dir / "index.json"
    vectors_file = index_dir / "vectors.npy"
    if not index_file.exists() or not vectors_file.exists():
        return None

    payload = json.loads(index_file.read_text(encoding="utf-8"))
    if expected_model and payload.get("model") != expected_model:
        raise ValueError(
            f"索引使用的模型是 {payload.get('model')}，当前配置是 {expected_model}，请重新执行 --ingest"
        )

    return KnowledgeIndex(
        model=payload["model"],
        dim=int(payload["dim"]),
        chunks=[Chunk(**item) for item in payload["chunks"]],
        vectors=np.load(vectors_file),
    )
