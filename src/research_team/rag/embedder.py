from __future__ import annotations

import hashlib
from typing import Protocol

import httpx


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Dependency-free lexical vectorizer (character n-grams hashed into a fixed dim).

    Useful as an offline fallback; not a semantic model.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        normalized = text.lower()
        grams = [normalized[i : i + 2] for i in range(max(1, len(normalized) - 1))]
        grams += [normalized[i : i + 3] for i in range(max(1, len(normalized) - 2))]
        for gram in grams:
            if not gram.strip():
                continue
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            vector[int.from_bytes(digest, "big") % self.dim] += 1.0
        return vector


class OpenAICompatibleEmbedder:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        *,
        batch_size: int = 16,
        timeout: float = 60.0,
    ):
        if not api_key:
            raise ValueError("缺少 EMBEDDING_API_KEY，无法调用 embedding 接口")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        with httpx.Client(timeout=self.timeout) as client:
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                response = client.post(
                    f"{self.base_url}/embeddings",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model, "input": batch},
                )
                response.raise_for_status()
                data = response.json()["data"]
                data.sort(key=lambda item: item.get("index", 0))
                vectors.extend(item["embedding"] for item in data)
        return vectors
