from __future__ import annotations

from .config import Settings
from .rag.embedder import Embedder, HashEmbedder, OpenAICompatibleEmbedder
from .rag.index import load_index
from .rag.retriever import KnowledgeRetriever


def make_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "hash":
        return HashEmbedder()
    if settings.embedding_provider == "openai":
        return OpenAICompatibleEmbedder(
            settings.embedding_api_key,
            settings.embedding_base_url,
            settings.embedding_model,
        )
    raise ValueError(f"未知的 embedding provider: {settings.embedding_provider}")


def load_knowledge(settings: Settings):
    try:
        index = load_index(settings.knowledge_index_dir, settings.embedding_model)
    except ValueError as exc:
        print(f"[warn] 知识库不可用: {exc}")
        return None
    if index is None:
        return None

    try:
        return KnowledgeRetriever(index, make_embedder(settings))
    except ValueError as exc:
        print(f"[warn] 知识库索引存在但缺少 embedding 配置: {exc}，将只用网络搜索")
        return None
