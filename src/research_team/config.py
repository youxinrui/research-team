from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model: str
    output_dir: Path
    max_steps: int
    max_revisions: int
    search_provider: str
    checkpoint_backend: str
    checkpoint_path: Path
    chat_db_path: Path
    embedding_api_key: str
    embedding_base_url: str
    embedding_model: str
    embedding_provider: str
    knowledge_index_dir: Path
    kb_top_k: int
    kb_vec_min: float
    kb_bm25_min: float
    kb_vector_weight: float


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    values = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "").strip(),
        "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL", "").strip(),
        "MODEL_NAME": os.getenv("MODEL_NAME", "").strip(),
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ValueError(
            f"缺少环境变量: {', '.join(missing)}。请复制 .env.example 为 .env 并填写。"
        )

    output_dir = Path(os.getenv("OUTPUT_DIR", "output"))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    checkpoint_path = Path(os.getenv("CHECKPOINT_PATH", "output/checkpoints.sqlite"))
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    chat_db_path = Path(os.getenv("CHAT_DB_PATH", "output/chat.db"))
    if not chat_db_path.is_absolute():
        chat_db_path = PROJECT_ROOT / chat_db_path

    knowledge_index_dir = Path(os.getenv("KNOWLEDGE_INDEX_DIR", "knowledge/.index"))
    if not knowledge_index_dir.is_absolute():
        knowledge_index_dir = PROJECT_ROOT / knowledge_index_dir

    return Settings(
        api_key=values["OPENAI_API_KEY"],
        base_url=values["OPENAI_BASE_URL"],
        model=values["MODEL_NAME"],
        output_dir=output_dir,
        max_steps=int(os.getenv("MAX_STEPS", "20")),
        max_revisions=int(os.getenv("MAX_REVISIONS", "2")),
        search_provider=os.getenv("SEARCH_PROVIDER", "so360"),
        checkpoint_backend=os.getenv("CHECKPOINT_BACKEND", "sqlite"),
        checkpoint_path=checkpoint_path,
        chat_db_path=chat_db_path,
        embedding_api_key=os.getenv("EMBEDDING_API_KEY", "").strip(),
        embedding_base_url=os.getenv(
            "EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1"
        ).strip(),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3").strip(),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower(),
        knowledge_index_dir=knowledge_index_dir,
        kb_top_k=int(os.getenv("KB_TOP_K", "3")),
        kb_vec_min=float(os.getenv("KB_VEC_MIN", "0.55")),
        kb_bm25_min=float(os.getenv("KB_BM25_MIN", "5.0")),
        kb_vector_weight=float(os.getenv("KB_VECTOR_WEIGHT", "0.1")),
    )
