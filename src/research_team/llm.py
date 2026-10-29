from __future__ import annotations

from langchain_openai import ChatOpenAI

from .config import Settings


def build_llm(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=0.3,
        timeout=120,
        max_retries=2,
    )
