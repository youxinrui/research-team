from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..state import ResearchState
from .common import format_sources, strip_code_fence

SYSTEM_PROMPT = (
    "你是一名研究分析师。根据给定资料提炼要点并生成报告大纲。\n"
    "要求：至少 3 个章节；每章列出 2-4 个要点；如资料存在矛盾或不足，单独列出。\n"
    "只输出 Markdown 大纲，不要写正文。"
)


def make_analyst(llm: BaseChatModel):
    def analyst(state: ResearchState) -> dict:
        content = (
            f"主题：{state['topic']}\n\n"
            f"资料：\n{format_sources(state.get('sources', []))}"
        )
        response = llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=content),
            ]
        )
        return {"outline": strip_code_fence(str(response.content))}

    return analyst
