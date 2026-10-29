from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..state import ResearchState
from .common import format_sources, strip_code_fence

SYSTEM_PROMPT = (
    "你是一名严格循证的研究分析师。只能依据给定资料提炼要点并生成报告大纲。\n"
    "要求：至少 3 个章节；每章列出 2-4 个要点，并在要点后标注资料编号；"
    "如果资料不足以支撑某个章节，不要强行生成，改在“资料缺口”下列出。\n"
    "只输出 Markdown 大纲，不要写正文，不要补充资料之外的知识。"
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
