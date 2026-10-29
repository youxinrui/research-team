from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..state import ResearchState
from .common import format_sources, strip_code_fence

SYSTEM_PROMPT = (
    "你是一名研究报告撰写者。根据给定主题、大纲和资料撰写完整的中文 Markdown 报告。\n"
    "要求：包含标题、摘要、正文、参考资料；正文按大纲展开；"
    "引用资料时用 [编号] 标注；参考资料列出编号与 URL。\n"
    "只输出报告内容，不要输出额外解释。"
)


def make_writer(llm: BaseChatModel):
    def writer(state: ResearchState) -> dict:
        parts = [f"研究主题：{state['topic']}"]
        if state.get("outline"):
            parts.append(f"报告大纲：\n{state['outline']}")
        if state.get("sources"):
            parts.append(f"可用资料：\n{format_sources(state['sources'])}")
        if state.get("review"):
            reasons = state["review"].get("reasons", [])
            parts.append("上一版未通过审校，问题如下，请修订：\n" + "\n".join(f"- {r}" for r in reasons))
        parts.append("请撰写完整报告。")

        response = llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content="\n\n".join(parts)),
            ]
        )

        update = {"draft": strip_code_fence(str(response.content))}
        if state.get("review") is not None:
            update["review"] = None
            update["revisions"] = state.get("revisions", 0) + 1
        return update

    return writer
