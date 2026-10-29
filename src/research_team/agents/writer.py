from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..state import ResearchState
from .common import format_sources, strip_code_fence

SYSTEM_PROMPT = (
    "你是一名严格循证的研究报告撰写者。只能使用给定资料中明确出现的信息。\n"
    "硬性规则：\n"
    "1. 不得引入资料之外的事实、数据、时间、人物职务、机构关系或实现细节。\n"
    "2. 每一条事实性陈述都要在句末用 [编号] 标注来源；没有来源的陈述不要写。\n"
    "3. 资料没有覆盖的内容直接省略；如果必须说明，写成“资料未覆盖”。\n"
    "4. 可以基于资料做归纳和比较，但不得补充资料之外的知识或推断。\n"
    "5. 在资料范围内充分展开：每章至少写成 2 个自然段，把资料中的要点解释清楚，"
    "不要只列短句；但不得为了凑长度添加资料外内容。\n"
    "输出格式：标题、摘要、正文（按大纲展开）、参考资料（编号 + URL）。"
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
