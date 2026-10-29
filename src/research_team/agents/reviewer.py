from __future__ import annotations

import json
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..state import ResearchState
from .common import format_sources

SYSTEM_PROMPT = (
    "你是一名严格的研究报告审校员。对照资料检查草稿：引用是否与资料一致、"
    "是否存在资料中没有依据的断言、结构是否完整。\n"
    '只输出 JSON，格式为 {"passed": true 或 false, "reasons": ["问题1", "问题2"]}。'
    "通过时 reasons 为空数组；不通过时列出具体问题。不要输出 JSON 以外的内容。"
)


def parse_review(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if match is None:
        return {"passed": False, "reasons": ["审校结果无法解析为 JSON"]}

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"passed": False, "reasons": ["审校结果不是合法 JSON"]}

    reasons = data.get("reasons") or []
    return {
        "passed": bool(data.get("passed")),
        "reasons": [str(reason) for reason in reasons],
    }


def make_reviewer(llm: BaseChatModel):
    def reviewer(state: ResearchState) -> dict:
        content = (
            f"主题：{state['topic']}\n\n"
            f"资料：\n{format_sources(state.get('sources', []))}\n\n"
            f"报告草稿：\n{state.get('draft', '')}"
        )
        response = llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=content),
            ]
        )
        return {"review": parse_review(str(response.content))}

    return reviewer
