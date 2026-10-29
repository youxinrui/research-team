from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .agents.common import format_sources
from .agents.reviewer import make_reviewer
from .agents.writer import make_writer
from .config import Settings
from .graph import build_graph
from .llm import build_llm
from .persistence import make_checkpointer
from .runtime import load_knowledge
from .storage import ChatStore
from .tools.search import SearchProvider, get_search_provider

MODIFY_KEYWORDS = (
    "修改", "改成", "改为", "补充", "删掉", "删除", "精简", "扩写",
    "重写", "加入报告", "更新报告", "缩短", "延长", "调整",
)

CHAT_SYSTEM_PROMPT = (
    "你是研究报告团队的对话助手。基于当前报告和资料回答用户的问题。\n"
    "规则：只使用当前报告与资料中的信息；资料没有的内容要明确说明；"
    "回答简洁，必要时用 [编号] 引用资料。"
)


def is_modification(text: str) -> bool:
    stripped = text.strip()
    if not stripped or stripped.endswith(("？", "?")):
        return False
    return any(keyword in stripped for keyword in MODIFY_KEYWORDS)


class ResearchChat:
    def __init__(
        self,
        settings: Settings,
        llm: BaseChatModel,
        search: SearchProvider,
        knowledge=None,
        checkpointer=None,
        event_callback=None,
        store: ChatStore | None = None,
    ):
        self.settings = settings
        self.llm = llm
        self.search = search
        self.knowledge = knowledge
        self.event_callback = event_callback
        self.store = store
        self.checkpointer = checkpointer or make_checkpointer(
            settings.checkpoint_backend, settings.checkpoint_path
        )
        self.history: list[dict] = []
        self.state: dict = {}
        self.run_id: str | None = None
        self.report_path: str | None = None

    @classmethod
    def from_settings(cls, settings: Settings, event_callback=None) -> "ResearchChat":
        return cls(
            settings,
            build_llm(settings),
            get_search_provider(settings.search_provider),
            load_knowledge(settings),
            event_callback=event_callback,
            store=ChatStore(settings.chat_db_path),
        )

    def ask(self, user_input: str) -> str:
        text = user_input.strip()
        if not text:
            return ""
        if text.startswith("/new"):
            topic = text[4:].strip()
            if not topic:
                return "用法: /new <研究主题>"
            return self.start_research(topic)
        if text.startswith("/ask"):
            question = text[4:].strip()
            if not question:
                return "用法: /ask <问题>"
            if not self.state.get("draft"):
                return "还没有报告。请先输入研究主题，或用 /load <run_id> 载入已有报告。"
            return self.answer_question(question)
        if text.startswith("/load"):
            run_id = text[5:].strip()
            if not run_id:
                return "用法: /load <run_id>"
            return self.load_session(run_id)
        if text.startswith("/edit"):
            instruction = text[5:].strip()
            if not instruction:
                return "用法: /edit <修改要求>"
            return self.revise_report(instruction)
        if self.run_id is None:
            return self.start_research(text)
        if is_modification(text):
            return self.revise_report(text)
        return self.answer_question(text)

    def start_research(self, topic: str, run_id: str | None = None) -> str:
        run_id = run_id or f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        graph = build_graph(
            self.llm,
            self.search,
            output_dir=self.settings.output_dir,
            run_id=run_id,
            max_steps=self.settings.max_steps,
            max_revisions=self.settings.max_revisions,
            checkpointer=self.checkpointer,
            knowledge=self.knowledge,
            kb_top_k=self.settings.kb_top_k,
            kb_vec_min=self.settings.kb_vec_min,
            kb_bm25_min=self.settings.kb_bm25_min,
            kb_vector_weight=self.settings.kb_vector_weight,
        )
        config = {"configurable": {"thread_id": run_id}}
        self.run_id = run_id
        if self.store:
            self.store.create_session(run_id, topic)
        initial = {
            "topic": topic,
            "steps": 0,
            "revisions": 0,
            "research_attempts": 0,
        }

        self._emit(f"\n[新任务] run_id={run_id}")
        self._emit(f"[主题] {topic}")
        try:
            for update in graph.stream(initial, config, stream_mode="updates"):
                for node, payload in update.items():
                    keys = ", ".join(sorted(payload or {})) or "无字段更新"
                    self._emit(f"[{node}] {keys}")
        except Exception:
            if self.store:
                self.store.update_session(run_id, status="failed")
            raise

        self.state = dict(graph.get_state(config).values)
        self.report_path = self.state.get("final_report")

        reply = (
            f"报告已完成：{self.report_path}\n"
            "可以继续追问内容，或直接告诉我要怎么改（例如“补充一节关于成本的讨论”）。"
        )
        self.history.append({"role": "user", "content": f"研究主题：{topic}"})
        self.history.append({"role": "assistant", "content": reply})
        if self.store:
            self.store.update_session(
                run_id,
                status="ready",
                report_path=self.report_path,
                sources=self.state.get("sources", []),
            )
            self.store.add_message(run_id, "user", f"研究主题：{topic}")
            self.store.add_message(run_id, "assistant", reply)
        return reply

    def answer_question(self, question: str) -> str:
        messages = [SystemMessage(content=self._context_prompt())]
        for turn in self.history[-10:]:
            cls = HumanMessage if turn["role"] == "user" else AIMessage
            messages.append(cls(content=turn["content"]))
        messages.append(HumanMessage(content=question))

        response = self.llm.invoke(messages)
        reply = str(response.content).strip()

        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": reply})
        if self.store and self.run_id:
            self.store.add_message(self.run_id, "user", question)
            self.store.add_message(self.run_id, "assistant", reply)
        return reply

    def revise_report(self, instruction: str) -> str:
        if not self.state.get("draft"):
            return "当前还没有报告，请先给我一个研究主题。"

        writer = make_writer(self.llm)
        update = writer(
            {
                **self.state,
                "review": {"passed": False, "reasons": [instruction]},
            }
        )
        new_draft = update["draft"]

        reviewer = make_reviewer(self.llm)
        review = reviewer({**self.state, "draft": new_draft})["review"]

        self.state["draft"] = new_draft
        self.state["review"] = review

        content = new_draft
        if review and not review.get("passed") and review.get("reasons"):
            content += "\n\n## 审校意见\n" + "\n".join(
                f"- {reason}" for reason in review["reasons"]
            )
        path = Path(self.report_path or "output/report.md")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

        verdict = "通过" if review.get("passed") else "未通过（意见已附在报告末尾）"
        reply = f"报告已更新，审校{verdict}：{path}"
        self.history.append({"role": "user", "content": instruction})
        self.history.append({"role": "assistant", "content": reply})
        if self.store and self.run_id:
            self.store.add_message(self.run_id, "user", instruction)
            self.store.add_message(self.run_id, "assistant", reply)
        return reply

    def load_session(self, run_id: str) -> str:
        if self.store:
            session = self.store.get_session(run_id)
            if session:
                report_path = session.get("report_path")
                if not report_path or not Path(report_path).exists():
                    return "该会话没有可用的报告文件"

                sources = []
                if session.get("sources_json"):
                    try:
                        sources = json.loads(session["sources_json"])
                    except json.JSONDecodeError:
                        sources = []

                self.run_id = run_id
                self.report_path = report_path
                self.history = [
                    {"role": item["role"], "content": item["content"]}
                    for item in self.store.get_messages(run_id)
                ]
                self.state = {
                    "topic": session.get("topic", ""),
                    "sources": sources,
                    "draft": Path(report_path).read_text(encoding="utf-8"),
                }
                return (
                    f"已载入会话 {run_id}，报告：{report_path}\n"
                    "现在可以用 /ask <问题> 提问，或用 /edit <要求> 修改。"
                )

        # Legacy fallback for sessions written before the SQLite store.
        path = self.settings.output_dir / "sessions" / f"{run_id}.json"
        if not path.exists():
            return f"找不到会话: {run_id}"

        payload = json.loads(path.read_text(encoding="utf-8"))
        report_path = payload.get("report")
        if not report_path or not Path(report_path).exists():
            return "该会话没有可用的报告文件"

        self.run_id = run_id
        self.report_path = report_path
        self.history = payload.get("history", [])
        self.state = {
            "topic": payload.get("topic", ""),
            "sources": payload.get("sources", []),
            "draft": Path(report_path).read_text(encoding="utf-8"),
        }
        return (
            f"已载入会话 {run_id}，报告：{report_path}\n"
            "现在可以用 /ask <问题> 提问，或用 /edit <要求> 修改。"
        )

    def _emit(self, message: str) -> None:
        print(message)
        if self.event_callback is not None:
            self.event_callback(message)

    def _context_prompt(self) -> str:
        return (
            f"{CHAT_SYSTEM_PROMPT}\n\n"
            f"研究主题：{self.state.get('topic', '')}\n\n"
            f"当前报告：\n{self.state.get('draft', '')}\n\n"
            f"可用资料：\n{format_sources(self.state.get('sources', []))}"
        )


def chat_main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")

    from .config import load_settings

    try:
        settings = load_settings()
    except ValueError as exc:
        print(f"配置错误: {exc}")
        return 1

    chat = ResearchChat.from_settings(settings)
    print("Research Team · 多轮对话模式")
    print("直接输入研究主题开始；之后可以继续追问，或要求修改报告。")
    print(
        "命令：/new <主题> 换主题 | /ask <问题> 提问 | /edit <要求> 修改报告 | "
        "/load <run_id> 载入报告 | /report 报告路径 | /quit 退出"
    )

    while True:
        try:
            user = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return 0

        if not user:
            continue
        if user in {"/quit", "/exit"}:
            print("再见。")
            return 0
        if user == "/help":
            print(
                "命令：/new <主题> | /ask <问题> | /edit <要求> | "
                "/load <run_id> | /report | /quit"
            )
            continue
        if user == "/report":
            print(chat.report_path or "还没有报告，请先输入研究主题。")
            continue

        try:
            reply = chat.ask(user)
        except KeyboardInterrupt:
            if chat.run_id:
                print(
                    "\n已中断。可从检查点恢复: "
                    f".venv\\Scripts\\python.exe main.py --resume {chat.run_id}"
                )
            else:
                print("\n已中断。")
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"\n出错: {type(exc).__name__}: {exc}")
            continue

        if reply:
            print(f"\n助手> {reply}")
