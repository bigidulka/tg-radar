from __future__ import annotations

from typing import Any

from tg_radar.agent_core.models import AgentAction, AgentActionType, AgentContextPack, AgentRequest, AgentStep


class RuleBasedCollectionRuntime:
    def __init__(self, default_task_name: str = "collection-agent-task") -> None:
        self.default_task_name = default_task_name

    async def next_action(
        self,
        request: AgentRequest,
        context: AgentContextPack,
        steps: list[AgentStep],
    ) -> AgentAction:
        if steps and steps[-1].observation and steps[-1].observation.ok:
            next_article_action = self._next_article_action(request, context, steps)
            if next_article_action:
                return next_article_action
            if self._should_review_evidence(request, context, steps):
                return AgentAction(
                    type=AgentActionType.tool_call,
                    tool_name="review_evidence",
                    args={
                        "topic": self._input(request, "topic") or self._input(request, "task_name") or self._slug(request.goal),
                        "limit": min(int(self._input(request, "limit", 80) or 80), 120),
                        "min_required": int(self._input(request, "min_required_evidence", 5) or 5),
                    },
                    reason="article goal needs evidence quality review before writing",
                    expected_result="usable and rejected evidence counts",
                )
            if self._should_write_article(request, context, steps):
                return AgentAction(
                    type=AgentActionType.tool_call,
                    tool_name="write_article",
                    args={
                        "topic": self._input(request, "topic") or self._input(request, "task_name") or self._slug(request.goal),
                        "instruction": request.goal,
                        "limit": min(int(self._input(request, "limit", 80) or 80), 120),
                        "min_required": int(self._input(request, "min_required_evidence", 5) or 5),
                        "use_llm": False,
                    },
                    reason="evidence review passed; continue to article workflow",
                    expected_result="content cards, outline, and draft",
                )
            return AgentAction(type=AgentActionType.final, reason=self._final_report(request, steps[-1]))
        if steps and steps[-1].observation and not steps[-1].observation.ok:
            return AgentAction(type=AgentActionType.stop, reason="tool failed")
        if self._input(request, "tool_name"):
            tool_name = str(self._input(request, "tool_name"))
            return AgentAction(
                type=AgentActionType.tool_call,
                tool_name=tool_name,
                args=dict(self._input(request, "tool_args", {})),
                reason=f"explicit tool request routed to {tool_name}",
                expected_result="tool observation",
            )
        if not self._has_search_input(request) and request.mode not in {"status", "run_task"}:
            return AgentAction(
                type=AgentActionType.ask_user,
                reason="missing search input",
                expected_result="keywords or seed_channels",
            )

        tool_name = self._select_tool(request)
        return AgentAction(
            type=AgentActionType.tool_call,
            tool_name=tool_name,
            args=self._tool_args(tool_name, request),
            reason=f"{request.mode} request routed to {tool_name}",
            expected_result="collection core observation",
        )

    def _select_tool(self, request: AgentRequest) -> str:
        if request.mode in {"status", "discover", "ingest", "create_task", "run_task"}:
            return "core_status" if request.mode == "status" else request.mode
        return "ingest" if bool(self._input(request, "crawl", True)) else "discover"

    def _tool_args(self, tool_name: str, request: AgentRequest) -> dict[str, Any]:
        if tool_name == "core_status":
            return {}
        if tool_name == "run_task":
            return {"name": self._task_name(request)}
        if tool_name == "create_task":
            return {
                "name": self._task_name(request),
                "keywords": self._input(request, "keywords", []),
                "seed_channels": self._input(request, "seed_channels", []),
                "depth": self._input(request, "depth", 1),
                "limit": self._input(request, "limit", 50),
                "pages_per_channel": self._input(request, "pages_per_channel", 1),
                "crawl": self._input(request, "crawl", True),
                "crawl_mode": self._input(request, "crawl_mode", "backfill"),
                "freshness_days": self._freshness_days(request),
                "interval_seconds": self._input(request, "interval_seconds", 300),
                "enabled": self._input(request, "enabled", True),
            }
        return {
            "topic": self._input(request, "topic") or self._input(request, "task_name"),
            "keywords": self._input(request, "keywords", []),
            "seed_channels": self._input(request, "seed_channels", []),
            "depth": self._input(request, "depth", 1),
            "limit": self._input(request, "limit", 50),
            "pages_per_channel": self._input(request, "pages_per_channel", 1),
            "crawl": self._input(request, "crawl", True),
            "crawl_mode": self._input(request, "crawl_mode", "backfill"),
            "freshness_days": self._freshness_days(request),
            "since": self._input(request, "since"),
            "max_live_crawl": 0 if self._goal_wants_article(request) else None,
        }

    def _has_search_input(self, request: AgentRequest) -> bool:
        return bool(
            self._input(request, "keywords", [])
            or self._input(request, "seed_channels", [])
        )

    def _task_name(self, request: AgentRequest) -> str:
        return str(self._input(request, "task_name") or self._slug(request.goal))

    def _slug(self, value: str) -> str:
        allowed = []
        previous_dash = False
        for char in value.strip().lower():
            keep = char.isalnum() or char in {"_", "-"}
            next_char = char if keep else "-"
            if next_char == "-" and previous_dash:
                continue
            allowed.append(next_char)
            previous_dash = next_char == "-"
        slug = "".join(allowed).strip("-")
        return slug[:64] or self.default_task_name

    def _input(self, request: AgentRequest, key: str, default: Any = None) -> Any:
        return request.inputs.get(key, default)

    def _freshness_days(self, request: AgentRequest) -> int | None:
        explicit = self._input(request, "freshness_days")
        if explicit:
            return int(explicit)
        if self._goal_wants_freshness(request):
            return 14
        return None

    def _goal_wants_freshness(self, request: AgentRequest) -> bool:
        text = request.goal.lower()
        return any(
            word in text
            for word in (
                "digest",
                "news",
                "\u0434\u0430\u0439\u0434\u0436\u0435\u0441\u0442",
                "\u043d\u043e\u0432\u043e\u0441\u0442",
                "\u0441\u0432\u0435\u0436",
            )
        )

    def _final_report(self, request: AgentRequest, step: AgentStep) -> str:
        observation = step.observation
        if not observation:
            return "done"
        if observation.tool_name == "write_article":
            data = observation.data
            return "\n".join(
                [
                    f"Article draft ready: {data.get('topic') or self._input(request, 'topic') or request.goal}",
                    f"Evidence: usable={data.get('usable_evidence', 0)}, rejected={data.get('rejected_evidence', 0)}.",
                    f"Cards: {data.get('cards', 0)}; outline blocks: {data.get('blocks', 0)}; draft chars: {data.get('draft_chars', 0)}.",
                    "Open Writing to inspect/edit the draft.",
                ]
            )
        if observation.tool_name == "cluster_signals":
            data = observation.data
            return f"Signals clustered: signals={len(data.get('signals') or [])}, usable={data.get('usable', 0)}."
        if observation.tool_name == "infer_signal_taxonomy":
            data = observation.data
            return f"Signal taxonomy ready: signals={len(data.get('taxonomy') or [])}, usable={data.get('usable', 0)}."
        if observation.tool_name == "build_evidence_map":
            data = observation.data
            return f"Evidence map ready: claims={len(data.get('claims') or [])}, signals={len(data.get('signals') or [])}."
        if observation.tool_name == "propose_angles":
            data = observation.data
            return f"Angles proposed: angles={len(data.get('angles') or [])}."
        if observation.tool_name == "build_outline":
            data = observation.data
            return f"Outline ready: article_id={data.get('article_id')}, blocks={data.get('outline_blocks', 0)}."
        if observation.tool_name == "write_draft":
            data = observation.data
            return f"Draft ready: article_id={data.get('article_id')}, draft_id={data.get('draft_id')}, chars={data.get('draft_chars', 0)}."
        if observation.tool_name == "verify_claims":
            data = observation.data
            return f"Claim verification: ok={bool(data.get('ok'))}, checks={len(data.get('checks') or [])}, unsupported={data.get('unsupported_claims', 0)}."
        if observation.tool_name == "final_article":
            data = observation.data
            return f"Final article ready: article_id={data.get('article_id')}, draft_id={data.get('draft_id')}, chars={data.get('final_chars', 0)}."
        if observation.tool_name == "review_evidence":
            data = observation.data
            if data.get("ok"):
                return (
                    f"Evidence review passed: usable={data.get('usable', 0)}, "
                    f"rejected={data.get('rejected', 0)}."
                )
            return "\n".join(
                [
                    f"Evidence review failed: usable={data.get('usable', 0)}/{data.get('min_required', 0)}.",
                    "Article was not written because collected messages do not match the topic strongly enough.",
                    f"Rejected: {data.get('rejected', 0)} of {data.get('reviewed', 0)} reviewed messages.",
                ]
            )
        if observation.tool_name == "core_status":
            return observation.summary
        if observation.tool_name == "discover":
            candidates = observation.data.get("candidates") or []
            top = [f"@{item.get('username')}" for item in candidates[:8] if item.get("username")]
            return (
                f"Discovery complete: candidates={len(candidates)}"
                + (f"; top={', '.join(top)}" if top else "")
            )
        if observation.tool_name == "ingest":
            data = observation.data
            candidates = int(data.get("candidates_found") or 0)
            crawled = int(data.get("channels_crawled") or 0)
            messages = int(data.get("messages_saved") or 0) + int(data.get("cached_messages_used") or 0)
            errors = data.get("errors") or []
            topic = self._input(request, "topic") or self._input(request, "task_name") or request.goal
            return "\n".join(
                [
                    f"Research brief: {topic}",
                    f"Evidence collected: candidates={candidates}, channels={crawled}, messages={messages}.",
                    "Next: open Data for raw evidence and Writing for cards/outline/draft.",
                    "No article claims were invented; use collected Telegram messages as evidence.",
                    f"Errors: {len(errors)}" if errors else "Errors: 0",
                ]
            )
        return observation.summary

    def _next_article_action(self, request: AgentRequest, context: AgentContextPack, steps: list[AgentStep]) -> AgentAction | None:
        if not self._goal_wants_article(request):
            return None
        last = steps[-1]
        observation = last.observation
        if not observation:
            return None
        tools = {tool.name for tool in context.tool_manifest}
        topic = self._input(request, "topic") or self._input(request, "task_name") or self._slug(request.goal)
        limit = min(int(self._input(request, "limit", 80) or 80), 120)
        min_required = int(self._input(request, "min_required_evidence", 5) or 5)
        article_focus = self._input(request, "article_focus") or request.goal
        focus_keywords = self._input(request, "focus_keywords", []) or []
        focus_args = {"article_focus": article_focus, "focus_keywords": focus_keywords}
        if observation.tool_name == "review_evidence" and not observation.data.get("ok"):
            return None
        sequence = [
            ("ingest", "review_evidence", {"topic": topic, "limit": limit, "min_required": min_required, **focus_args}),
            ("review_evidence", "infer_signal_taxonomy", {"topic": topic, "limit": limit, "min_required": min_required, **focus_args}),
            ("infer_signal_taxonomy", "cluster_signals", {"topic": topic, "limit": limit, "min_required": min_required, **focus_args}),
            ("cluster_signals", "build_evidence_map", {"topic": topic, "limit": limit, "min_required": min_required, **focus_args}),
            ("build_evidence_map", "propose_angles", {"topic": topic, "limit": limit, **focus_args}),
            ("propose_angles", "build_outline", {"topic": topic, "limit": limit, **focus_args}),
            ("build_outline", "write_draft", {"topic": topic, "limit": limit, "use_llm": False, **focus_args}),
            ("write_draft", "verify_claims", {"topic": topic, **focus_args}),
            ("verify_claims", "final_article", {"topic": topic, **focus_args}),
        ]
        for after_tool, next_tool, args in sequence:
            if observation.tool_name != after_tool:
                continue
            if next_tool not in tools:
                continue
            if any(step.action.tool_name == next_tool for step in steps):
                continue
            return AgentAction(
                type=AgentActionType.tool_call,
                tool_name=next_tool,
                args=args,
                reason=f"article workflow continues with {next_tool}",
                expected_result="article workflow observation",
            )
        return None

    def _should_review_evidence(self, request: AgentRequest, context: AgentContextPack, steps: list[AgentStep]) -> bool:
        last = steps[-1]
        if not last.observation or last.observation.tool_name != "ingest":
            return False
        if "review_evidence" not in {tool.name for tool in context.tool_manifest}:
            return False
        if any(step.action.tool_name == "review_evidence" for step in steps):
            return False
        return self._goal_wants_article(request)

    def _should_write_article(self, request: AgentRequest, context: AgentContextPack, steps: list[AgentStep]) -> bool:
        last = steps[-1]
        if not last.observation or last.observation.tool_name != "review_evidence":
            return False
        if not last.observation.data.get("ok"):
            return False
        if "write_article" not in {tool.name for tool in context.tool_manifest}:
            return False
        if any(step.action.tool_name == "write_article" for step in steps):
            return False
        return True

    def _goal_wants_article(self, request: AgentRequest) -> bool:
        goal = request.goal.lower()
        return any(
            token in goal
            for token in [
                "\u0441\u0442\u0430\u0442\u044c",
                "\u0447\u0435\u0440\u043d\u043e\u0432",
                "draft",
                "article",
                "outline",
                "evidence map",
                "\u0437\u0430\u0433\u043e\u043b\u043e\u0432",
            ]
        )
