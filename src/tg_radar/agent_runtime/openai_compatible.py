from __future__ import annotations

import asyncio
import json
from typing import Any

from tg_radar.agent_core.models import AgentAction, AgentActionType, AgentContextPack, AgentRequest, AgentStep
from tg_radar.agent_core.ports import RuntimeAdapter
from tg_radar.agent_runtime.prompt_loader import PromptLoader
from tg_radar.agent_runtime.openai_transport import ChatCompletionsTransport, OpenAITransport, OpenAITransportConfig
from tg_radar.agent_runtime.rule_based import RuleBasedCollectionRuntime


class OpenAICompatibleRuntime:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float,
        system_prompt_name: str,
        prompt_loader: PromptLoader | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1200,
        planner_timeout_seconds: float = 40.0,
        fallback: RuntimeAdapter | None = None,
        transport: OpenAITransport | None = None,
    ) -> None:
        self.config = OpenAITransportConfig(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.system_prompt_name = system_prompt_name
        self.prompt_loader = prompt_loader or PromptLoader()
        self.fallback = fallback or RuleBasedCollectionRuntime()
        self.transport = transport or ChatCompletionsTransport(self.config)
        self.planner_timeout_seconds = planner_timeout_seconds

    async def next_action(
        self,
        request: AgentRequest,
        context: AgentContextPack,
        steps: list[AgentStep],
    ) -> AgentAction:
        terminal = self._terminal_tool_final(steps)
        if terminal:
            return terminal
        if self._prefer_rule_based(request):
            return await self.fallback.next_action(request, context, steps)
        try:
            content = await asyncio.wait_for(
                self.transport.complete_json(self._messages(request, context)),
                timeout=min(self.config.timeout_seconds, self.planner_timeout_seconds),
            )
            action = AgentAction.model_validate(json.loads(content))
            forced = self._force_final_article(action, context, request, steps)
            if forced:
                return forced
            return self._validate_tool(action, context, request, steps)
        except Exception as exc:
            if request.mode == "auto":
                return AgentAction(
                    type=AgentActionType.stop,
                    reason=f"planner failed: {type(exc).__name__}",
                )
            return await self.fallback.next_action(request, context, steps)

    def _prefer_rule_based(self, request: AgentRequest) -> bool:
        if request.inputs.get("tool_name"):
            return False
        if request.mode in {"status", "discover", "ingest", "create_task", "run_task"}:
            return True
        return False

    def _messages(self, request: AgentRequest, context: AgentContextPack) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.prompt_loader.load(self.system_prompt_name)},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "request": request.model_dump(mode="json"),
                        "context": context.model_dump(mode="json"),
                    },
                    ensure_ascii=False,
                ),
            },
        ]

    def _validate_tool(
        self,
        action: AgentAction,
        context: AgentContextPack,
        request: AgentRequest,
        steps: list[AgentStep],
    ) -> AgentAction:
        if action.type != AgentActionType.tool_call:
            return action
        available = {tool.name for tool in context.tool_manifest}
        if not action.tool_name or action.tool_name not in available:
            return AgentAction(
                type=AgentActionType.stop,
                reason=f"invalid tool selected: {action.tool_name}",
            )
        tool_name = action.tool_name
        if request.mode == "auto" and tool_name in {"create_task", "run_task"} and self._goal_wants_article(request):
            return AgentAction(
                type=AgentActionType.stop,
                reason=f"unsafe recurring tool for one-shot article goal: {tool_name}",
            )
        if tool_name == "write_article" and not self._has_passed_evidence_review(steps):
            if "review_evidence" in available:
                return AgentAction(
                    type=AgentActionType.tool_call,
                    tool_name="review_evidence",
                    args=self._policy_args("review_evidence", action.args, request, steps),
                    reason="guardrail: review evidence before writing article",
                    expected_result="usable and rejected evidence counts",
                )
            return AgentAction(type=AgentActionType.stop, reason="write_article requires review_evidence tool")
        if tool_name == "write_article" and self._has_failed_evidence_review(steps):
            return AgentAction(type=AgentActionType.final, reason=self._failed_review_message(steps))
        if tool_name == "final_article" and not self._has_passed_tool(steps, "verify_claims"):
            if "verify_claims" in available:
                return AgentAction(
                    type=AgentActionType.tool_call,
                    tool_name="verify_claims",
                    args=self._policy_args("verify_claims", action.args, request, steps),
                    reason="guardrail: verify claims before final article",
                    expected_result="claim verification report",
                )
            return AgentAction(type=AgentActionType.stop, reason="final_article requires verify_claims tool")
        return AgentAction(
            type=action.type,
            tool_name=tool_name,
            args=self._policy_args(tool_name, action.args, request, steps),
            reason=action.reason,
            expected_result=action.expected_result,
        )

    def _force_final_article(
        self,
        action: AgentAction,
        context: AgentContextPack,
        request: AgentRequest,
        steps: list[AgentStep],
    ) -> AgentAction | None:
        if action.type != AgentActionType.final or not self._goal_wants_article(request):
            return None
        available = {tool.name for tool in context.tool_manifest}
        if "final_article" not in available or self._has_passed_tool(steps, "final_article"):
            return None
        last = steps[-1].observation if steps else None
        if not last or last.tool_name != "verify_claims" or not last.ok or not bool(last.data.get("ok")):
            return None
        return AgentAction(
            type=AgentActionType.tool_call,
            tool_name="final_article",
            args=self._policy_args("final_article", action.args, request, steps),
            reason="guardrail: final article goal requires final_article after verified claims",
            expected_result="final article artifact",
        )

    def _policy_args(self, tool_name: str, args: dict[str, Any], request: AgentRequest, steps: list[AgentStep] | None = None) -> dict[str, Any]:
        if tool_name == "core_status":
            return {}
        if tool_name == "run_task":
            return {"name": args.get("name") or request.inputs.get("task_name") or self._slug(request.goal)}
        if tool_name == "create_task":
            return {
                "name": args.get("name") or request.inputs.get("task_name") or self._slug(request.goal),
                "keywords": list(args.get("keywords") or request.inputs.get("keywords") or []),
                "seed_channels": list(args.get("seed_channels") or request.inputs.get("seed_channels") or []),
                "depth": args.get("depth") or request.inputs.get("depth", 1),
                "limit": args.get("limit") or request.inputs.get("limit", 50),
                "pages_per_channel": args.get("pages_per_channel") or request.inputs.get("pages_per_channel", 1),
                "crawl": args.get("crawl") if "crawl" in args else request.inputs.get("crawl", True),
                "crawl_mode": args.get("crawl_mode") or request.inputs.get("crawl_mode", "backfill"),
                "freshness_days": args.get("freshness_days") or self._freshness_days(request),
                "interval_seconds": args.get("interval_seconds") or request.inputs.get("interval_seconds", 300),
                "enabled": args.get("enabled") if "enabled" in args else request.inputs.get("enabled", True),
            }
        if tool_name == "review_evidence":
            return {
                "topic": self._topic_arg(args, request),
                "article_focus": self._article_focus_arg(args, request),
                "focus_keywords": self._focus_keywords_arg(args, request),
                "limit": args.get("limit") or request.inputs.get("limit", 50),
                "min_required": args.get("min_required") or request.inputs.get("min_required_evidence", 5),
            }
        if tool_name in {"infer_signal_taxonomy", "cluster_signals", "build_evidence_map"}:
            return {
                "topic": self._topic_arg(args, request),
                "article_focus": self._article_focus_arg(args, request),
                "focus_keywords": self._focus_keywords_arg(args, request),
                "limit": args.get("limit") or request.inputs.get("limit", 50),
                "min_required": args.get("min_required") or request.inputs.get("min_required_evidence", 5),
                **({"taxonomy": args.get("taxonomy")} if isinstance(args.get("taxonomy"), list) else {}),
            }
        if tool_name in {"propose_angles", "build_outline"}:
            return {
                "topic": self._topic_arg(args, request),
                "article_focus": self._article_focus_arg(args, request),
                "focus_keywords": self._focus_keywords_arg(args, request),
                "limit": args.get("limit") or request.inputs.get("limit", 50),
            }
        if tool_name == "write_draft":
            return {
                "topic": self._topic_arg(args, request),
                "article_focus": self._article_focus_arg(args, request),
                "focus_keywords": self._focus_keywords_arg(args, request),
                "limit": args.get("limit") or request.inputs.get("limit", 50),
                "use_llm": bool(args.get("use_llm", False)),
            }
        if tool_name in {"verify_claims", "final_article"}:
            payload = {
                "topic": self._topic_arg(args, request),
                "article_focus": self._article_focus_arg(args, request),
                "focus_keywords": self._focus_keywords_arg(args, request),
            }
            article_id = self._latest_article_id(steps or []) or args.get("article_id")
            if article_id:
                payload["article_id"] = article_id
            return payload
        if tool_name == "write_article":
            return {
                "topic": self._topic_arg(args, request),
                "article_focus": self._article_focus_arg(args, request),
                "focus_keywords": self._focus_keywords_arg(args, request),
                "title": args.get("title"),
                "angle": args.get("angle"),
                "instruction": args.get("instruction") or request.goal,
                "limit": args.get("limit") or request.inputs.get("limit", 50),
                "min_required": args.get("min_required") or request.inputs.get("min_required_evidence", 5),
                "use_llm": bool(args.get("use_llm", False)),
            }
        return {
            "topic": args.get("topic") or request.inputs.get("topic") or request.inputs.get("task_name"),
            "keywords": list(args.get("keywords") or request.inputs.get("keywords") or []),
            "seed_channels": list(args.get("seed_channels") or request.inputs.get("seed_channels") or []),
            "depth": args.get("depth") or request.inputs.get("depth", 1),
            "limit": args.get("limit") or request.inputs.get("limit", 50),
            "pages_per_channel": args.get("pages_per_channel") or request.inputs.get("pages_per_channel", 1),
            "crawl": args.get("crawl") if "crawl" in args else request.inputs.get("crawl", True),
            "crawl_mode": args.get("crawl_mode") or request.inputs.get("crawl_mode", "backfill"),
            "freshness_days": args.get("freshness_days") or self._freshness_days(request),
            "since": args.get("since") or request.inputs.get("since"),
            "max_live_crawl": args.get("max_live_crawl") or (0 if self._goal_wants_article(request) else None),
        }

    def _topic_arg(self, args: dict[str, Any], request: AgentRequest) -> str:
        return str(args.get("topic") or request.inputs.get("topic") or request.inputs.get("task_name") or self._slug(request.goal))

    def _article_focus_arg(self, args: dict[str, Any], request: AgentRequest) -> str | None:
        value = args.get("article_focus") or request.inputs.get("article_focus")
        if value:
            return str(value)
        if self._goal_wants_article(request):
            return request.goal
        return None

    def _freshness_days(self, request: AgentRequest) -> int | None:
        explicit = request.inputs.get("freshness_days")
        if explicit:
            return int(explicit)
        text = request.goal.lower()
        if any(
            word in text
            for word in (
                "digest",
                "news",
                "\u0434\u0430\u0439\u0434\u0436\u0435\u0441\u0442",
                "\u043d\u043e\u0432\u043e\u0441\u0442",
                "\u0441\u0432\u0435\u0436",
            )
        ):
            return 14
        return None

    def _focus_keywords_arg(self, args: dict[str, Any], request: AgentRequest) -> list[str]:
        raw = args.get("focus_keywords") or request.inputs.get("focus_keywords") or []
        return [str(item) for item in raw if str(item).strip()]

    def _latest_article_id(self, steps: list[AgentStep]) -> int | None:
        for step in reversed(steps):
            observation = step.observation
            if not observation or not observation.ok:
                continue
            if observation.tool_name not in {"build_outline", "write_draft", "verify_claims", "final_article"}:
                continue
            article_id = observation.data.get("article_id")
            if isinstance(article_id, int):
                return article_id
            if isinstance(article_id, str) and article_id.isdigit():
                return int(article_id)
        return None

    def _terminal_tool_final(self, steps: list[AgentStep]) -> AgentAction | None:
        if not steps:
            return None
        observation = steps[-1].observation
        if not observation or not observation.ok:
            return None
        if observation.tool_name == "final_article" and bool(observation.data.get("final")):
            return AgentAction(type=AgentActionType.final, reason=observation.summary)
        return None

    def _has_passed_evidence_review(self, steps: list[AgentStep]) -> bool:
        return any(
            step.observation
            and step.observation.tool_name == "review_evidence"
            and step.observation.ok
            and bool(step.observation.data.get("ok"))
            for step in steps
        )

    def _has_passed_tool(self, steps: list[AgentStep], tool_name: str) -> bool:
        return any(
            step.observation
            and step.observation.tool_name == tool_name
            and step.observation.ok
            and bool(step.observation.data.get("ok", True))
            for step in steps
        )

    def _has_failed_evidence_review(self, steps: list[AgentStep]) -> bool:
        return any(
            step.observation
            and step.observation.tool_name == "review_evidence"
            and (not step.observation.ok or not bool(step.observation.data.get("ok")))
            for step in steps
        )

    def _failed_review_message(self, steps: list[AgentStep]) -> str:
        for step in reversed(steps):
            observation = step.observation
            if observation and observation.tool_name == "review_evidence":
                data = observation.data
                return (
                    f"Evidence review failed: usable={data.get('usable', 0)}/{data.get('min_required', 0)}. "
                    "Article was not written because collected messages do not match the topic strongly enough."
                )
        return "Evidence review failed."

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
        return slug[:64] or "collection-agent-task"
