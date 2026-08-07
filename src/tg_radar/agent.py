from __future__ import annotations

from typing import Any

from tg_radar.agent_context import CollectionContextBuilder
from tg_radar.agent_core import (
    AgentAction,
    AgentContextPack,
    AgentObservation,
    AgentOrchestrator,
    AgentRequest,
    AgentStep,
    AgentStateStore,
    AgentVerifier,
    StepCompactor,
    ToolSpec,
)
from tg_radar.agent_core.ports import ToolRegistry
from tg_radar.agent_runtime.openai_compatible import OpenAICompatibleRuntime
from tg_radar.agent_runtime.rule_based import RuleBasedCollectionRuntime as CoreRuleBasedCollectionRuntime
from tg_radar.schemas import (
    AgentActionType,
    CollectionAgentAction,
    CollectionAgentObservation,
    CollectionAgentRequest,
    CollectionAgentResponse,
    CollectionAgentStep,
)


class RuleBasedCollectionRuntime:
    def __init__(self) -> None:
        self.runtime = CoreRuleBasedCollectionRuntime()

    async def next_action(
        self,
        request: CollectionAgentRequest | AgentRequest,
        context: dict[str, Any] | AgentContextPack,
        steps: list[CollectionAgentStep] | list[AgentStep],
    ) -> CollectionAgentAction | AgentAction:
        core_request = _to_core_request(request)
        core_context = _to_core_context(context)
        core_steps = _to_core_steps(steps)
        action = await self.runtime.next_action(core_request, core_context, core_steps)
        return action if isinstance(request, AgentRequest) else _to_collection_action(action)


class OpenAICompatibleCollectionRuntime:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float,
        system_prompt_name: str = "collection_agent_system.md",
        prompt_dir: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1200,
        planner_timeout_seconds: float = 40.0,
    ) -> None:
        from tg_radar.agent_runtime.prompt_loader import PromptLoader

        self.runtime = OpenAICompatibleRuntime(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
            system_prompt_name=system_prompt_name,
            prompt_loader=PromptLoader(override_dir=prompt_dir),
            temperature=temperature,
            max_tokens=max_tokens,
            planner_timeout_seconds=planner_timeout_seconds,
        )

    async def next_action(
        self,
        request: CollectionAgentRequest | AgentRequest,
        context: dict[str, Any] | AgentContextPack,
        steps: list[CollectionAgentStep] | list[AgentStep],
    ) -> CollectionAgentAction | AgentAction:
        core_request = _to_core_request(request)
        core_context = _to_core_context(context)
        core_steps = _to_core_steps(steps)
        action = await self.runtime.next_action(core_request, core_context, core_steps)
        return action if isinstance(request, AgentRequest) else _to_collection_action(action)


class DataCollectionAgent:
    def __init__(
        self,
        context_builder: CollectionContextBuilder,
        tool_registry: ToolRegistry,
        runtime: RuleBasedCollectionRuntime | OpenAICompatibleCollectionRuntime | None = None,
        state_store: AgentStateStore | None = None,
        compactor: StepCompactor | None = None,
        verifier: AgentVerifier | None = None,
    ) -> None:
        self.context_builder = context_builder
        self.tool_registry = tool_registry
        selected_runtime = runtime or RuleBasedCollectionRuntime()
        self.orchestrator = AgentOrchestrator(
            context_builder=context_builder,
            tool_registry=tool_registry,
            runtime=selected_runtime,
            state_store=state_store,
            compactor=compactor,
            verifier=verifier,
        )

    async def run(self, request: CollectionAgentRequest) -> CollectionAgentResponse:
        response = await self.orchestrator.run(_to_core_request(request))
        return CollectionAgentResponse(
            run_id=request.run_id,
            final=response.final,
            steps=[_to_collection_step(step) for step in response.steps],
            context=response.context,
        )


def _to_core_request(request: CollectionAgentRequest | AgentRequest) -> AgentRequest:
    if isinstance(request, AgentRequest):
        return request
    return AgentRequest(
        goal=request.goal,
        mode=request.mode.value,
        max_steps=request.max_steps,
        inputs={
            "phase": request.phase,
            "run_id": request.run_id,
            "topic": request.topic,
            "article_focus": request.article_focus,
            "focus_keywords": request.focus_keywords,
            "keywords": request.keywords,
            "seed_channels": request.seed_channels,
            "task_name": request.task_name,
            "depth": request.depth,
            "limit": request.limit,
            "pages_per_channel": request.pages_per_channel,
            "crawl": request.crawl,
            "crawl_mode": request.crawl_mode.value,
            "freshness_days": request.freshness_days,
            "since": request.since.isoformat() if request.since else None,
            "interval_seconds": request.interval_seconds,
            "enabled": request.enabled,
            "tool_name": request.tool_name,
            "tool_args": request.tool_args,
            "approved_tools": request.approved_tools,
        },
    )


def _to_core_context(context: dict[str, Any] | AgentContextPack) -> AgentContextPack:
    if isinstance(context, AgentContextPack):
        return context
    raw_tools = context.get("tool_manifest") or context.get("available_tools") or []
    tools = []
    for item in raw_tools:
        if isinstance(item, ToolSpec):
            tools.append(item)
        elif isinstance(item, dict) and item.get("name"):
            tools.append(
                ToolSpec(
                    name=str(item["name"]),
                    risk_level=str(item.get("risk_level") or "safe"),
                    description=str(item.get("description") or ""),
                    args_schema=dict(item.get("args_schema") or {}),
                )
            )
    return AgentContextPack(
        user_goal=str(context.get("goal") or context.get("user_goal") or ""),
        task_state=dict(context.get("task_state") or {}),
        domain_context={key: value for key, value in context.items() if key not in {"tool_manifest", "available_tools"}},
        tool_manifest=tools,
    )


def _to_core_steps(steps: list[CollectionAgentStep] | list[AgentStep]) -> list[AgentStep]:
    if not steps:
        return []
    if isinstance(steps[0], AgentStep):
        return steps  # type: ignore[return-value]
    return [_to_core_step(step) for step in steps]  # type: ignore[arg-type]


def _to_core_step(step: CollectionAgentStep) -> AgentStep:
    return AgentStep(
        step=step.step,
        action=AgentAction(
            type=step.action.type.value,
            tool_name=step.action.tool_name,
            args=step.action.args,
            reason=step.action.reason,
            expected_result=step.action.expected_result,
        ),
        observation=_to_core_observation(step.observation) if step.observation else None,
    )


def _to_core_observation(observation: CollectionAgentObservation) -> AgentObservation:
    return AgentObservation(
        tool_name=observation.tool_name,
        ok=observation.ok,
        summary=observation.summary,
        data=observation.data,
    )


def _to_collection_step(step: AgentStep) -> CollectionAgentStep:
    return CollectionAgentStep(
        step=step.step,
        action=_to_collection_action(step.action),
        observation=_to_collection_observation(step.observation) if step.observation else None,
    )


def _to_collection_action(action: AgentAction) -> CollectionAgentAction:
    return CollectionAgentAction(
        type=AgentActionType(action.type.value),
        tool_name=action.tool_name,
        args=action.args,
        reason=action.reason,
        expected_result=action.expected_result,
    )


def _to_collection_observation(observation: AgentObservation) -> CollectionAgentObservation:
    return CollectionAgentObservation(
        tool_name=observation.tool_name,
        ok=observation.ok,
        summary=observation.summary,
        data=observation.data,
    )
