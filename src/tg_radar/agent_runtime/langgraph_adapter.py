from __future__ import annotations

from typing import Any, Protocol

from tg_radar.agent_core.models import AgentAction, AgentContextPack, AgentRequest, AgentStep
from tg_radar.agent_core.ports import RuntimeAdapter
from tg_radar.agent_runtime.rule_based import RuleBasedCollectionRuntime


class LangGraphLike(Protocol):
    async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
        ...

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        ...


class LangGraphRuntime:
    def __init__(
        self,
        graph: LangGraphLike | None = None,
        fallback: RuntimeAdapter | None = None,
    ) -> None:
        self.graph = graph
        self.fallback = fallback or RuleBasedCollectionRuntime()

    async def next_action(
        self,
        request: AgentRequest,
        context: AgentContextPack,
        steps: list[AgentStep],
    ) -> AgentAction:
        if self.graph is None:
            return await self.fallback.next_action(request, context, steps)
        state = self._state(request, context, steps)
        if hasattr(self.graph, "ainvoke"):
            result = await self.graph.ainvoke(state)
        else:
            result = self.graph.invoke(state)
        action = result.get("action") if isinstance(result, dict) else None
        if isinstance(action, AgentAction):
            return action
        if isinstance(action, dict):
            return AgentAction.model_validate(action)
        return await self.fallback.next_action(request, context, steps)

    def _state(
        self,
        request: AgentRequest,
        context: AgentContextPack,
        steps: list[AgentStep],
    ) -> dict[str, Any]:
        return {
            "request": request.model_dump(mode="json"),
            "context": context.model_dump(mode="json"),
            "steps": [step.model_dump(mode="json") for step in steps],
        }
