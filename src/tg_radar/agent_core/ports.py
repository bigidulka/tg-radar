from __future__ import annotations

from typing import Protocol

from tg_radar.agent_core.models import AgentAction, AgentContextPack, AgentObservation, AgentRequest, AgentStep, ToolSpec


class ContextBuilder(Protocol):
    async def build(
        self,
        request: AgentRequest,
        tools: list[ToolSpec],
        steps: list[AgentStep],
    ) -> AgentContextPack:
        ...


class RuntimeAdapter(Protocol):
    async def next_action(
        self,
        request: AgentRequest,
        context: AgentContextPack,
        steps: list[AgentStep],
    ) -> AgentAction:
        ...


class ToolRegistry(Protocol):
    def specs(self) -> list[ToolSpec]:
        ...

    async def run(self, name: str, args: dict) -> AgentObservation:
        ...
