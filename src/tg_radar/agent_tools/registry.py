from __future__ import annotations

from tg_radar.agent_core.models import AgentObservation, ToolSpec
from tg_radar.agent_core.ports import ToolRegistry


class CompositeToolRegistry:
    def __init__(self, registries: list[ToolRegistry]) -> None:
        self.registries = registries

    def specs(self) -> list[ToolSpec]:
        seen: set[str] = set()
        tools: list[ToolSpec] = []
        for registry in self.registries:
            for spec in registry.specs():
                if spec.name in seen:
                    continue
                seen.add(spec.name)
                tools.append(spec)
        return tools

    async def run(self, name: str, args: dict) -> AgentObservation:
        for registry in self.registries:
            if name in {spec.name for spec in registry.specs()}:
                return await registry.run(name, args)
        return AgentObservation(tool_name=name, ok=False, summary="unknown tool")
