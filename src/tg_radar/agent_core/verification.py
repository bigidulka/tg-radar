from __future__ import annotations

from typing import Protocol

from tg_radar.agent_core.models import AgentStep, ToolDangerLevel, ToolSpec
from tg_radar.agent_core.ports import ToolRegistry


class AgentVerifier(Protocol):
    async def verify(
        self,
        step: AgentStep,
        tools: list[ToolSpec],
        tool_registry: ToolRegistry,
        next_step_index: int,
    ) -> AgentStep | None:
        ...


class NoOpAgentVerifier:
    async def verify(
        self,
        step: AgentStep,
        tools: list[ToolSpec],
        tool_registry: ToolRegistry,
        next_step_index: int,
    ) -> AgentStep | None:
        return None


class ToolResultVerifier:
    def __init__(
        self,
        verify_tool_name: str,
        trigger_danger_levels: set[ToolDangerLevel] | None = None,
    ) -> None:
        self.verify_tool_name = verify_tool_name
        self.trigger_danger_levels = trigger_danger_levels or {ToolDangerLevel.write}

    async def verify(
        self,
        step: AgentStep,
        tools: list[ToolSpec],
        tool_registry: ToolRegistry,
        next_step_index: int,
    ) -> AgentStep | None:
        if not step.observation or not step.observation.ok or not step.action.tool_name:
            return None
        tool_by_name = {tool.name: tool for tool in tools}
        source_tool = tool_by_name.get(step.action.tool_name)
        verify_tool = tool_by_name.get(self.verify_tool_name)
        if source_tool is None or verify_tool is None:
            return None
        if source_tool.danger_level not in self.trigger_danger_levels:
            return None
        verify_action = step.action.model_copy(
            update={
                "tool_name": self.verify_tool_name,
                "args": {},
                "reason": f"verify after {step.action.tool_name}",
                "expected_result": "verification observation",
            }
        )
        verify_step = AgentStep(step=next_step_index, action=verify_action)
        verify_step.observation = await tool_registry.run(self.verify_tool_name, {})
        return verify_step
