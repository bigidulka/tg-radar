from __future__ import annotations

from tg_radar.agent_core.models import AgentAction, AgentActionType, ToolDangerLevel, ToolSpec


class AgentPolicy:
    def __init__(
        self,
        denied_danger_levels: set[ToolDangerLevel] | None = None,
        require_approval: bool = True,
    ) -> None:
        self.denied_danger_levels = denied_danger_levels or {ToolDangerLevel.secret, ToolDangerLevel.destructive}
        self.require_approval = require_approval

    def validate_action(
        self,
        action: AgentAction,
        tools: list[ToolSpec],
        approved_tools: set[str] | None = None,
    ) -> AgentAction:
        if action.type != AgentActionType.tool_call:
            return action
        if not action.tool_name:
            return AgentAction(type=AgentActionType.stop, reason="tool_call missing tool_name")
        tool = {item.name: item for item in tools}.get(action.tool_name)
        if tool is None:
            return AgentAction(type=AgentActionType.stop, reason=f"invalid tool selected: {action.tool_name}")
        if tool.danger_level in self.denied_danger_levels:
            return AgentAction(type=AgentActionType.stop, reason=f"tool denied by policy: {tool.name}")
        if self.require_approval and tool.requires_approval and tool.name not in (approved_tools or set()):
            return AgentAction(
                type=AgentActionType.ask_user,
                reason=f"tool requires approval: {tool.name}",
                expected_result="approval",
            )
        return action
