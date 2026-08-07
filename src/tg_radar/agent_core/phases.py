from __future__ import annotations

from tg_radar.agent_core.models import AgentAction, AgentActionType, AgentPhase, AgentRequest, AgentStep, ToolSpec


class AgentPhasePolicy:
    def initial_phase(self, request: AgentRequest) -> AgentPhase:
        raw = request.inputs.get("phase") or request.mode
        return self._phase(raw, AgentPhase.understand)

    def tools_for_phase(self, phase: AgentPhase, tools: list[ToolSpec]) -> list[ToolSpec]:
        return [tool for tool in tools if not tool.allowed_phases or phase in tool.allowed_phases]

    def request_for_phase(self, request: AgentRequest, phase: AgentPhase) -> AgentRequest:
        return request.model_copy(update={"inputs": request.inputs | {"phase": phase.value}})

    def next_phase(self, current: AgentPhase, action: AgentAction, step: AgentStep) -> AgentPhase:
        requested = action.args.get("phase") if isinstance(action.args, dict) else None
        if requested:
            return self._phase(requested, current)
        if action.type == AgentActionType.plan_update:
            return AgentPhase.plan
        if action.type != AgentActionType.tool_call or not step.observation:
            return current
        if not step.observation.ok:
            return AgentPhase.fix
        if action.tool_name in {"write_file", "apply_patch", "memory_write", "task_tracker"}:
            return AgentPhase.verify
        if action.tool_name in {"run_tests", "git_diff", "git_status"} and current in {AgentPhase.verify, AgentPhase.fix}:
            return AgentPhase.finalize
        return current

    def _phase(self, value: object, default: AgentPhase) -> AgentPhase:
        try:
            return AgentPhase(str(value))
        except ValueError:
            return default
