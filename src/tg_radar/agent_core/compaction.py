from __future__ import annotations

from typing import Protocol

from tg_radar.agent_core.models import AgentAction, AgentActionType, AgentObservation, AgentStep


class StepCompactor(Protocol):
    def compact(self, steps: list[AgentStep]) -> list[AgentStep]:
        ...


class NoOpStepCompactor:
    def compact(self, steps: list[AgentStep]) -> list[AgentStep]:
        return list(steps)


class WindowStepCompactor:
    def __init__(self, recent_steps: int, summary_tool_name: str = "agent_compactor") -> None:
        self.recent_steps = recent_steps
        self.summary_tool_name = summary_tool_name

    def compact(self, steps: list[AgentStep]) -> list[AgentStep]:
        if len(steps) <= self.recent_steps:
            return list(steps)
        hidden = steps[: -self.recent_steps]
        visible = steps[-self.recent_steps :]
        return [self._summary_step(hidden), *visible]

    def _summary_step(self, steps: list[AgentStep]) -> AgentStep:
        ok_count = sum(1 for step in steps if step.observation and step.observation.ok)
        failed_count = sum(1 for step in steps if step.observation and not step.observation.ok)
        return AgentStep(
            step=0,
            action=AgentAction(
                type=AgentActionType.plan_update,
                reason="compacted prior agent steps",
                expected_result="context summary",
            ),
            observation=AgentObservation(
                tool_name=self.summary_tool_name,
                ok=True,
                summary=f"compacted steps={len(steps)} ok={ok_count} failed={failed_count}",
                data={
                    "compacted_steps": len(steps),
                    "ok_steps": ok_count,
                    "failed_steps": failed_count,
                    "last_compacted_step": steps[-1].step if steps else 0,
                },
            ),
        )
