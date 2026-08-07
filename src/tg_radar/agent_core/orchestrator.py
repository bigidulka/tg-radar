from __future__ import annotations

import time

from tg_radar.agent_core.compaction import NoOpStepCompactor, StepCompactor
from tg_radar.agent_core.models import AgentActionType, AgentRequest, AgentResponse, AgentStep
from tg_radar.agent_core.phases import AgentPhasePolicy
from tg_radar.agent_core.policy import AgentPolicy
from tg_radar.agent_core.ports import ContextBuilder, RuntimeAdapter, ToolRegistry
from tg_radar.agent_core.state import AgentStateStore, MemoryAgentStateStore
from tg_radar.agent_core.verification import AgentVerifier, NoOpAgentVerifier


class AgentOrchestrator:
    def __init__(
        self,
        context_builder: ContextBuilder,
        tool_registry: ToolRegistry,
        runtime: RuntimeAdapter,
        policy: AgentPolicy | None = None,
        state_store: AgentStateStore | None = None,
        compactor: StepCompactor | None = None,
        verifier: AgentVerifier | None = None,
        phase_policy: AgentPhasePolicy | None = None,
    ) -> None:
        self.context_builder = context_builder
        self.tool_registry = tool_registry
        self.runtime = runtime
        self.policy = policy or AgentPolicy()
        self.state_store = state_store or MemoryAgentStateStore()
        self.compactor = compactor or NoOpStepCompactor()
        self.verifier = verifier or NoOpAgentVerifier()
        self.phase_policy = phase_policy or AgentPhasePolicy()

    async def run(self, request: AgentRequest) -> AgentResponse:
        steps: list[AgentStep] = []
        final = "stopped"
        context = None
        run_id = self._run_id(request)
        phase = self.phase_policy.initial_phase(request)

        for _ in range(request.max_steps):
            all_tools = self.tool_registry.specs()
            tools = self.phase_policy.tools_for_phase(phase, all_tools)
            runtime_steps = self.compactor.compact(steps)
            phase_request = self.phase_policy.request_for_phase(request, phase)
            context = await self.context_builder.build(phase_request, tools, runtime_steps)
            action = await self.runtime.next_action(phase_request, context, runtime_steps)
            action = self.policy.validate_action(action, tools, self._approved_tools(request))
            step = AgentStep(step=len(steps) + 1, action=action)
            steps.append(step)

            if action.type == AgentActionType.final:
                final = self._final_message(steps)
                await self.state_store.append_step(run_id, step)
                break
            if action.type == AgentActionType.ask_user:
                final = action.expected_result or action.reason
                await self.state_store.append_step(run_id, step)
                break
            if action.type == AgentActionType.stop:
                final = action.reason
                await self.state_store.append_step(run_id, step)
                break
            if action.type == AgentActionType.plan_update:
                await self.state_store.append_step(run_id, step)
                phase = self.phase_policy.next_phase(phase, action, step)
                continue
            if action.type == AgentActionType.tool_call and action.tool_name:
                started = time.perf_counter()
                step.observation = await self.tool_registry.run(action.tool_name, action.args)
                step.observation.data["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
                await self.state_store.append_step(run_id, step)
                verification = await self.verifier.verify(step, all_tools, self.tool_registry, len(steps) + 1)
                if verification:
                    steps.append(verification)
                    await self.state_store.append_step(run_id, verification)
                if not step.observation.ok:
                    final = step.observation.summary
                    break
                phase = self.phase_policy.next_phase(phase, action, step)
        else:
            final = "max_steps reached"

        return AgentResponse(
            final=final,
            steps=steps,
            context=context.model_dump(mode="json") if context else {},
        )

    def _final_message(self, steps: list[AgentStep]) -> str:
        if steps and steps[-1].action.type == AgentActionType.final:
            reason = steps[-1].action.reason.strip()
            if reason and reason != "last tool completed":
                return reason
        observations = [step.observation for step in steps if step.observation]
        if not observations:
            return "done"
        return observations[-1].summary

    def _run_id(self, request: AgentRequest) -> str:
        explicit = request.inputs.get("run_id")
        if isinstance(explicit, str) and explicit:
            return explicit
        return f"{request.mode}:{request.goal}:{request.max_steps}"

    def _approved_tools(self, request: AgentRequest) -> set[str]:
        value = request.inputs.get("approved_tools") or []
        if isinstance(value, str):
            return {item.strip() for item in value.split(",") if item.strip()}
        if isinstance(value, list | tuple | set):
            return {str(item) for item in value if str(item)}
        return set()
