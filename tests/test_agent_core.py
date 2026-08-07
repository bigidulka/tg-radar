from __future__ import annotations

from tg_radar.agent_core.models import (
    AgentAction,
    AgentActionType,
    AgentContextPack,
    AgentObservation,
    AgentRequest,
    AgentStep,
    ToolDangerLevel,
    ToolSpec,
)
from tg_radar.agent_core.compaction import WindowStepCompactor
from tg_radar.agent_core.orchestrator import AgentOrchestrator
from tg_radar.agent_core.policy import AgentPolicy
from tg_radar.agent_core.state import MemoryAgentStateStore
from tg_radar.agent_core.verification import ToolResultVerifier


class StaticContextBuilder:
    async def build(self, request: AgentRequest, tools: list[ToolSpec], steps: list[AgentStep]) -> AgentContextPack:
        return AgentContextPack(user_goal=request.goal, tool_manifest=tools)


class RecordingContextBuilder:
    def __init__(self) -> None:
        self.step_counts: list[int] = []
        self.compaction_summaries: list[str] = []

    async def build(self, request: AgentRequest, tools: list[ToolSpec], steps: list[AgentStep]) -> AgentContextPack:
        self.step_counts.append(len(steps))
        if steps and steps[0].step == 0 and steps[0].observation:
            self.compaction_summaries.append(steps[0].observation.summary)
        return AgentContextPack(user_goal=request.goal, tool_manifest=tools)


class StaticRuntime:
    def __init__(self, action: AgentAction) -> None:
        self.action = action

    async def next_action(
        self,
        request: AgentRequest,
        context: AgentContextPack,
        steps: list[AgentStep],
    ) -> AgentAction:
        return self.action


class SequenceRuntime:
    def __init__(self, actions: list[AgentAction]) -> None:
        self.actions = actions
        self.context_tools: list[list[str]] = []
        self.phases: list[str | None] = []

    async def next_action(
        self,
        request: AgentRequest,
        context: AgentContextPack,
        steps: list[AgentStep],
    ) -> AgentAction:
        self.context_tools.append([tool.name for tool in context.tool_manifest])
        self.phases.append(request.inputs.get("phase"))
        if self.actions:
            return self.actions.pop(0)
        return AgentAction(type=AgentActionType.final, reason="done")


class StaticToolRegistry:
    def __init__(self, tools: list[ToolSpec]) -> None:
        self.tools = tools
        self.calls: list[tuple[str, dict]] = []

    def specs(self) -> list[ToolSpec]:
        return self.tools

    async def run(self, name: str, args: dict) -> AgentObservation:
        self.calls.append((name, args))
        return AgentObservation(tool_name=name, ok=True, summary="tool ok")


async def test_orchestrator_stops_invalid_tool_action():
    state_store = MemoryAgentStateStore()
    orchestrator = AgentOrchestrator(
        StaticContextBuilder(),
        StaticToolRegistry([ToolSpec(name="allowed", description="Allowed tool")]),
        StaticRuntime(
            AgentAction(
                type=AgentActionType.tool_call,
                tool_name="missing",
                reason="runtime selected unavailable tool",
            )
        ),
        state_store=state_store,
    )
    request = AgentRequest(goal="inspect project")

    response = await orchestrator.run(request)
    events = await state_store.events("auto:inspect project:6")

    assert response.final == "invalid tool selected: missing"
    assert response.steps[0].action.type == AgentActionType.stop
    assert len(events) == 1


async def test_orchestrator_requests_approval_for_guarded_tool():
    registry = StaticToolRegistry([
        ToolSpec(name="write_file", description="Write file", side_effects=True, requires_approval=True)
    ])
    orchestrator = AgentOrchestrator(
        StaticContextBuilder(),
        registry,
        StaticRuntime(
            AgentAction(
                type=AgentActionType.tool_call,
                tool_name="write_file",
                args={"path": "src/example.py"},
                reason="need write",
            )
        ),
    )

    response = await orchestrator.run(AgentRequest(goal="modify project"))

    assert response.final == "approval"
    assert response.steps[0].action.type == AgentActionType.ask_user
    assert registry.calls == []


async def test_orchestrator_filters_tools_by_current_phase():
    runtime = SequenceRuntime(
        [
            AgentAction(
                type=AgentActionType.tool_call,
                tool_name="write_file",
                args={"path": "x.txt"},
                reason="write before edit phase",
            )
        ]
    )
    registry = StaticToolRegistry(
        [
            ToolSpec(name="read_file", description="Read"),
            ToolSpec(name="write_file", description="Write", allowed_phases=["edit"]),
        ]
    )
    orchestrator = AgentOrchestrator(StaticContextBuilder(), registry, runtime)

    response = await orchestrator.run(AgentRequest(goal="write"))

    assert runtime.context_tools == [["read_file"]]
    assert response.final == "invalid tool selected: write_file"
    assert registry.calls == []


async def test_orchestrator_changes_phase_from_plan_update():
    runtime = SequenceRuntime(
        [
            AgentAction(type=AgentActionType.plan_update, args={"phase": "edit"}, reason="move to edit"),
            AgentAction(
                type=AgentActionType.tool_call,
                tool_name="write_file",
                args={"path": "x.txt"},
                reason="write in edit phase",
            ),
        ]
    )
    registry = StaticToolRegistry(
        [
            ToolSpec(name="read_file", description="Read"),
            ToolSpec(name="write_file", description="Write", allowed_phases=["edit"]),
        ]
    )
    orchestrator = AgentOrchestrator(StaticContextBuilder(), registry, runtime, policy=AgentPolicy(require_approval=False))

    response = await orchestrator.run(AgentRequest(goal="write", max_steps=2))

    assert runtime.context_tools == [["read_file"], ["read_file", "write_file"]]
    assert runtime.phases == ["understand", "edit"]
    assert registry.calls == [("write_file", {"path": "x.txt"})]
    assert response.steps[-1].observation is not None


async def test_orchestrator_compacts_context_view_without_dropping_run_steps():
    state_store = MemoryAgentStateStore()
    context_builder = RecordingContextBuilder()
    registry = StaticToolRegistry([ToolSpec(name="allowed", description="Allowed tool")])
    orchestrator = AgentOrchestrator(
        context_builder,
        registry,
        StaticRuntime(
            AgentAction(
                type=AgentActionType.tool_call,
                tool_name="allowed",
                reason="keep running",
            )
        ),
        state_store=state_store,
        compactor=WindowStepCompactor(recent_steps=1),
    )

    response = await orchestrator.run(AgentRequest(goal="loop", max_steps=4))
    events = await state_store.events("auto:loop:4")

    assert response.final == "max_steps reached"
    assert len(response.steps) == 4
    assert len(events) == 4
    assert context_builder.step_counts == [0, 1, 2, 2]
    assert context_builder.compaction_summaries == [
        "compacted steps=1 ok=1 failed=0",
        "compacted steps=2 ok=2 failed=0",
    ]


async def test_orchestrator_runs_verifier_after_write_tool():
    state_store = MemoryAgentStateStore()
    registry = StaticToolRegistry(
        [
            ToolSpec(name="write_file", description="Write file", danger_level=ToolDangerLevel.write),
            ToolSpec(name="run_tests", description="Run tests", danger_level=ToolDangerLevel.shell),
        ]
    )
    orchestrator = AgentOrchestrator(
        StaticContextBuilder(),
        registry,
        StaticRuntime(
            AgentAction(
                type=AgentActionType.tool_call,
                tool_name="write_file",
                args={"path": "x.txt"},
                reason="write",
            )
        ),
        state_store=state_store,
        verifier=ToolResultVerifier("run_tests"),
    )

    response = await orchestrator.run(AgentRequest(goal="verify write", max_steps=1))
    events = await state_store.events("auto:verify write:1")

    assert [step.action.tool_name for step in response.steps] == ["write_file", "run_tests"]
    assert registry.calls == [("write_file", {"path": "x.txt"}), ("run_tests", {})]
    assert len(events) == 2


async def test_orchestrator_uses_explicit_run_id():
    state_store = MemoryAgentStateStore()
    orchestrator = AgentOrchestrator(
        StaticContextBuilder(),
        StaticToolRegistry([ToolSpec(name="allowed", description="Allowed tool")]),
        StaticRuntime(AgentAction(type=AgentActionType.final, reason="done")),
        state_store=state_store,
    )

    await orchestrator.run(AgentRequest(goal="custom run", inputs={"run_id": "ui-run-1"}))

    assert len(await state_store.events("ui-run-1")) == 1
    assert await state_store.events("auto:custom run:6") == []


def test_tool_spec_syncs_legacy_fields():
    spec = ToolSpec(
        name="ingest",
        description="Ingest data",
        risk_level="network_write",
        args_schema={"type": "object"},
    )

    assert spec.input_schema == {"type": "object"}
    assert spec.danger_level == ToolDangerLevel.network


def test_agent_event_table_is_mapped():
    from tg_radar.db import AgentEventLog, Base

    assert AgentEventLog.__tablename__ == "agent_events"
    assert "agent_events" in Base.metadata.tables
