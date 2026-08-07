from __future__ import annotations

import json

from tg_radar.agent_context import (
    AgentMemoryModule,
    ConstraintsModule,
    CurrentDiffModule,
    ModularContextBuilder,
    RecentObservationsModule,
    RelevantFilesModule,
    TaskStateModule,
    ToolManifestModule,
    WorkspaceRepoMapModule,
)
from tg_radar.agent_core.models import AgentAction, AgentActionType, AgentObservation, AgentRequest, AgentStep, ToolSpec


async def test_modular_context_builder_merges_context_parts():
    builder = ModularContextBuilder(
        [
            TaskStateModule(),
            RecentObservationsModule(limit=1),
            ToolManifestModule(),
            ConstraintsModule(context_budget=5000, strategy="unit"),
        ]
    )
    step = AgentStep(
        step=1,
        action=AgentAction(type=AgentActionType.tool_call, tool_name="read", reason="inspect"),
        observation=AgentObservation(tool_name="read", ok=True, summary="ok"),
    )

    context = await builder.build(
        AgentRequest(goal="inspect project", inputs={"topic": "agents"}, max_steps=4),
        [ToolSpec(name="read", description="Read")],
        [step],
    )

    assert context.user_goal == "inspect project"
    assert context.task_state["inputs"] == {"topic": "agents"}
    assert context.task_state["steps_count"] == 1
    assert context.recent_observations[0]["observation"]["summary"] == "ok"
    assert context.tool_manifest[0].name == "read"
    assert context.constraints["runtime_contract"] == "structured_action_only"
    assert context.token_budget_report["context_budget"] == 5000


async def test_workspace_context_modules_build_dev_context(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    target = source / "agent_kernel.py"
    target.write_text("class AgentKernel:\n    pass\n", encoding="utf-8")
    ignored = tmp_path / "node_modules"
    ignored.mkdir()
    (ignored / "skip.py").write_text("AgentKernel\n", encoding="utf-8")

    builder = ModularContextBuilder(
        [
            WorkspaceRepoMapModule(tmp_path, {"node_modules"}, 20),
            RelevantFilesModule(tmp_path, {"node_modules"}, 3, 200),
            CurrentDiffModule(tmp_path, 200),
        ]
    )

    context = await builder.build(AgentRequest(goal="inspect AgentKernel"), [], [])

    assert {"path": "src", "type": "dir"} in context.repo_map
    assert any(item["path"] == "src/agent_kernel.py" for item in context.repo_map)
    assert [item["path"] for item in context.relevant_files] == ["src/agent_kernel.py"]
    assert context.relevant_files[0]["matches"][0]["line"] == 1
    assert "not a git repository" in context.current_diff.lower()


async def test_agent_memory_module_reads_project_state(tmp_path):
    state_dir = tmp_path / ".tg-radar-agent"
    state_dir.mkdir()
    (state_dir / "memory.json").write_text(
        json.dumps([{"key": "decision", "value": "custom-core"}]),
        encoding="utf-8",
    )
    (state_dir / "tasks.json").write_text(
        json.dumps([{"id": "t1", "status": "done"}]),
        encoding="utf-8",
    )
    builder = ModularContextBuilder(
        [AgentMemoryModule(tmp_path, ".tg-radar-agent", "memory.json", "tasks.json", 5)]
    )

    context = await builder.build(AgentRequest(goal="continue"), [], [])

    assert context.long_term_memory == [{"key": "decision", "value": "custom-core"}]
    assert context.task_board == [{"id": "t1", "status": "done"}]
