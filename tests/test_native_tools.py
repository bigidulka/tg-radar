from __future__ import annotations

from pathlib import Path
import sys

from tg_radar.agent_context import ModularContextBuilder, TaskStateModule, ToolManifestModule
from tg_radar.agent_core.models import AgentActionType, AgentRequest, ToolDangerLevel
from tg_radar.agent_core.orchestrator import AgentOrchestrator
from tg_radar.agent_runtime.rule_based import RuleBasedCollectionRuntime
from tg_radar.agent_tools import CompositeToolRegistry, NativeDevToolRegistry


def test_native_tool_specs_are_configured():
    registry = NativeDevToolRegistry(Path.cwd())
    specs = {tool.name: tool for tool in registry.specs()}

    assert {
        "list_files",
        "read_file",
        "search_text",
        "search_symbols",
        "memory_read",
        "memory_write",
        "task_tracker",
        "checkpoint_create",
        "checkpoint_list",
        "checkpoint_restore",
        "write_file",
        "apply_patch",
        "git_status",
        "git_diff",
        "run_shell",
        "run_tests",
    } <= set(specs)
    assert specs["run_tests"].danger_level == ToolDangerLevel.shell
    assert specs["write_file"].requires_approval is True
    assert specs["run_shell"].requires_approval is True
    assert specs["checkpoint_restore"].requires_approval is True
    assert {phase.value for phase in specs["write_file"].allowed_phases} == {"edit", "fix"}
    assert {phase.value for phase in specs["run_tests"].allowed_phases} == {"verify", "fix", "finalize"}
    assert specs["read_file"].input_schema["required"] == ["path"]


async def test_native_tools_read_and_search_workspace():
    registry = NativeDevToolRegistry(Path.cwd())

    listed = await registry.run("list_files", {"path": "src/tg_radar/agent_tools", "max_entries": 20})
    read = await registry.run("read_file", {"path": "src/tg_radar/agent_tools/native.py", "max_chars": 1000})
    searched = await registry.run("search_text", {"query": "NativeDevToolRegistry", "path": "src/tg_radar/agent_tools", "max_matches": 5})
    symbols = await registry.run("search_symbols", {"query": "NativeDevToolRegistry", "path": "src/tg_radar/agent_tools"})
    outside = await registry.run("read_file", {"path": "../outside"})

    assert listed.ok is True
    assert any(item["path"] == "src/tg_radar/agent_tools/native.py" for item in listed.data["entries"])
    assert read.ok is True
    assert "NativeDevToolRegistry" in read.data["text"]
    assert searched.ok is True
    assert searched.data["matches"]
    assert symbols.ok is True
    assert symbols.data["matches"][0]["kind"] == "class"
    assert outside.ok is False
    assert "outside workspace" in outside.summary


async def test_native_git_status_runs_configured_command():
    registry = NativeDevToolRegistry(Path.cwd())

    result = await registry.run("git_status", {"max_chars": 500})

    assert result.tool_name == "git_status"
    assert result.data["returncode"] in {0, 128}


async def test_native_write_patch_and_shell_tools(tmp_path):
    registry = NativeDevToolRegistry(tmp_path)

    written = await registry.run("write_file", {"path": "notes/todo.txt", "content": "alpha\n", "create_dirs": True})
    patched = await registry.run(
        "apply_patch",
        {"path": "notes/todo.txt", "old_text": "alpha", "new_text": "beta"},
    )
    read = await registry.run("read_file", {"path": "notes/todo.txt"})
    shell = await registry.run("run_shell", {"command": [sys.executable, "-c", "print('ok')"], "max_chars": 20})

    assert written.ok is True
    assert patched.ok is True
    assert patched.data["replacements"] == 1
    assert read.data["text"] == "beta\n"
    assert shell.ok is True
    assert shell.data["output"] == "ok"


async def test_native_workspace_policy_blocks_protected_paths_and_shell(tmp_path):
    registry = NativeDevToolRegistry(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text("SECRET=x\n", encoding="utf-8")

    write = await registry.run("write_file", {"path": ".env", "content": "SECRET=y\n"})
    patch = await registry.run("apply_patch", {"path": ".env", "old_text": "x", "new_text": "y"})
    shell = await registry.run("run_shell", {"command": ["sh", "-c", "echo no"]})

    assert write.ok is False
    assert write.summary == "protected path"
    assert patch.ok is False
    assert patch.summary == "protected path"
    assert shell.ok is False
    assert shell.summary == "blocked command"
    assert env_file.read_text(encoding="utf-8") == "SECRET=x\n"


async def test_native_memory_and_task_tracker_tools(tmp_path):
    registry = NativeDevToolRegistry(tmp_path)

    wrote = await registry.run("memory_write", {"key": "decision", "value": {"choice": "custom-core"}, "tags": ["arch"]})
    read = await registry.run("memory_read", {"key": "decision"})
    added = await registry.run("task_tracker", {"action": "add", "id": "t1", "title": "Wire runtime", "status": "todo"})
    updated = await registry.run("task_tracker", {"action": "update", "id": "t1", "status": "done"})
    listed = await registry.run("task_tracker", {"action": "list"})

    assert wrote.ok is True
    assert read.data["items"][0]["value"] == {"choice": "custom-core"}
    assert added.ok is True
    assert updated.data["task"]["status"] == "done"
    assert listed.data["tasks"][0]["title"] == "Wire runtime"
    assert (tmp_path / ".tg-radar-agent" / "memory.json").is_file()
    assert (tmp_path / ".tg-radar-agent" / "tasks.json").is_file()


async def test_native_checkpoint_create_list_and_restore(tmp_path):
    registry = NativeDevToolRegistry(tmp_path)
    source = tmp_path / "src"
    source.mkdir()
    target = source / "app.py"
    target.write_text("version = 1\n", encoding="utf-8")
    protected = tmp_path / ".env"
    protected.write_text("SECRET=x\n", encoding="utf-8")

    created = await registry.run("checkpoint_create", {"id": "before-edit", "paths": ["."], "metadata": {"phase": "edit"}})
    target.write_text("version = 2\n", encoding="utf-8")
    listed = await registry.run("checkpoint_list", {})
    restored = await registry.run("checkpoint_restore", {"id": "before-edit"})

    assert created.ok is True
    assert created.data["id"] == "before-edit"
    assert listed.data["checkpoints"][0]["metadata"] == {"phase": "edit"}
    assert restored.ok is True
    assert target.read_text(encoding="utf-8") == "version = 1\n"
    assert protected.read_text(encoding="utf-8") == "SECRET=x\n"


async def test_composite_tool_registry_dispatches_by_tool_name():
    registry = CompositeToolRegistry([
        NativeDevToolRegistry(Path.cwd(), enabled_tools={"read_file"}),
        NativeDevToolRegistry(Path.cwd(), enabled_tools={"search_text"}),
    ])

    specs = {tool.name for tool in registry.specs()}
    result = await registry.run("search_text", {"query": "CompositeToolRegistry", "path": "tests/test_native_tools.py"})

    assert specs == {"read_file", "search_text"}
    assert result.ok is True
    assert result.data["matches"]


async def test_agent_loop_can_run_explicit_native_tool():
    registry = NativeDevToolRegistry(Path.cwd(), enabled_tools={"read_file"})
    orchestrator = AgentOrchestrator(
        ModularContextBuilder([TaskStateModule(), ToolManifestModule()]),
        registry,
        RuleBasedCollectionRuntime(),
    )

    response = await orchestrator.run(
        AgentRequest(
            goal="read package config",
            inputs={"tool_name": "read_file", "tool_args": {"path": "pyproject.toml", "max_chars": 200}},
            max_steps=2,
        )
    )

    assert response.final == "read pyproject.toml"
    assert response.steps[0].observation is not None
    assert "tg-radar" in response.steps[0].observation.data["text"]


async def test_agent_policy_requires_approval_for_write_tool(tmp_path):
    registry = NativeDevToolRegistry(tmp_path, enabled_tools={"write_file"})
    orchestrator = AgentOrchestrator(
        ModularContextBuilder([TaskStateModule(), ToolManifestModule()]),
        registry,
        RuleBasedCollectionRuntime(),
    )

    response = await orchestrator.run(
        AgentRequest(
            goal="write file",
            inputs={"phase": "edit", "tool_name": "write_file", "tool_args": {"path": "x.txt", "content": "x"}},
            max_steps=2,
        )
    )

    assert response.final == "approval"
    assert response.steps[0].action.type == AgentActionType.ask_user
    assert response.steps[0].observation is None


async def test_agent_policy_runs_preapproved_write_tool(tmp_path):
    registry = NativeDevToolRegistry(tmp_path, enabled_tools={"write_file"})
    orchestrator = AgentOrchestrator(
        ModularContextBuilder([TaskStateModule(), ToolManifestModule()]),
        registry,
        RuleBasedCollectionRuntime(),
    )

    response = await orchestrator.run(
        AgentRequest(
            goal="write file",
            inputs={
                "phase": "edit",
                "tool_name": "write_file",
                "tool_args": {"path": "x.txt", "content": "x"},
                "approved_tools": ["write_file"],
            },
            max_steps=2,
        )
    )

    assert response.final == "wrote x.txt"
    assert response.steps[0].observation is not None
    assert (tmp_path / "x.txt").read_text(encoding="utf-8") == "x"


async def test_agent_policy_requires_approval_for_checkpoint_restore(tmp_path):
    registry = NativeDevToolRegistry(tmp_path, enabled_tools={"checkpoint_restore"})
    orchestrator = AgentOrchestrator(
        ModularContextBuilder([TaskStateModule(), ToolManifestModule()]),
        registry,
        RuleBasedCollectionRuntime(),
    )

    response = await orchestrator.run(
        AgentRequest(
            goal="restore checkpoint",
            inputs={
                "phase": "fix",
                "tool_name": "checkpoint_restore",
                "tool_args": {"id": "before-edit"},
            },
            max_steps=2,
        )
    )

    assert response.final == "approval"
    assert response.steps[0].action.type == AgentActionType.ask_user
