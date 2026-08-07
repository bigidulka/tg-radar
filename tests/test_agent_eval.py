from __future__ import annotations

import json

from tg_radar.agent_context import ModularContextBuilder, TaskStateModule, ToolManifestModule
from tg_radar.agent_core.models import AgentAction, AgentActionType, AgentContextPack, AgentObservation, AgentRequest, AgentStep, ToolSpec
from tg_radar.agent_core.orchestrator import AgentOrchestrator
from tg_radar.agent_eval import AgentEvalCase, AgentEvalRunner
from tg_radar.agent_runtime.rule_based import RuleBasedCollectionRuntime
from tg_radar.agent_tools import NativeDevToolRegistry


class ScriptedRuntime:
    def __init__(self, tools: list[str]) -> None:
        self.tools = tools

    async def next_action(self, request: AgentRequest, context: AgentContextPack, steps: list[AgentStep]) -> AgentAction:
        index = len([step for step in steps if step.action.tool_name])
        if index >= len(self.tools):
            return AgentAction(type=AgentActionType.final, reason="script complete")
        return AgentAction(
            type=AgentActionType.tool_call,
            tool_name=self.tools[index],
            args={"topic": request.inputs.get("topic", "vibe-coding-ai")},
            reason=f"scripted {self.tools[index]}",
        )


class ScriptedArticleTools:
    names = [
        "review_evidence",
        "infer_signal_taxonomy",
        "cluster_signals",
        "build_evidence_map",
        "propose_angles",
        "build_outline",
        "write_draft",
        "verify_claims",
        "final_article",
        "write_article",
    ]

    def specs(self) -> list[ToolSpec]:
        return [ToolSpec(name=name, description=name) for name in self.names]

    async def run(self, name: str, args: dict) -> AgentObservation:
        return AgentObservation(tool_name=name, ok=True, summary=f"{name} ok", data={"ok": True, "topic": args.get("topic")})


async def test_agent_eval_runner_scores_tool_sequence(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    orchestrator = AgentOrchestrator(
        ModularContextBuilder([TaskStateModule(), ToolManifestModule()]),
        NativeDevToolRegistry(tmp_path, enabled_tools={"read_file"}),
        RuleBasedCollectionRuntime(),
    )
    runner = AgentEvalRunner(orchestrator)

    results = await runner.run_cases(
        [
            AgentEvalCase(
                name="read-config",
                request=AgentRequest(
                    goal="read config",
                    inputs={"tool_name": "read_file", "tool_args": {"path": "pyproject.toml"}},
                    max_steps=2,
                ),
                expected_final="read pyproject.toml",
                expected_tools=["read_file"],
            )
        ]
    )

    assert results[0].ok is True
    assert results[0].score == 1.0
    assert results[0].failures == []


async def test_agent_eval_runner_loads_json_file(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "name": "read-config",
                        "request": {
                            "goal": "read config",
                            "inputs": {"tool_name": "read_file", "tool_args": {"path": "pyproject.toml"}},
                            "max_steps": 2,
                        },
                        "expected_final": "read pyproject.toml",
                        "expected_tools": ["read_file"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    runner = AgentEvalRunner(
        AgentOrchestrator(
            ModularContextBuilder([TaskStateModule(), ToolManifestModule()]),
            NativeDevToolRegistry(tmp_path, enabled_tools={"read_file"}),
            RuleBasedCollectionRuntime(),
        )
    )

    results = await runner.run_file(cases)

    assert results[0].ok is True


async def test_agent_eval_runner_detects_required_and_forbidden_tools(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    runner = AgentEvalRunner(
        AgentOrchestrator(
            ModularContextBuilder([TaskStateModule(), ToolManifestModule()]),
            NativeDevToolRegistry(tmp_path, enabled_tools={"read_file"}),
            RuleBasedCollectionRuntime(),
        )
    )

    results = await runner.run_cases(
        [
            AgentEvalCase(
                name="harness-regression",
                request=AgentRequest(
                    goal="read config",
                    inputs={"tool_name": "read_file", "tool_args": {"path": "pyproject.toml"}},
                    max_steps=2,
                ),
                expected_tool_prefix=["ingest"],
                required_tools=["verify_claims"],
                forbidden_tools=["read_file"],
                expected_final_contains="missing text",
                min_steps=2,
                max_steps=2,
                harness={"model": "gpt-5.5", "planner_timeout_seconds": 40, "prompt": "collection_agent_system.md"},
            )
        ]
    )

    result = results[0]
    assert result.ok is False
    assert result.score < 1.0
    assert result.harness["model"] == "gpt-5.5"
    assert "tool prefix mismatch" in result.failures
    assert "missing required tools: verify_claims" in result.failures
    assert "forbidden tools used: read_file" in result.failures
    assert "final missing expected text" in result.failures
    assert "too few tool steps" in result.failures


async def test_agent_eval_runner_loads_harness_profile_from_json(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "name": "read-config-profile",
                        "request": {
                            "goal": "read config",
                            "inputs": {"tool_name": "read_file", "tool_args": {"path": "pyproject.toml"}},
                            "max_steps": 2,
                        },
                        "required_tools": ["read_file"],
                        "expected_final_contains": "pyproject.toml",
                        "harness": {
                            "model": "gpt-5.5",
                            "temperature": 0,
                            "planner_timeout_seconds": 40,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    runner = AgentEvalRunner(
        AgentOrchestrator(
            ModularContextBuilder([TaskStateModule(), ToolManifestModule()]),
            NativeDevToolRegistry(tmp_path, enabled_tools={"read_file"}),
            RuleBasedCollectionRuntime(),
        )
    )

    results = await runner.run_file(cases)

    assert results[0].ok is True
    assert results[0].harness["planner_timeout_seconds"] == 40


async def test_agent_eval_runner_scores_granular_article_loop_without_legacy_tool():
    expected = [
        "review_evidence",
        "infer_signal_taxonomy",
        "cluster_signals",
        "build_evidence_map",
        "propose_angles",
        "build_outline",
        "write_draft",
        "verify_claims",
        "final_article",
    ]
    runner = AgentEvalRunner(
        AgentOrchestrator(
            ModularContextBuilder([TaskStateModule(), ToolManifestModule()]),
            ScriptedArticleTools(),
            ScriptedRuntime(expected),
        )
    )

    results = await runner.run_cases(
        [
            AgentEvalCase(
                name="granular-article-loop",
                request=AgentRequest(goal="write evidence based article", inputs={"topic": "vibe-coding-ai"}, max_steps=12),
                expected_tool_prefix=expected,
                required_tools=expected,
                forbidden_tools=["write_article"],
                expected_final="script complete",
                harness={
                    "model": "scripted-gpt-5.5",
                    "runtime_provider": "openai_compatible",
                    "temperature": 0,
                    "planner_timeout_seconds": 40,
                },
            )
        ]
    )

    assert results[0].ok is True
    assert results[0].tools == expected
    assert results[0].harness["runtime_provider"] == "openai_compatible"
