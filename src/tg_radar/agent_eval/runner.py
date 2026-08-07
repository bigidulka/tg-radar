from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tg_radar.agent_core.models import AgentRequest, AgentResponse
from tg_radar.agent_core.orchestrator import AgentOrchestrator


class AgentEvalCase(BaseModel):
    name: str
    request: AgentRequest
    expected_final: str | None = None
    expected_final_contains: str | None = None
    expected_tools: list[str] = Field(default_factory=list)
    expected_tool_prefix: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    min_steps: int | None = None
    max_steps: int | None = None
    harness: dict[str, Any] = Field(default_factory=dict)


class AgentEvalResult(BaseModel):
    name: str
    ok: bool
    score: float
    final: str
    tools: list[str] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    harness: dict[str, Any] = Field(default_factory=dict)


class AgentEvalRunner:
    def __init__(self, orchestrator: AgentOrchestrator) -> None:
        self.orchestrator = orchestrator

    async def run_cases(self, cases: list[AgentEvalCase]) -> list[AgentEvalResult]:
        results = []
        for case in cases:
            response = await self.orchestrator.run(case.request)
            results.append(self._result(case, response))
        return results

    async def run_file(self, path: str | Path) -> list[AgentEvalResult]:
        cases = self._load_cases(path)
        return await self.run_cases(cases)

    def _load_cases(self, path: str | Path) -> list[AgentEvalCase]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        items = data.get("cases") if isinstance(data, dict) else data
        if not isinstance(items, list):
            return []
        return [AgentEvalCase.model_validate(item) for item in items]

    def _result(self, case: AgentEvalCase, response: AgentResponse) -> AgentEvalResult:
        tools = [step.action.tool_name for step in response.steps if step.action.tool_name]
        failures = self._failures(case, response.final, tools)
        return AgentEvalResult(
            name=case.name,
            ok=not failures,
            score=self._score(failures),
            final=response.final,
            tools=tools,
            failures=failures,
            harness=case.harness,
        )

    def _failures(self, case: AgentEvalCase, final: str, tools: list[str]) -> list[str]:
        failures: list[str] = []
        if case.expected_final is not None and final != case.expected_final:
            failures.append("final mismatch")
        if case.expected_final_contains is not None and case.expected_final_contains not in final:
            failures.append("final missing expected text")
        if case.expected_tools and tools != case.expected_tools:
            failures.append("tool sequence mismatch")
        if case.expected_tool_prefix and tools[: len(case.expected_tool_prefix)] != case.expected_tool_prefix:
            failures.append("tool prefix mismatch")
        missing = [tool for tool in case.required_tools if tool not in tools]
        if missing:
            failures.append(f"missing required tools: {', '.join(missing)}")
        forbidden = [tool for tool in case.forbidden_tools if tool in tools]
        if forbidden:
            failures.append(f"forbidden tools used: {', '.join(forbidden)}")
        if case.min_steps is not None and len(tools) < case.min_steps:
            failures.append("too few tool steps")
        if case.max_steps is not None and len(tools) > case.max_steps:
            failures.append("too many tool steps")
        return failures

    def _score(self, failures: list[str]) -> float:
        return max(0.0, 1.0 - len(failures) * 0.25)
