from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class AgentActionType(StrEnum):
    tool_call = "tool_call"
    plan_update = "plan_update"
    final = "final"
    ask_user = "ask_user"
    delegate = "delegate"
    stop = "stop"


class AgentPhase(StrEnum):
    understand = "understand"
    plan = "plan"
    edit = "edit"
    verify = "verify"
    fix = "fix"
    finalize = "finalize"


class ToolDangerLevel(StrEnum):
    safe = "safe"
    write = "write"
    shell = "shell"
    network = "network"
    secret = "secret"
    destructive = "destructive"


class ToolSpec(BaseModel):
    name: str
    risk_level: str = "safe"
    description: str
    args_schema: dict[str, Any] = Field(default_factory=dict)
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] | None = None
    side_effects: bool = False
    danger_level: ToolDangerLevel = ToolDangerLevel.safe
    timeout_sec: int = Field(default=30, ge=1, le=3600)
    requires_approval: bool = False
    allowed_phases: list[AgentPhase] = Field(default_factory=list)

    @model_validator(mode="after")
    def sync_legacy_fields(self) -> "ToolSpec":
        if not self.input_schema and self.args_schema:
            self.input_schema = dict(self.args_schema)
        if self.danger_level == ToolDangerLevel.safe:
            risk = self.risk_level.lower()
            if "secret" in risk:
                self.danger_level = ToolDangerLevel.secret
            elif "destructive" in risk or "delete" in risk:
                self.danger_level = ToolDangerLevel.destructive
            elif "shell" in risk:
                self.danger_level = ToolDangerLevel.shell
            elif "network" in risk:
                self.danger_level = ToolDangerLevel.network
            elif "write" in risk:
                self.danger_level = ToolDangerLevel.write
        return self


class AgentRequest(BaseModel):
    goal: str
    mode: str = "auto"
    inputs: dict[str, Any] = Field(default_factory=dict)
    max_steps: int = Field(default=6, ge=1, le=80)


class AgentAction(BaseModel):
    type: AgentActionType
    tool_name: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    reason: str
    expected_result: str | None = None


class AgentObservation(BaseModel):
    tool_name: str
    ok: bool
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)


class AgentStep(BaseModel):
    step: int
    action: AgentAction
    observation: AgentObservation | None = None


class AgentContextPack(BaseModel):
    system_rules: str = ""
    user_goal: str
    task_state: dict[str, Any] = Field(default_factory=dict)
    domain_context: dict[str, Any] = Field(default_factory=dict)
    repo_map: list[dict[str, Any]] = Field(default_factory=list)
    relevant_files: list[dict[str, Any]] = Field(default_factory=list)
    current_diff: str = ""
    long_term_memory: list[dict[str, Any]] = Field(default_factory=list)
    task_board: list[dict[str, Any]] = Field(default_factory=list)
    recent_observations: list[dict[str, Any]] = Field(default_factory=list)
    tool_manifest: list[ToolSpec] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)
    token_budget_report: dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    final: str
    steps: list[AgentStep]
    context: dict[str, Any] = Field(default_factory=dict)


class AgentRunState(BaseModel):
    request: AgentRequest
    phase: AgentPhase = AgentPhase.understand
    steps: list[AgentStep] = Field(default_factory=list)


class AgentEvent(BaseModel):
    run_id: str
    step: AgentStep
