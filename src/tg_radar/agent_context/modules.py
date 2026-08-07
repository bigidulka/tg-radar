from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from typing import Protocol

from tg_radar.agent_core.models import AgentContextPack, AgentRequest, AgentStep, ToolSpec

TEXT_SEPARATORS = (
    " ",
    "\n",
    "\t",
    ".",
    ",",
    ":",
    ";",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    "/",
    "\\",
    "-",
    "_",
    '"',
    "'",
)


class ContextModule(Protocol):
    async def build(
        self,
        request: AgentRequest,
        tools: list[ToolSpec],
        steps: list[AgentStep],
    ) -> AgentContextPack:
        ...


class ModularContextBuilder:
    def __init__(self, modules: list[ContextModule]) -> None:
        self.modules = modules

    async def build(
        self,
        request: AgentRequest,
        tools: list[ToolSpec],
        steps: list[AgentStep],
    ) -> AgentContextPack:
        context = AgentContextPack(user_goal=request.goal)
        for module in self.modules:
            context = merge_context(context, await module.build(request, tools, steps))
        return context


class TaskStateModule:
    async def build(
        self,
        request: AgentRequest,
        tools: list[ToolSpec],
        steps: list[AgentStep],
    ) -> AgentContextPack:
        return AgentContextPack(
            user_goal=request.goal,
            task_state={
                "mode": request.mode,
                "inputs": request.inputs,
                "steps_count": len(steps),
            },
        )


class RecentObservationsModule:
    def __init__(self, limit: int) -> None:
        self.limit = limit

    async def build(
        self,
        request: AgentRequest,
        tools: list[ToolSpec],
        steps: list[AgentStep],
    ) -> AgentContextPack:
        return AgentContextPack(
            user_goal=request.goal,
            recent_observations=[step.model_dump(mode="json") for step in steps[-self.limit :]],
        )


class ToolManifestModule:
    async def build(
        self,
        request: AgentRequest,
        tools: list[ToolSpec],
        steps: list[AgentStep],
    ) -> AgentContextPack:
        return AgentContextPack(user_goal=request.goal, tool_manifest=tools)


class ConstraintsModule:
    def __init__(self, context_budget: int, strategy: str) -> None:
        self.context_budget = context_budget
        self.strategy = strategy

    async def build(
        self,
        request: AgentRequest,
        tools: list[ToolSpec],
        steps: list[AgentStep],
    ) -> AgentContextPack:
        return AgentContextPack(
            user_goal=request.goal,
            constraints={
                "max_steps": request.max_steps,
                "runtime_contract": "structured_action_only",
            },
            token_budget_report={
                "context_budget": self.context_budget,
                "strategy": self.strategy,
            },
        )


class WorkspaceRepoMapModule:
    def __init__(
        self,
        workspace: str | Path,
        exclude_dirs: set[str],
        max_entries: int,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.exclude_dirs = exclude_dirs
        self.max_entries = max_entries

    async def build(
        self,
        request: AgentRequest,
        tools: list[ToolSpec],
        steps: list[AgentStep],
    ) -> AgentContextPack:
        entries: list[dict[str, Any]] = []
        if not self.workspace.exists():
            return AgentContextPack(user_goal=request.goal, repo_map=entries)
        for path in sorted(self.workspace.rglob("*")):
            if self._is_excluded(path):
                continue
            entries.append(self._entry(path))
            if len(entries) >= self.max_entries:
                break
        return AgentContextPack(user_goal=request.goal, repo_map=entries)

    def _entry(self, path: Path) -> dict[str, Any]:
        item = {"path": self._relative(path), "type": "dir" if path.is_dir() else "file"}
        if path.is_file():
            item["size"] = path.stat().st_size
        return item

    def _is_excluded(self, path: Path) -> bool:
        try:
            parts = path.resolve().relative_to(self.workspace).parts
        except ValueError:
            return True
        return any(part in self.exclude_dirs for part in parts)

    def _relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.workspace).as_posix()


class RelevantFilesModule:
    def __init__(
        self,
        workspace: str | Path,
        exclude_dirs: set[str],
        max_files: int,
        max_chars: int,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.exclude_dirs = exclude_dirs
        self.max_files = max_files
        self.max_chars = max_chars

    async def build(
        self,
        request: AgentRequest,
        tools: list[ToolSpec],
        steps: list[AgentStep],
    ) -> AgentContextPack:
        terms = self._terms(request)
        if not terms or not self.workspace.exists():
            return AgentContextPack(user_goal=request.goal, relevant_files=[])
        ranked = self._ranked_files(terms)
        files = [self._snippet(path, terms) for _, path in ranked[: self.max_files]]
        return AgentContextPack(user_goal=request.goal, relevant_files=files)

    def _ranked_files(self, terms: list[str]) -> list[tuple[int, Path]]:
        ranked: list[tuple[int, Path]] = []
        for path in sorted(item for item in self.workspace.rglob("*") if item.is_file()):
            if self._is_excluded(path):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            lowered = text.casefold()
            score = sum(lowered.count(term) for term in terms)
            if score:
                ranked.append((score, path))
        ranked.sort(key=lambda item: (-item[0], self._relative(item[1])))
        return ranked

    def _snippet(self, path: Path, terms: list[str]) -> dict[str, Any]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        matched: list[dict[str, Any]] = []
        for index, line in enumerate(lines, start=1):
            lowered = line.casefold()
            if any(term in lowered for term in terms):
                matched.append({"line": index, "text": line[: self.max_chars]})
            if len(matched) >= self.max_files:
                break
        return {
            "path": self._relative(path),
            "matches": matched,
            "preview": text[: self.max_chars],
            "truncated": len(text) > self.max_chars,
        }

    def _terms(self, request: AgentRequest) -> list[str]:
        raw = [request.goal, *self._input_strings(request.inputs)]
        terms: list[str] = []
        for value in raw:
            for token in self._split_text(value):
                lowered = token.casefold()
                if len(lowered) >= 3 and lowered not in terms:
                    terms.append(lowered)
        return terms[:20]

    def _split_text(self, value: str) -> list[str]:
        text = value
        for separator in TEXT_SEPARATORS:
            text = text.replace(separator, " ")
        return [part for part in text.split(" ") if part]

    def _input_strings(self, value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            items: list[str] = []
            for key, item in value.items():
                items.extend(self._input_strings(str(key)))
                items.extend(self._input_strings(item))
            return items
        if isinstance(value, list | tuple | set):
            items = []
            for item in value:
                items.extend(self._input_strings(item))
            return items
        return []

    def _is_excluded(self, path: Path) -> bool:
        try:
            parts = path.resolve().relative_to(self.workspace).parts
        except ValueError:
            return True
        return any(part in self.exclude_dirs for part in parts)

    def _relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.workspace).as_posix()


class CurrentDiffModule:
    def __init__(self, workspace: str | Path, max_chars: int) -> None:
        self.workspace = Path(workspace).resolve()
        self.max_chars = max_chars

    async def build(
        self,
        request: AgentRequest,
        tools: list[ToolSpec],
        steps: list[AgentStep],
    ) -> AgentContextPack:
        if not self.workspace.exists():
            return AgentContextPack(user_goal=request.goal)
        try:
            process = await asyncio.create_subprocess_exec(
                "git",
                "diff",
                "--",
                cwd=self.workspace,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            return AgentContextPack(user_goal=request.goal, current_diff=f"{type(exc).__name__}: {exc}")
        stdout, stderr = await process.communicate()
        text = stdout.decode(errors="ignore") if process.returncode == 0 else stderr.decode(errors="ignore")
        return AgentContextPack(user_goal=request.goal, current_diff=text[: self.max_chars])


class AgentMemoryModule:
    def __init__(
        self,
        workspace: str | Path,
        state_dir: str,
        memory_file: str,
        tasks_file: str,
        max_items: int,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.state_dir = state_dir
        self.memory_file = memory_file
        self.tasks_file = tasks_file
        self.max_items = max_items

    async def build(
        self,
        request: AgentRequest,
        tools: list[ToolSpec],
        steps: list[AgentStep],
    ) -> AgentContextPack:
        state_path = self._safe_path(self.state_dir)
        memory = self._read_items(state_path / self.memory_file)
        tasks = self._read_items(state_path / self.tasks_file)
        return AgentContextPack(
            user_goal=request.goal,
            long_term_memory=memory[: self.max_items],
            task_board=tasks[: self.max_items],
        )

    def _read_items(self, path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []
        return [item for item in data if isinstance(item, dict)]

    def _safe_path(self, value: str) -> Path:
        candidate = Path(value)
        path = candidate.resolve() if candidate.is_absolute() else (self.workspace / candidate).resolve()
        if path != self.workspace and self.workspace not in path.parents:
            return self.workspace
        return path


def merge_context(left: AgentContextPack, right: AgentContextPack) -> AgentContextPack:
    return AgentContextPack(
        system_rules=right.system_rules or left.system_rules,
        user_goal=right.user_goal or left.user_goal,
        task_state=left.task_state | right.task_state,
        domain_context=left.domain_context | right.domain_context,
        repo_map=[*left.repo_map, *right.repo_map],
        relevant_files=[*left.relevant_files, *right.relevant_files],
        current_diff=right.current_diff or left.current_diff,
        long_term_memory=[*left.long_term_memory, *right.long_term_memory],
        task_board=[*left.task_board, *right.task_board],
        recent_observations=[*left.recent_observations, *right.recent_observations],
        tool_manifest=[*left.tool_manifest, *right.tool_manifest],
        constraints=left.constraints | right.constraints,
        token_budget_report=left.token_budget_report | right.token_budget_report,
    )
