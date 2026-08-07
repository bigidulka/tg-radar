from __future__ import annotations

import ast
import asyncio
import json
import time
import zipfile
from pathlib import Path
from typing import Any

from tg_radar.agent_core.models import AgentObservation, AgentPhase, ToolDangerLevel, ToolSpec
from tg_radar.rule_loader import load_rules


class NativeDevToolRegistry:
    def __init__(
        self,
        workspace: str | Path,
        rules: dict[str, Any] | None = None,
        enabled_tools: set[str] | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.rules = rules or load_rules("agent_tool_rules.json")["native_dev_tools"]
        self.enabled_tools = enabled_tools
        self.tool_rules = {item["name"]: item for item in self.rules.get("tools", [])}
        self.exclude_dirs = set(self.rules.get("exclude_dirs", []))
        self.protected_paths = set(self.rules.get("protected_paths", []))
        self.blocked_shell_commands = set(self.rules.get("blocked_shell_commands", []))

    def specs(self) -> list[ToolSpec]:
        specs = [
            ToolSpec(
                name=item["name"],
                risk_level=str(item.get("risk_level") or "safe"),
                description=str(item.get("description") or ""),
                args_schema=dict(item.get("args_schema") or {}),
                danger_level=ToolDangerLevel(str(item.get("danger_level") or "safe")),
                timeout_sec=int(item.get("timeout_sec") or 30),
                requires_approval=bool(item.get("requires_approval") or False),
                side_effects=bool(item.get("side_effects") or False),
                allowed_phases=[AgentPhase(str(phase)) for phase in item.get("allowed_phases") or []],
            )
            for item in self.rules.get("tools", [])
        ]
        if self.enabled_tools is None:
            return specs
        return [tool for tool in specs if tool.name in self.enabled_tools]

    async def run(self, name: str, args: dict[str, Any]) -> AgentObservation:
        if self.enabled_tools is not None and name not in self.enabled_tools:
            return AgentObservation(tool_name=name, ok=False, summary="tool disabled")
        if name not in self.tool_rules:
            return AgentObservation(tool_name=name, ok=False, summary="unknown tool")
        try:
            if name == "list_files":
                return self._list_files(args)
            if name == "read_file":
                return self._read_file(args)
            if name == "search_text":
                return self._search_text(args)
            if name == "search_symbols":
                return self._search_symbols(args)
            if name == "memory_read":
                return self._memory_read(args)
            if name == "memory_write":
                return self._memory_write(args)
            if name == "task_tracker":
                return self._task_tracker(args)
            if name == "checkpoint_create":
                return self._checkpoint_create(args)
            if name == "checkpoint_list":
                return self._checkpoint_list(args)
            if name == "checkpoint_restore":
                return self._checkpoint_restore(args)
            if name == "write_file":
                return self._write_file(args)
            if name == "apply_patch":
                return self._apply_patch(args)
            if name == "git_status":
                return await self._run_command(name, self._command(name), args)
            if name == "git_diff":
                return await self._git_diff(args)
            if name == "run_shell":
                return await self._run_shell(args)
            if name == "run_tests":
                return await self._run_command(name, self._command(name), args)
        except Exception as exc:
            return AgentObservation(tool_name=name, ok=False, summary=f"{type(exc).__name__}: {exc}")
        return AgentObservation(tool_name=name, ok=False, summary="unknown tool")

    def _list_files(self, args: dict[str, Any]) -> AgentObservation:
        root = self._safe_path(str(args.get("path") or "."))
        max_entries = int(args.get("max_entries") or self.rules.get("default_max_entries") or 200)
        entries: list[dict[str, str]] = []
        for path in sorted(root.rglob("*")):
            if self._is_excluded(path):
                continue
            entries.append({"path": self._relative(path), "type": "dir" if path.is_dir() else "file"})
            if len(entries) >= max_entries:
                break
        return AgentObservation(
            tool_name="list_files",
            ok=True,
            summary=f"listed {len(entries)} entries",
            data={"entries": entries},
        )

    def _read_file(self, args: dict[str, Any]) -> AgentObservation:
        path = self._safe_path(str(args.get("path") or ""))
        if not path.is_file():
            return AgentObservation(tool_name="read_file", ok=False, summary="file not found")
        max_bytes = int(self.rules.get("max_file_bytes") or 200000)
        if path.stat().st_size > max_bytes:
            return AgentObservation(tool_name="read_file", ok=False, summary="file too large")
        max_chars = int(args.get("max_chars") or self.rules.get("default_max_chars") or 12000)
        text = path.read_text(encoding="utf-8", errors="ignore")
        return AgentObservation(
            tool_name="read_file",
            ok=True,
            summary=f"read {self._relative(path)}",
            data={"path": self._relative(path), "text": text[:max_chars], "truncated": len(text) > max_chars},
        )

    def _search_text(self, args: dict[str, Any]) -> AgentObservation:
        query = str(args.get("query") or "")
        if not query:
            return AgentObservation(tool_name="search_text", ok=False, summary="query required")
        root = self._safe_path(str(args.get("path") or "."))
        needle = query.lower()
        max_matches = int(args.get("max_matches") or self.rules.get("default_max_matches") or 50)
        matches: list[dict[str, Any]] = []
        files = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
        for path in files:
            if self._is_excluded(path) or path.stat().st_size > int(self.rules.get("max_file_bytes") or 200000):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for index, line in enumerate(text.splitlines(), start=1):
                if needle in line.lower():
                    matches.append({"path": self._relative(path), "line": index, "text": line.strip()})
                    if len(matches) >= max_matches:
                        return self._search_observation(matches)
        return self._search_observation(matches)

    def _search_symbols(self, args: dict[str, Any]) -> AgentObservation:
        query = str(args.get("query") or "")
        if not query:
            return AgentObservation(tool_name="search_symbols", ok=False, summary="query required")
        root = self._safe_path(str(args.get("path") or "."))
        needle = query.casefold()
        max_matches = int(args.get("max_matches") or self.rules.get("default_max_matches") or 50)
        matches: list[dict[str, Any]] = []
        files = [root] if root.is_file() else sorted(path for path in root.rglob("*.py") if path.is_file())
        for path in files:
            if self._is_excluded(path) or path.stat().st_size > int(self.rules.get("max_file_bytes") or 200000):
                continue
            matches.extend(self._file_symbol_matches(path, needle))
            if len(matches) >= max_matches:
                matches = matches[:max_matches]
                break
        return AgentObservation(
            tool_name="search_symbols",
            ok=True,
            summary=f"symbols={len(matches)}",
            data={"matches": matches},
        )

    def _file_symbol_matches(self, path: Path, needle: str) -> list[dict[str, Any]]:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            return []
        matches: list[dict[str, Any]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                name = node.name
                if needle in name.casefold():
                    matches.append(
                        {
                            "path": self._relative(path),
                            "line": node.lineno,
                            "kind": self._symbol_kind(node),
                            "name": name,
                        }
                    )
        return matches

    def _memory_read(self, args: dict[str, Any]) -> AgentObservation:
        key = str(args.get("key") or "")
        max_items = int(args.get("max_items") or self.rules.get("default_max_matches") or 50)
        memory = self._load_json_list(self._state_file("memory_file"))
        items = [item for item in memory if not key or item.get("key") == key][:max_items]
        return AgentObservation(
            tool_name="memory_read",
            ok=True,
            summary=f"memory={len(items)}",
            data={"items": items},
        )

    def _memory_write(self, args: dict[str, Any]) -> AgentObservation:
        key = str(args.get("key") or "")
        if not key:
            return AgentObservation(tool_name="memory_write", ok=False, summary="key required")
        memory = self._load_json_list(self._state_file("memory_file"))
        entry = {
            "key": key,
            "value": args.get("value"),
            "tags": [str(item) for item in args.get("tags") or []],
        }
        memory = [item for item in memory if item.get("key") != key]
        memory.append(entry)
        self._save_json_list(self._state_file("memory_file"), memory)
        return AgentObservation(
            tool_name="memory_write",
            ok=True,
            summary=f"memory wrote {key}",
            data={"entry": entry},
        )

    def _task_tracker(self, args: dict[str, Any]) -> AgentObservation:
        action = str(args.get("action") or "")
        tasks = self._load_json_list(self._state_file("tasks_file"))
        if action == "list":
            max_items = int(args.get("max_items") or self.rules.get("default_max_matches") or 50)
            return AgentObservation(
                tool_name="task_tracker",
                ok=True,
                summary=f"tasks={len(tasks[:max_items])}",
                data={"tasks": tasks[:max_items]},
            )
        if action == "clear":
            self._save_json_list(self._state_file("tasks_file"), [])
            return AgentObservation(tool_name="task_tracker", ok=True, summary="tasks cleared", data={"tasks": []})
        if action not in {"add", "update"}:
            return AgentObservation(tool_name="task_tracker", ok=False, summary="unsupported action")
        task_id = str(args.get("id") or "")
        if not task_id:
            return AgentObservation(tool_name="task_tracker", ok=False, summary="id required")
        existing = {str(item.get("id")): item for item in tasks if item.get("id")}
        task = existing.get(task_id, {"id": task_id})
        if args.get("title") is not None:
            task["title"] = str(args.get("title"))
        if args.get("status") is not None:
            task["status"] = str(args.get("status"))
        if args.get("notes") is not None:
            task["notes"] = str(args.get("notes"))
        if "status" not in task:
            task["status"] = "pending"
        existing[task_id] = task
        saved = list(existing.values())
        self._save_json_list(self._state_file("tasks_file"), saved)
        return AgentObservation(
            tool_name="task_tracker",
            ok=True,
            summary=f"task {action} {task_id}",
            data={"task": task, "tasks": saved},
        )

    def _checkpoint_create(self, args: dict[str, Any]) -> AgentObservation:
        checkpoint_id = self._safe_id(str(args.get("id") or f"cp-{int(time.time() * 1000)}"))
        if not checkpoint_id:
            return AgentObservation(tool_name="checkpoint_create", ok=False, summary="id required")
        checkpoint_dir = self._checkpoint_dir()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        archive = checkpoint_dir / f"{checkpoint_id}.zip"
        if archive.exists():
            return AgentObservation(tool_name="checkpoint_create", ok=False, summary="checkpoint exists")
        files = self._checkpoint_files(args.get("paths") or ["."])
        manifest = {
            "id": checkpoint_id,
            "files": [self._relative(path) for path in files],
            "metadata": args.get("metadata") if isinstance(args.get("metadata"), dict) else {},
        }
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
            zipped.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            for path in files:
                zipped.write(path, self._relative(path))
        return AgentObservation(
            tool_name="checkpoint_create",
            ok=True,
            summary=f"checkpoint created {checkpoint_id} files={len(files)}",
            data={"id": checkpoint_id, "path": self._relative(archive), "files": len(files)},
        )

    def _checkpoint_list(self, args: dict[str, Any]) -> AgentObservation:
        checkpoints: list[dict[str, Any]] = []
        checkpoint_dir = self._checkpoint_dir()
        if checkpoint_dir.is_dir():
            for archive in sorted(checkpoint_dir.glob("*.zip")):
                item = {"id": archive.stem, "path": self._relative(archive), "bytes": archive.stat().st_size}
                manifest = self._checkpoint_manifest(archive)
                if manifest:
                    item["files"] = len(manifest.get("files") or [])
                    item["metadata"] = manifest.get("metadata") or {}
                checkpoints.append(item)
        return AgentObservation(
            tool_name="checkpoint_list",
            ok=True,
            summary=f"checkpoints={len(checkpoints)}",
            data={"checkpoints": checkpoints},
        )

    def _checkpoint_restore(self, args: dict[str, Any]) -> AgentObservation:
        checkpoint_id = self._safe_id(str(args.get("id") or ""))
        if not checkpoint_id:
            return AgentObservation(tool_name="checkpoint_restore", ok=False, summary="id required")
        archive = self._checkpoint_dir() / f"{checkpoint_id}.zip"
        if not archive.is_file():
            return AgentObservation(tool_name="checkpoint_restore", ok=False, summary="checkpoint not found")
        restored: list[str] = []
        with zipfile.ZipFile(archive) as zipped:
            for name in zipped.namelist():
                if name == "manifest.json" or name.endswith("/"):
                    continue
                target = self._safe_path(name)
                if self._is_protected(target):
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zipped.read(name))
                restored.append(self._relative(target))
        return AgentObservation(
            tool_name="checkpoint_restore",
            ok=True,
            summary=f"checkpoint restored {checkpoint_id} files={len(restored)}",
            data={"id": checkpoint_id, "restored": restored},
        )

    def _write_file(self, args: dict[str, Any]) -> AgentObservation:
        path = self._safe_path(str(args.get("path") or ""))
        if self._is_protected(path):
            return AgentObservation(tool_name="write_file", ok=False, summary="protected path")
        content = str(args.get("content") or "")
        max_bytes = int(self.rules.get("max_file_bytes") or 200000)
        if len(content.encode("utf-8")) > max_bytes:
            return AgentObservation(tool_name="write_file", ok=False, summary="content too large")
        if bool(args.get("create_dirs") or False):
            path.parent.mkdir(parents=True, exist_ok=True)
        if not path.parent.is_dir():
            return AgentObservation(tool_name="write_file", ok=False, summary="parent directory not found")
        path.write_text(content, encoding="utf-8")
        return AgentObservation(
            tool_name="write_file",
            ok=True,
            summary=f"wrote {self._relative(path)}",
            data={"path": self._relative(path), "bytes": len(content.encode("utf-8"))},
        )

    def _apply_patch(self, args: dict[str, Any]) -> AgentObservation:
        path = self._safe_path(str(args.get("path") or ""))
        if self._is_protected(path):
            return AgentObservation(tool_name="apply_patch", ok=False, summary="protected path")
        if not path.is_file():
            return AgentObservation(tool_name="apply_patch", ok=False, summary="file not found")
        old_text = str(args.get("old_text") or "")
        new_text = str(args.get("new_text") or "")
        if not old_text:
            return AgentObservation(tool_name="apply_patch", ok=False, summary="old_text required")
        text = path.read_text(encoding="utf-8", errors="ignore")
        matches = text.count(old_text)
        if not matches:
            return AgentObservation(tool_name="apply_patch", ok=False, summary="old_text not found")
        max_replacements = int(args.get("max_replacements") or 1)
        updated = text.replace(old_text, new_text, max_replacements)
        path.write_text(updated, encoding="utf-8")
        return AgentObservation(
            tool_name="apply_patch",
            ok=True,
            summary=f"patched {self._relative(path)} replacements={min(matches, max_replacements)}",
            data={
                "path": self._relative(path),
                "matches": matches,
                "replacements": min(matches, max_replacements),
            },
        )

    async def _git_diff(self, args: dict[str, Any]) -> AgentObservation:
        command = list(self._command("git_diff"))
        path = args.get("path")
        if path:
            command.append(self._relative(self._safe_path(str(path))))
        return await self._run_command("git_diff", command, args)

    async def _run_command(self, name: str, command: list[str], args: dict[str, Any]) -> AgentObservation:
        timeout = int(self.tool_rules[name].get("timeout_sec") or 30)
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=self.workspace,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        max_chars = int(args.get("max_chars") or self.rules.get("default_max_chars") or 12000)
        out = stdout.decode(errors="ignore")
        err = stderr.decode(errors="ignore")
        text = "\n".join(part for part in [out, err] if part).strip()
        return AgentObservation(
            tool_name=name,
            ok=process.returncode == 0,
            summary=f"{name} exit={process.returncode}",
            data={
                "returncode": process.returncode,
                "output": text[:max_chars],
                "truncated": len(text) > max_chars,
            },
        )

    async def _run_shell(self, args: dict[str, Any]) -> AgentObservation:
        raw_command = args.get("command")
        if not isinstance(raw_command, list) or not raw_command:
            return AgentObservation(tool_name="run_shell", ok=False, summary="command list required")
        command = [str(item) for item in raw_command]
        if self._is_blocked_command(command[0]):
            return AgentObservation(tool_name="run_shell", ok=False, summary="blocked command")
        return await self._run_command("run_shell", command, args)

    def _command(self, name: str) -> list[str]:
        return [str(item) for item in self.rules.get("commands", {}).get(name, [])]

    def _search_observation(self, matches: list[dict[str, Any]]) -> AgentObservation:
        return AgentObservation(
            tool_name="search_text",
            ok=True,
            summary=f"matches={len(matches)}",
            data={"matches": matches},
        )

    def _symbol_kind(self, node: ast.AST) -> str:
        if isinstance(node, ast.ClassDef):
            return "class"
        if isinstance(node, ast.AsyncFunctionDef):
            return "async_function"
        return "function"

    def _state_file(self, rule_key: str) -> Path:
        state_dir = self._safe_path(str(self.rules.get("state_dir") or ".tg-radar-agent"))
        return self._safe_path(f"{self._relative(state_dir)}/{self.rules.get(rule_key)}")

    def _checkpoint_dir(self) -> Path:
        state_dir = self._safe_path(str(self.rules.get("state_dir") or ".tg-radar-agent"))
        checkpoint_dir = str(self.rules.get("checkpoint_dir") or "checkpoints")
        return self._safe_path(f"{self._relative(state_dir)}/{checkpoint_dir}")

    def _checkpoint_files(self, raw_paths: Any) -> list[Path]:
        roots = raw_paths if isinstance(raw_paths, list) else [raw_paths]
        max_bytes = int(self.rules.get("max_checkpoint_file_bytes") or self.rules.get("max_file_bytes") or 200000)
        files: list[Path] = []
        for raw in roots:
            root = self._safe_path(str(raw or "."))
            candidates = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
            for path in candidates:
                if path.is_symlink() or self._is_excluded(path) or self._is_protected(path) or self._is_state_path(path):
                    continue
                if path.stat().st_size > max_bytes:
                    continue
                files.append(path)
        return list(dict.fromkeys(files))

    def _checkpoint_manifest(self, archive: Path) -> dict[str, Any]:
        try:
            with zipfile.ZipFile(archive) as zipped:
                return json.loads(zipped.read("manifest.json").decode("utf-8"))
        except (KeyError, OSError, json.JSONDecodeError, zipfile.BadZipFile):
            return {}

    def _load_json_list(self, path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return []
        return [item for item in data if isinstance(item, dict)]

    def _save_json_list(self, path: Path, items: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    def _safe_path(self, value: str) -> Path:
        candidate = Path(value)
        path = candidate.resolve() if candidate.is_absolute() else (self.workspace / candidate).resolve()
        if path != self.workspace and self.workspace not in path.parents:
            raise ValueError("path outside workspace")
        return path

    def _is_excluded(self, path: Path) -> bool:
        try:
            parts = path.resolve().relative_to(self.workspace).parts
        except ValueError:
            return True
        return any(part in self.exclude_dirs for part in parts)

    def _is_protected(self, path: Path) -> bool:
        try:
            relative = path.resolve().relative_to(self.workspace)
        except ValueError:
            return True
        parts = relative.parts
        value = relative.as_posix()
        return any(rule == value or rule in parts for rule in self.protected_paths)

    def _is_blocked_command(self, executable: str) -> bool:
        return Path(executable).name in self.blocked_shell_commands

    def _is_state_path(self, path: Path) -> bool:
        state_dir = self._safe_path(str(self.rules.get("state_dir") or ".tg-radar-agent"))
        resolved = path.resolve()
        return resolved == state_dir or state_dir in resolved.parents

    def _safe_id(self, value: str) -> str:
        allowed = []
        for char in value.strip():
            if char.isalnum() or char in {"-", "_", "."}:
                allowed.append(char)
        return "".join(allowed)[:80]

    def _relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.workspace).as_posix()
