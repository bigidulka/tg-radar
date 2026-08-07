from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from tg_radar.agent_core.models import AgentContextPack, AgentRequest, AgentStep, ToolSpec
from tg_radar.agent_context.modules import (
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
from tg_radar.auto_search import AutoSearchEngine
from tg_radar.db import Channel, DiscoveredChannel, Message, SearchTask


class CollectionContextBuilder:
    def __init__(
        self,
        sessionmaker: async_sessionmaker[AsyncSession],
        auto_search: AutoSearchEngine,
        context_budget: int = 120000,
        recent_steps_limit: int = 4,
        recent_tasks_limit: int = 8,
        workspace_context_enabled: bool = False,
        workspace: str = ".",
        workspace_exclude_dirs: set[str] | None = None,
        workspace_repo_map_enabled: bool = True,
        workspace_retrieval_enabled: bool = True,
        workspace_git_diff_enabled: bool = True,
        workspace_repo_map_max_entries: int = 200,
        workspace_relevant_files_limit: int = 5,
        workspace_relevant_file_chars: int = 4000,
        workspace_diff_chars: int = 12000,
        memory_context_enabled: bool = True,
        memory_state_dir: str = ".tg-radar-agent",
        memory_file: str = "memory.json",
        tasks_file: str = "tasks.json",
        memory_context_max_items: int = 20,
    ) -> None:
        self.sessionmaker = sessionmaker
        self.auto_search = auto_search
        self.context_budget = context_budget
        self.recent_steps_limit = recent_steps_limit
        self.recent_tasks_limit = recent_tasks_limit
        modules = [
            TaskStateModule(),
            RecentObservationsModule(recent_steps_limit),
            ToolManifestModule(),
            ConstraintsModule(context_budget, "bounded_recent_steps"),
        ]
        if workspace_context_enabled:
            excludes = workspace_exclude_dirs or set()
            if workspace_repo_map_enabled:
                modules.append(WorkspaceRepoMapModule(workspace, excludes, workspace_repo_map_max_entries))
            if workspace_retrieval_enabled:
                modules.append(
                    RelevantFilesModule(
                        workspace,
                        excludes,
                        workspace_relevant_files_limit,
                        workspace_relevant_file_chars,
                    )
                )
            if workspace_git_diff_enabled:
                modules.append(CurrentDiffModule(workspace, workspace_diff_chars))
        if memory_context_enabled:
            modules.append(
                AgentMemoryModule(
                    workspace,
                    memory_state_dir,
                    memory_file,
                    tasks_file,
                    memory_context_max_items,
                )
            )
        self.base_builder = ModularContextBuilder(modules)

    async def build(
        self,
        request: AgentRequest,
        tools: list[ToolSpec],
        steps: list[AgentStep],
    ) -> AgentContextPack:
        stats = await self._stats()
        tasks = await self.auto_search.list_tasks()
        recent_tasks = sorted(tasks, key=lambda task: self._task_ts(task.last_run_at or task.next_run_at), reverse=True)[
            : self.recent_tasks_limit
        ]
        context = await self.base_builder.build(request, tools, steps)
        return context.model_copy(
            update={
                "domain_context": context.domain_context | {
                    "stats": stats,
                    "recent_tasks": [task.model_dump(mode="json") for task in recent_tasks],
                },
            }
        )

    async def _stats(self) -> dict[str, Any]:
        async with self.sessionmaker() as session:
            candidates = await session.scalar(select(func.count()).select_from(DiscoveredChannel))
            channels = await session.scalar(select(func.count()).select_from(Channel))
            messages = await session.scalar(select(func.count()).select_from(Message))
            running_tasks = await session.scalar(
                select(func.count()).select_from(SearchTask).where(SearchTask.running.is_(True))
            )
        return {
            "candidates": int(candidates or 0),
            "channels": int(channels or 0),
            "messages": int(messages or 0),
            "running_tasks": int(running_tasks or 0),
            "engine_running": self.auto_search.running,
            "active_tasks": self.auto_search.active_count,
        }

    def _task_ts(self, value: datetime | None) -> float:
        return value.timestamp() if value else 0.0
