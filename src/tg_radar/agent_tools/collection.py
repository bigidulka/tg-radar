from __future__ import annotations

from typing import Any

from tg_radar.agent_core.models import AgentObservation, ToolSpec
from tg_radar.auto_search import AutoSearchEngine
from tg_radar.discovery import Discoverer
from tg_radar.schemas import DiscoverResponse, IngestRequest, SearchTaskRequest
from tg_radar.service import IngestService


class CollectionToolRegistry:
    def __init__(
        self,
        discoverer: Discoverer,
        ingesting: IngestService,
        auto_search: AutoSearchEngine,
        enabled_tools: set[str] | None = None,
    ) -> None:
        self.discoverer = discoverer
        self.ingesting = ingesting
        self.auto_search = auto_search
        self.enabled_tools = enabled_tools

    def specs(self) -> list[ToolSpec]:
        specs = [
            ToolSpec(
                name="core_status",
                risk_level="safe",
                description="Read collection core and task engine status.",
            ),
            ToolSpec(
                name="discover",
                risk_level="network",
                description="Discover Telegram channel candidates without crawling messages.",
                args_schema=self._collection_args_schema(include_crawl=False),
            ),
            ToolSpec(
                name="ingest",
                risk_level="network_write",
                description="Discover, crawl public Telegram pages, store messages, feed search index.",
                args_schema=self._collection_args_schema(include_crawl=True),
            ),
            ToolSpec(
                name="create_task",
                risk_level="db_write",
                description="Create or update autonomous recurring collection task.",
                args_schema=self._task_args_schema(),
            ),
            ToolSpec(
                name="run_task",
                risk_level="network_write",
                description="Run one existing collection task immediately.",
                args_schema={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
            ),
        ]
        if self.enabled_tools is None:
            return specs
        return [tool for tool in specs if tool.name in self.enabled_tools]

    async def run(self, name: str, args: dict[str, Any]) -> AgentObservation:
        if self.enabled_tools is not None and name not in self.enabled_tools:
            return AgentObservation(tool_name=name, ok=False, summary="tool disabled")
        try:
            if name == "core_status":
                return AgentObservation(
                    tool_name=name,
                    ok=True,
                    summary="core status loaded",
                    data={
                        "engine_running": self.auto_search.running,
                        "active_tasks": self.auto_search.active_count,
                        "tasks": [task.model_dump(mode="json") for task in await self.auto_search.list_tasks()],
                    },
                )
            if name == "discover":
                candidates = await self.discoverer.discover(
                    args.get("keywords", []),
                    args.get("seed_channels", []),
                    int(args.get("depth", 1)),
                    int(args.get("limit", 50)),
                )
                return AgentObservation(
                    tool_name=name,
                    ok=True,
                    summary=f"discovered {len(candidates)} candidates",
                    data=DiscoverResponse(candidates=candidates).model_dump(mode="json"),
                )
            if name == "ingest":
                payload = IngestRequest(**args)
                response = await self.ingesting.ingest(
                    payload.keywords,
                    payload.seed_channels,
                    payload.depth,
                    payload.limit,
                    payload.pages_per_channel,
                    payload.crawl,
                    task_name=payload.topic or payload.task_name,
                    max_live_crawl=payload.max_live_crawl,
                    crawl_mode=payload.crawl_mode.value,
                    freshness_days=payload.freshness_days,
                    since=payload.since,
                )
                return AgentObservation(
                    tool_name=name,
                    ok=True,
                    summary=(
                        f"candidates={response.candidates_found} "
                        f"crawled={response.channels_crawled} "
                        f"messages={response.messages_saved + response.cached_messages_used}"
                    ),
                    data=response.model_dump(mode="json"),
                )
            if name == "create_task":
                payload = SearchTaskRequest(**args)
                status = await self.auto_search.upsert_task(payload)
                return AgentObservation(
                    tool_name=name,
                    ok=True,
                    summary=f"task {status.name} saved",
                    data=status.model_dump(mode="json"),
                )
            if name == "run_task":
                task_name = str(args.get("name") or "")
                result = await self.auto_search.run_task_now(task_name)
                if not result:
                    return AgentObservation(
                        tool_name=name,
                        ok=False,
                        summary=f"task {task_name} not found",
                    )
                return AgentObservation(
                    tool_name=name,
                    ok=True,
                    summary=f"task {task_name} run completed",
                    data=result.model_dump(mode="json"),
                )
        except Exception as exc:
            return AgentObservation(
                tool_name=name,
                ok=False,
                summary=f"{type(exc).__name__}: {exc}",
            )
        return AgentObservation(tool_name=name, ok=False, summary="unknown tool")

    def _collection_args_schema(self, include_crawl: bool) -> dict[str, Any]:
        properties: dict[str, Any] = {
            "keywords": {"type": "array", "items": {"type": "string"}},
            "seed_channels": {"type": "array", "items": {"type": "string"}},
            "topic": {"type": "string"},
            "depth": {"type": "integer", "minimum": 0, "maximum": 5},
            "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
            "pages_per_channel": {"type": "integer", "minimum": 1, "maximum": 20},
            "crawl_mode": {"type": "string", "enum": ["refresh", "backfill"]},
            "freshness_days": {"type": "integer", "minimum": 1, "maximum": 3650},
            "since": {"type": "string", "format": "date-time"},
            "max_live_crawl": {"type": "integer", "minimum": 0, "maximum": 1000},
        }
        if include_crawl:
            properties["crawl"] = {"type": "boolean"}
        return {"type": "object", "properties": properties}

    def _task_args_schema(self) -> dict[str, Any]:
        schema = self._collection_args_schema(include_crawl=True)
        schema["properties"] |= {
            "name": {"type": "string"},
            "interval_seconds": {"type": "integer", "minimum": 30, "maximum": 86400},
            "enabled": {"type": "boolean"},
        }
        schema["required"] = ["name"]
        return schema
