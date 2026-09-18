from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from tg_radar.db import ResearchTopic, SearchTask, TaskRun, task_to_status, upsert_search_task
from tg_radar.schemas import IngestResponse, SearchTaskRequest, SearchTaskRunResponse, SearchTaskStatus
from tg_radar.service import IngestService


def auto_policy(task: SearchTask) -> tuple[int, int, int, int]:
    if not getattr(task, "auto_tune", True):
        return task.depth, task.limit, task.pages_per_channel, task.interval_seconds
    if task.total_runs == 0:
        return 1, 40, 1, 180
    if task.last_candidates_found == 0:
        return 0, 80, 1, 900
    if task.last_messages_saved > 20:
        return 1, 60, 2, 180
    if task.last_channels_crawled == 0:
        return 0, 30, 1, 300
    return 1, 40, 1, 300


def empty_run_error(ingest) -> str | None:
    """Why a run that found no candidates failed, or None if it simply found none.

    Finding nothing is a result, not a failure: only the source reporting an
    error is one. This used to synthesise an error whenever the candidate count
    was zero, which made a healthy run look broken -- and got worse as coverage
    improved, because a well-covered topic legitimately returns zero new
    candidates most of the time.
    """
    if ingest is None or ingest.candidates_found or not ingest.errors:
        return None
    return "; ".join(ingest.errors[:3])


class AutoSearchEngine:
    def __init__(
        self,
        sessionmaker: async_sessionmaker[AsyncSession],
        ingesting: IngestService,
        tick_seconds: float,
        task_timeout_seconds: float,
    ) -> None:
        self.sessionmaker = sessionmaker
        self.ingesting = ingesting
        self.tick_seconds = tick_seconds
        self.task_timeout_seconds = task_timeout_seconds
        self._task: asyncio.Task | None = None
        self._active_tasks: set[asyncio.Task] = set()
        self._stop = asyncio.Event()

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def active_count(self) -> int:
        return len(self._active_tasks)

    def start(self) -> None:
        if self.running:
            return
        self._stop = asyncio.Event()
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if not self._task:
            return
        self._stop.set()
        await asyncio.wait([self._task], timeout=5)
        if self._active_tasks:
            await asyncio.wait(self._active_tasks, timeout=min(self.task_timeout_seconds, 30))
        self._task = None

    async def upsert_task(self, payload: SearchTaskRequest) -> SearchTaskStatus:
        async with self.sessionmaker() as session:
            async with session.begin():
                task = await upsert_search_task(session, payload)
                return task_to_status(task)

    async def list_tasks(self) -> list[SearchTaskStatus]:
        async with self.sessionmaker() as session:
            rows = (await session.execute(select(SearchTask).order_by(SearchTask.name))).scalars().all()
            return [task_to_status(row) for row in rows]

    async def get_task(self, name: str) -> SearchTaskStatus | None:
        async with self.sessionmaker() as session:
            task = await session.scalar(select(SearchTask).where(SearchTask.name == name))
            return task_to_status(task) if task else None

    async def run_task_now(self, name: str) -> SearchTaskRunResponse | None:
        async with self.sessionmaker() as session:
            task = await session.scalar(select(SearchTask).where(SearchTask.name == name))
            if not task:
                return None
            status, ingest = await self._run_task(task)
            return SearchTaskRunResponse(task=status, ingest=ingest)

    async def _loop(self) -> None:
        await self._reset_running_tasks()
        while not self._stop.is_set():
            try:
                await self._run_due_tasks()
            except Exception:
                pass
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.tick_seconds)
            except asyncio.TimeoutError:
                continue

    async def _reset_running_tasks(self) -> None:
        async with self.sessionmaker() as session:
            async with session.begin():
                rows = (await session.execute(select(SearchTask).where(SearchTask.running.is_(True)))).scalars().all()
                for task in rows:
                    task.running = False
                    task.running_started_at = None
                    task.last_error = "failed stale running task on worker startup"
                    session.add(TaskRun(task_name=task.name, status="failed", finished_at=datetime.utcnow(), error=task.last_error))

    async def _run_due_tasks(self) -> None:
        now = datetime.utcnow()
        async with self.sessionmaker() as session:
            tasks = (
                await session.execute(
                    select(SearchTask)
                    .where(SearchTask.enabled.is_(True), SearchTask.running.is_(False))
                    .where((SearchTask.next_run_at.is_(None)) | (SearchTask.next_run_at <= now))
                    .order_by(SearchTask.next_run_at.asc().nullsfirst())
                    .limit(3)
                )
            ).scalars().all()
        if tasks:
            for task in tasks:
                active = asyncio.create_task(self._run_task(task))
                self._active_tasks.add(active)
                active.add_done_callback(self._active_tasks.discard)

    async def _run_task(self, task: SearchTask) -> tuple[SearchTaskStatus, IngestResponse | None]:
        async with self.sessionmaker() as session:
            async with session.begin():
                current = await session.scalar(select(SearchTask).where(SearchTask.name == task.name).with_for_update())
                if not current or current.running:
                    return task_to_status(task), None
                current.running = True
                current.running_started_at = datetime.utcnow()
                current.last_error = None
                depth, limit, pages_per_channel, interval_seconds = auto_policy(current)
                current.depth = depth
                current.limit = limit
                current.pages_per_channel = pages_per_channel
                current.interval_seconds = interval_seconds
                crawl_mode = current.crawl_mode or "backfill"
                freshness_days = current.freshness_days
                topic_slug = current.topic_slug or current.name
                topic = await session.scalar(select(ResearchTopic).where(ResearchTopic.slug == topic_slug))
                keywords = current.keywords or (topic.keywords if topic else None) or []
                seed_channels = current.seed_channels or (topic.seed_channels if topic else None) or []
                crawl = current.crawl
                task_run = TaskRun(task_name=current.name, status="running", ingest_stats={})
                session.add(task_run)
                await session.flush()
                task_run_id = task_run.id

        ingest: IngestResponse | None = None
        error: str | None = None
        status = "completed"
        ingest, error, status = await self._ingest_with_retry(
            keywords=keywords,
            seed_channels=seed_channels,
            depth=depth,
            limit=limit,
            pages_per_channel=pages_per_channel,
            crawl=crawl,
            task_name=task.name,
            crawl_mode=crawl_mode,
            freshness_days=freshness_days,
            topic_slug=topic_slug,
        )

        async with self.sessionmaker() as session:
            async with session.begin():
                updated = await session.scalar(select(SearchTask).where(SearchTask.name == task.name))
                if not updated:
                    return task_to_status(task), ingest
                updated.running = False
                updated.running_started_at = None
                updated.last_run_at = datetime.utcnow()
                if not error:
                    error = empty_run_error(ingest)
                updated.last_error = error
                updated.total_runs += 1
                if ingest:
                    updated.last_candidates_found = ingest.candidates_found
                    updated.last_valid_candidates = ingest.valid_candidates
                    updated.last_junk_candidates = ingest.junk_candidates
                    updated.last_channels_crawled = ingest.channels_crawled
                    updated.last_messages_saved = ingest.messages_saved
                    updated.last_edges_saved = ingest.edges_saved
                    updated.total_candidates += ingest.candidates_found
                    updated.total_valid_candidates += ingest.valid_candidates
                    updated.total_junk_candidates += ingest.junk_candidates
                    updated.total_channels_crawled += ingest.channels_crawled
                    updated.total_messages_saved += ingest.messages_saved
                else:
                    updated.last_candidates_found = 0
                    updated.last_valid_candidates = 0
                    updated.last_junk_candidates = 0
                    updated.last_channels_crawled = 0
                    updated.last_messages_saved = 0
                    updated.last_edges_saved = 0
                if updated.auto_tune:
                    updated.depth, updated.limit, updated.pages_per_channel, updated.interval_seconds = auto_policy(updated)
                updated.next_run_at = updated.last_run_at + timedelta(seconds=updated.interval_seconds)
                run_row = await session.get(TaskRun, task_run_id)
                if run_row:
                    run_row.status = "failed" if error and status == "completed" else status
                    run_row.finished_at = updated.last_run_at
                    run_row.error = error
                    run_row.ingest_stats = ingest.model_dump(mode="json") if ingest else {}
                return task_to_status(updated), ingest

    async def _ingest_with_retry(
        self,
        *,
        keywords: list[str],
        seed_channels: list[str],
        depth: int,
        limit: int,
        pages_per_channel: int,
        crawl: bool,
        task_name: str,
        crawl_mode: str,
        freshness_days: int | None,
        topic_slug: str,
    ) -> tuple[IngestResponse | None, str | None, str]:
        attempts = 0
        last_error: str | None = None
        while attempts < 3:
            attempts += 1
            try:
                ingest = await asyncio.wait_for(
                    self.ingesting.ingest(
                        keywords,
                        seed_channels,
                        depth,
                        limit,
                        pages_per_channel,
                        crawl,
                        task_name,
                        crawl_mode=crawl_mode,
                        freshness_days=freshness_days,
                        topic_slug=topic_slug,
                    ),
                    timeout=self.task_timeout_seconds,
                )
                return ingest, None, "completed"
            except asyncio.TimeoutError:
                return None, f"TimeoutError: task exceeded {self.task_timeout_seconds:.0f}s", "timeout"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                lowered = last_error.lower()
                if "validation" in lowered or "fact" in lowered:
                    return None, last_error, "failed_validation"
                max_attempts = 2 if ("llm" in lowered or "planner" in lowered or "write" in lowered) else 3
                if attempts >= max_attempts:
                    return None, last_error, "failed"
                await asyncio.sleep(min(2 ** attempts, 8))
        return None, last_error, "failed"
