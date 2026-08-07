from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from uuid import uuid4

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from tg_radar.db import AgentEventLog, AgentRun
from tg_radar.schemas import CollectionAgentRequest, CollectionAgentResponse, CollectionAgentStep


TERMINAL_STATUSES = {"completed", "failed", "cancelled", "timeout"}


class AgentRunRecord:
    def __init__(
        self,
        run_id: str,
        status: str = "running",
        final: str | None = None,
        error: str | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
        steps: list[CollectionAgentStep] | None = None,
    ) -> None:
        self.run_id = run_id
        self.status = status
        self.final = final
        self.error = error
        self.started_at = started_at or datetime.utcnow()
        self.finished_at = finished_at
        self.steps = steps or []
        self.task: asyncio.Task | None = None

    @classmethod
    def from_row(cls, row: AgentRun) -> "AgentRunRecord":
        return cls(
            run_id=row.run_id,
            status=row.status,
            final=row.final,
            error=row.error,
            started_at=row.started_at,
            finished_at=row.finished_at,
        )


class AgentRunRegistry:
    def __init__(
        self,
        sessionmaker: async_sessionmaker[AsyncSession] | None = None,
        timeout_seconds: float = 600.0,
        poll_seconds: float = 1.0,
        worker_id: str | None = None,
    ) -> None:
        self.sessionmaker = sessionmaker
        self.timeout_seconds = timeout_seconds
        self.poll_seconds = poll_seconds
        self.worker_id = worker_id or f"agent-worker-{uuid4().hex[:12]}"
        self._runs: dict[str, AgentRunRecord] = {}
        self._lock = asyncio.Lock()
        self._worker_task: asyncio.Task | None = None
        self._active: dict[str, asyncio.Task] = {}
        self._stop = asyncio.Event()

    @property
    def running(self) -> bool:
        return self._worker_task is not None and not self._worker_task.done()

    async def start(self, request: CollectionAgentRequest, agent) -> AgentRunRecord:
        if self.sessionmaker is None:
            return await self._legacy_start(request, agent)
        run_id = request.run_id or f"run_{uuid4().hex}"
        payload = request.model_copy(update={"run_id": run_id})
        async with self.sessionmaker() as session:
            async with session.begin():
                existing = await session.get(AgentRun, run_id)
                if existing:
                    return AgentRunRecord.from_row(existing)
                row = AgentRun(
                    run_id=run_id,
                    status="queued",
                    request_json=payload.model_dump(mode="json"),
                    cancel_requested=False,
                )
                session.add(row)
                await session.flush()
                return AgentRunRecord.from_row(row)

    async def get(self, run_id: str) -> AgentRunRecord | None:
        if self.sessionmaker is None:
            async with self._lock:
                return self._runs.get(run_id)
        async with self.sessionmaker() as session:
            row = await session.get(AgentRun, run_id)
            return AgentRunRecord.from_row(row) if row else None

    async def cancel(self, run_id: str) -> AgentRunRecord | None:
        if self.sessionmaker is None:
            async with self._lock:
                record = self._runs.get(run_id)
            if record and record.task and not record.task.done():
                record.task.cancel()
            return record
        task = self._active.get(run_id)
        if task and not task.done():
            task.cancel()
        async with self.sessionmaker() as session:
            async with session.begin():
                row = await session.get(AgentRun, run_id)
                if not row:
                    return None
                row.cancel_requested = True
                if row.status == "queued":
                    row.status = "cancelled"
                    row.error = "cancelled"
                    row.finished_at = datetime.utcnow()
                return AgentRunRecord.from_row(row)

    async def retry(self, run_id: str) -> AgentRunRecord | None:
        if self.sessionmaker is None:
            async with self._lock:
                record = self._runs.get(run_id)
                if not record or record.status not in TERMINAL_STATUSES:
                    return record
                record.status = "running"
                record.final = None
                record.error = None
                record.finished_at = None
                return record
        async with self.sessionmaker() as session:
            async with session.begin():
                row = await session.get(AgentRun, run_id)
                if not row:
                    return None
                if row.status not in TERMINAL_STATUSES:
                    return AgentRunRecord.from_row(row)
                await session.execute(delete(AgentEventLog).where(AgentEventLog.run_id == run_id))
                row.status = "queued"
                row.final = None
                row.error = None
                row.finished_at = None
                row.locked_at = None
                row.worker_id = None
                row.cancel_requested = False
                row.started_at = datetime.utcnow()
                return AgentRunRecord.from_row(row)

    def start_worker(self, agent) -> None:
        if self.sessionmaker is None or self.running:
            return
        self._stop = asyncio.Event()
        self._worker_task = asyncio.create_task(self._worker_loop(agent))

    async def recover_stale(self) -> None:
        if self.sessionmaker is None:
            return
        now = datetime.utcnow()
        stale_after = now - timedelta(seconds=self.timeout_seconds)
        fail_after = now - timedelta(seconds=self.timeout_seconds * 2)
        async with self.sessionmaker() as session:
            async with session.begin():
                rows = (
                    await session.execute(
                        select(AgentRun).where(
                            AgentRun.status == "running",
                            (AgentRun.locked_at.is_(None)) | (AgentRun.locked_at <= stale_after),
                        )
                    )
                ).scalars().all()
                for row in rows:
                    if row.started_at and row.started_at <= fail_after:
                        row.status = "timeout"
                        row.error = f"stale running run exceeded {self.timeout_seconds:.0f}s timeout"
                        row.finished_at = now
                    else:
                        row.status = "queued"
                        row.locked_at = None
                        row.worker_id = None

    async def stop(self) -> None:
        if self.sessionmaker is None:
            await self._legacy_stop()
            return
        self._stop.set()
        for task in list(self._active.values()):
            task.cancel()
        if self._worker_task:
            await asyncio.wait([self._worker_task], timeout=5)
        if self._active:
            await asyncio.gather(*self._active.values(), return_exceptions=True)
        self._worker_task = None

    async def active_count(self) -> int:
        if self.sessionmaker is None:
            async with self._lock:
                return sum(1 for record in self._runs.values() if record.status in {"queued", "running"})
        async with self.sessionmaker() as session:
            rows = await session.execute(select(AgentRun.run_id).where(AgentRun.status.in_(("queued", "running"))))
            return len(rows.all())

    async def _worker_loop(self, agent) -> None:
        await self.recover_stale()
        while not self._stop.is_set():
            run_id = await self._claim_next()
            if not run_id:
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self.poll_seconds)
                except asyncio.TimeoutError:
                    continue
                continue
            task = asyncio.create_task(self._execute_claimed(run_id, agent))
            self._active[run_id] = task
            task.add_done_callback(lambda done, rid=run_id: self._active.pop(rid, None))

    async def _claim_next(self) -> str | None:
        if self.sessionmaker is None:
            return None
        now = datetime.utcnow()
        stale_after = now - timedelta(seconds=self.timeout_seconds)
        async with self.sessionmaker() as session:
            async with session.begin():
                row = (
                    await session.execute(
                        select(AgentRun)
                        .where(
                            (AgentRun.status == "queued")
                            | (
                                (AgentRun.status == "running")
                                & ((AgentRun.locked_at.is_(None)) | (AgentRun.locked_at <= stale_after))
                            )
                        )
                        .order_by(AgentRun.started_at.asc())
                        .with_for_update(skip_locked=True)
                        .limit(1)
                    )
                ).scalar_one_or_none()
                if not row:
                    return None
                if row.cancel_requested:
                    row.status = "cancelled"
                    row.error = "cancelled"
                    row.finished_at = now
                    return None
                row.status = "running"
                row.locked_at = now
                row.worker_id = self.worker_id
                row.error = None
                return row.run_id

    async def _execute_claimed(self, run_id: str, agent) -> None:
        try:
            request = await self._load_request(run_id)
            if not request:
                return
            response: CollectionAgentResponse = await self._run_with_cancel_poll(run_id, agent, request)
            await self._finish(run_id, "completed", final=response.final)
        except asyncio.CancelledError:
            await self._finish(run_id, "cancelled", error="cancelled")
        except asyncio.TimeoutError:
            await self._finish(run_id, "timeout", error=f"run exceeded {self.timeout_seconds:.0f}s")
        except Exception as exc:
            await self._finish(run_id, "failed", error=f"{type(exc).__name__}: {exc}")

    async def _run_with_cancel_poll(self, run_id: str, agent, request: CollectionAgentRequest) -> CollectionAgentResponse:
        task = asyncio.create_task(agent.run(request))
        deadline = asyncio.get_running_loop().time() + self.timeout_seconds
        try:
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    task.cancel()
                    raise asyncio.TimeoutError
                done, _ = await asyncio.wait({task}, timeout=min(1.0, remaining))
                if done:
                    return task.result()
                if await self._is_cancel_requested(run_id):
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    raise asyncio.CancelledError
        except Exception:
            if not task.done():
                task.cancel()
            raise

    async def _is_cancel_requested(self, run_id: str) -> bool:
        if self.sessionmaker is None:
            return False
        async with self.sessionmaker() as session:
            row = await session.get(AgentRun, run_id)
            return bool(row and row.cancel_requested)

    async def _load_request(self, run_id: str) -> CollectionAgentRequest | None:
        if self.sessionmaker is None:
            return None
        async with self.sessionmaker() as session:
            row = await session.get(AgentRun, run_id)
            if not row:
                return None
            if row.cancel_requested:
                await self._finish(run_id, "cancelled", error="cancelled")
                return None
            data = dict(row.request_json or {})
            data["run_id"] = run_id
            return CollectionAgentRequest.model_validate(data)

    async def _finish(self, run_id: str, status: str, final: str | None = None, error: str | None = None) -> None:
        if self.sessionmaker is None:
            return
        async with self.sessionmaker() as session:
            async with session.begin():
                row = await session.get(AgentRun, run_id)
                if not row:
                    return
                row.status = status
                row.final = final
                row.error = error
                row.finished_at = datetime.utcnow()
                row.locked_at = None

    async def _legacy_start(self, request: CollectionAgentRequest, agent) -> AgentRunRecord:
        run_id = request.run_id or f"run_{uuid4().hex}"
        payload = request.model_copy(update={"run_id": run_id})
        record = AgentRunRecord(run_id)
        async with self._lock:
            self._runs[run_id] = record
        record.task = asyncio.create_task(self._legacy_execute(record, payload, agent))
        return record

    async def _legacy_stop(self) -> None:
        async with self._lock:
            tasks = [record.task for record in self._runs.values() if record.task and not record.task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _legacy_execute(self, record: AgentRunRecord, request: CollectionAgentRequest, agent) -> None:
        try:
            response: CollectionAgentResponse = await agent.run(request)
            record.status = "completed"
            record.final = response.final
            record.steps = response.steps
        except asyncio.CancelledError:
            record.status = "cancelled"
            record.error = "cancelled"
            raise
        except Exception as exc:
            record.status = "failed"
            record.error = str(exc)
        finally:
            record.finished_at = datetime.utcnow()
