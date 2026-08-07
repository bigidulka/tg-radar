from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from tg_radar.agent_core.models import AgentEvent, AgentStep
from tg_radar.agent_core.state import AgentStateStore
from tg_radar.db import AgentEventLog


class SqlAlchemyAgentStateStore:
    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]) -> None:
        self.sessionmaker = sessionmaker

    async def append_step(self, run_id: str, step: AgentStep) -> None:
        event = AgentEvent(run_id=run_id, step=step)
        async with self.sessionmaker() as session:
            async with session.begin():
                session.add(
                    AgentEventLog(
                        run_id=run_id,
                        step_index=step.step,
                        event=event.model_dump(mode="json"),
                    )
                )

    async def events(self, run_id: str) -> list[AgentEvent]:
        async with self.sessionmaker() as session:
            rows = (
                await session.execute(
                    select(AgentEventLog)
                    .where(AgentEventLog.run_id == run_id)
                    .order_by(AgentEventLog.step_index, AgentEventLog.id)
                )
            ).scalars().all()
        return [AgentEvent.model_validate(row.event) for row in rows]


def _typecheck_store(store: SqlAlchemyAgentStateStore) -> AgentStateStore:
    return store
