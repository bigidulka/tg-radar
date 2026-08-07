from __future__ import annotations

from typing import Protocol

from tg_radar.agent_core.models import AgentEvent, AgentStep


class AgentStateStore(Protocol):
    async def append_step(self, run_id: str, step: AgentStep) -> None:
        ...

    async def events(self, run_id: str) -> list[AgentEvent]:
        ...


class MemoryAgentStateStore:
    def __init__(self) -> None:
        self._events: dict[str, list[AgentEvent]] = {}

    async def append_step(self, run_id: str, step: AgentStep) -> None:
        self._events.setdefault(run_id, []).append(AgentEvent(run_id=run_id, step=step))

    async def events(self, run_id: str) -> list[AgentEvent]:
        return list(self._events.get(run_id, []))
