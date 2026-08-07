from __future__ import annotations

from typing import AsyncIterator

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from tg_radar.app_state import AppState


def get_state(request: Request) -> AppState:
    return request.app.state.tg_radar


async def iter_session(state: AppState) -> AsyncIterator[AsyncSession]:
    async with state.sessionmaker() as session:
        async with session.begin():
            yield session


async def get_session(state: AppState = Depends(get_state)) -> AsyncIterator[AsyncSession]:
    async for session in iter_session(state):
        yield session
