from __future__ import annotations

import argparse
import asyncio

from tg_radar import cli
from tg_radar.cli import _close_state


class CancelOnClose:
    async def stop(self):
        raise asyncio.CancelledError()

    async def dispose(self):
        raise asyncio.CancelledError()


class DummyState:
    def __init__(self) -> None:
        self.auto_search = CancelOnClose()
        self.engine = CancelOnClose()


async def test_close_state_suppresses_cancelled_error():
    await _close_state(DummyState())


async def test_agent_shell_handles_keyboard_interrupt(monkeypatch):
    monkeypatch.setattr(cli, "build_state", lambda: DummyState())

    async def noop_init_db(engine):
        return None

    def interrupt(prompt):
        raise KeyboardInterrupt()

    monkeypatch.setattr(cli, "init_db", noop_init_db)
    monkeypatch.setattr("builtins.input", interrupt)

    await cli.agent_shell(argparse.Namespace())
