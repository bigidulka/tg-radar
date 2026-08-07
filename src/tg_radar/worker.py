from __future__ import annotations

import asyncio
from datetime import datetime

from tg_radar.app_state import build_state
from tg_radar.db import WorkerHeartbeat, init_db


async def heartbeat_loop(state, worker_id: str) -> None:
    while True:
        async with state.sessionmaker() as session:
            async with session.begin():
                row = await session.get(WorkerHeartbeat, worker_id)
                if not row:
                    row = WorkerHeartbeat(worker_id=worker_id, role="worker", payload={})
                    session.add(row)
                row.heartbeat_at = datetime.utcnow()
                row.payload = {
                    "auto_search": state.auto_search.running,
                    "agent_runs": state.agent_runs.running,
                }
        await asyncio.sleep(30)


async def run_worker() -> None:
    state = build_state()
    await init_db(state.engine)
    await state.agent_runs.recover_stale()
    state.auto_search.start()
    state.agent_runs.start_worker(state.agent)
    heartbeat_task = asyncio.create_task(heartbeat_loop(state, state.agent_runs.worker_id))
    try:
        await asyncio.Event().wait()
    finally:
        heartbeat_task.cancel()
        await asyncio.gather(heartbeat_task, return_exceptions=True)
        await state.agent_runs.stop()
        await state.auto_search.stop()
        await state.engine.dispose()


def main() -> None:
    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
