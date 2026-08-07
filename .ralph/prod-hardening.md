# TG Radar Prod Hardening

Implement production hardening from pasted request, excluding article quality block.

## Checklist
- [x] DB-backed agent runs and worker queue
- [x] Agent run cancel/retry/status recovery
- [x] Message refresh/freshness fields and update semantics
- [x] Task run telemetry and stale task handling
- [x] Eval telemetry API
- [x] API token auth and deep health
- [x] Worker compose service and backup docs
- [x] UI failure states, cancel, retry, eval status
- [x] Focused backend and frontend checks

## Verification
- `.venv/bin/python -m pytest -q` -> 98 passed
- `cd frontend && npm run build` -> passed
- `.venv/bin/alembic upgrade head --sql` -> rendered migration SQL
- Agent events include `duration_ms` for tool telemetry.
- `init_db` now only creates metadata for bootstrap; ALTER upgrades moved to Alembic.

## Notes
- Ralph tool unavailable in this session; using this file for loop state.
