from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from sqlalchemy import func, select
from starlette.responses import JSONResponse
from starlette.responses import Response

from tg_radar.api_routes.admin import router as admin_router
from tg_radar.api_routes.channels import router as channels_router
from tg_radar.api_routes.collector import router as collector_router
from tg_radar.api_routes.content import router as content_router
from tg_radar.api_routes.stats import router as stats_router
from tg_radar.api_routes.telethon import router as telethon_router
from tg_radar.api_routes.topics import router as topics_router
from tg_radar.app_state import build_state
from tg_radar.api_routes.system import router as system_router
from tg_radar.config import get_settings
from tg_radar.db import AgentEvalRun, AgentRun, TaskRun, init_db


REQUESTS = Counter("tg_radar_api_requests_total", "API requests", ["path"])
LATENCY = Histogram("tg_radar_api_latency_seconds", "API request latency", ["path"])
AGENT_RUNS_BY_STATUS = Gauge("tg_radar_agent_runs", "Agent runs by status", ["status"])
TASK_RUNS_BY_STATUS = Gauge("tg_radar_task_runs", "Task runs by status", ["status"])
LAST_EVAL_SCORE = Gauge("tg_radar_agent_eval_score", "Last agent eval score")
DAILY_CRAWL_BUDGET: dict[tuple[str, str], int] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = build_state()
    await init_db(state.engine)
    app.state.tg_radar = state
    await state.agent_runs.recover_stale()
    if state.settings.auto_worker_enabled:
        state.auto_search.start()
    if state.settings.agent_run_worker_enabled:
        state.agent_runs.start_worker(state.agent)
    yield
    await state.agent_runs.stop()
    await state.auto_search.stop()
    await state.engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(title="TG Radar", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def metrics_middleware(request, call_next):
        path = request.url.path
        auth_response = _auth_guard(request)
        if auth_response:
            return auth_response
        REQUESTS.labels(path=path).inc()
        with LATENCY.labels(path=path).time():
            return await call_next(request)

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.get("/metrics")
    async def metrics():
        await _refresh_db_metrics(app.state.tg_radar)
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    app.include_router(system_router)
    app.include_router(content_router)
    app.include_router(collector_router)
    app.include_router(topics_router)
    app.include_router(admin_router)
    app.include_router(channels_router)
    app.include_router(stats_router)
    app.include_router(telethon_router)

    return app


def _auth_guard(request) -> JSONResponse | None:
    settings = getattr(getattr(request.app.state, "tg_radar", None), "settings", None) or get_settings()
    token = settings.api_token
    if not token:
        return None
    path = request.url.path
    if path == "/health":
        return None
    if path == "/metrics" and settings.public_metrics:
        return None
    if _requires_token(request.method, path):
        expected = f"Bearer {token}"
        if request.headers.get("authorization") != expected:
            return JSONResponse({"detail": "missing or invalid bearer token"}, status_code=401)
    if _counts_against_crawl_budget(request.method, path):
        day = datetime.utcnow().strftime("%Y-%m-%d")
        key = (request.headers.get("authorization", "anonymous"), day)
        used = DAILY_CRAWL_BUDGET.get(key, 0)
        if used >= settings.api_max_daily_crawl_budget:
            return JSONResponse({"detail": "daily crawl budget exceeded"}, status_code=429)
        DAILY_CRAWL_BUDGET[key] = used + 1
    return None


def _requires_token(method: str, path: str) -> bool:
    if path == "/metrics":
        return True
    if method.upper() in {"POST", "PUT", "PATCH", "DELETE"}:
        return True
    if path.startswith("/telethon") or path.startswith("/core/telethon"):
        return True
    return False


def _counts_against_crawl_budget(method: str, path: str) -> bool:
    if method.upper() != "POST":
        return False
    return any(marker in path for marker in ("/ingest", "/crawl/", "/fetch-messages", "/tasks/")) or path.endswith("/agent/runs")


async def _refresh_db_metrics(state) -> None:
    async with state.sessionmaker() as session:
        agent_rows = (await session.execute(select(AgentRun.status, func.count()).group_by(AgentRun.status))).all()
        for status, count in agent_rows:
            AGENT_RUNS_BY_STATUS.labels(status=status).set(count)
        task_rows = (await session.execute(select(TaskRun.status, func.count()).group_by(TaskRun.status))).all()
        for status, count in task_rows:
            TASK_RUNS_BY_STATUS.labels(status=status).set(count)
        latest_eval = (
            await session.execute(select(AgentEvalRun).order_by(AgentEvalRun.started_at.desc()).limit(1))
        ).scalar_one_or_none()
        if latest_eval:
            LAST_EVAL_SCORE.set(latest_eval.score or 0.0)


def main() -> None:
    import uvicorn

    uvicorn.run(create_app(), host="0.0.0.0", port=8080)
