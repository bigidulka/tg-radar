from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select, text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from tg_radar.agent_eval import AgentEvalCase, AgentEvalRunner
from tg_radar.api_routes.deps import get_session, get_state
from tg_radar.app_state import AppState
from tg_radar.db import AgentEvalCaseRecord, AgentEvalRun, WorkerHeartbeat
from tg_radar.schemas import (
    AgentEvalCaseStatus,
    AgentEvalRunRequest,
    AgentEvalRunStatus,
    AgentRunStartResponse,
    AgentRunStatusResponse,
    CollectionAgentRequest,
    CollectionAgentResponse,
    CollectionAgentStep,
    DeepHealthStatus,
    SearchTaskRequest,
    SearchTaskRunResponse,
    SearchTaskStatus,
)


router = APIRouter()


@router.get("/core/status")
async def core_status(state: AppState = Depends(get_state), session: AsyncSession = Depends(get_session)):
    latest_eval = (
        await session.execute(select(AgentEvalRun).order_by(AgentEvalRun.started_at.desc()).limit(1))
    ).scalar_one_or_none()
    return {
        "ok": True,
        "engine_running": state.auto_search.running,
        "active_tasks": state.auto_search.active_count,
        "agent_worker_running": state.agent_runs.running,
        "content_factory_enabled": state.settings.content_factory_enabled,
        "llm_enabled": bool(state.settings.llm_base_url and state.settings.llm_api_key),
        "last_eval_score": latest_eval.score if latest_eval else None,
        "last_eval_at": latest_eval.finished_at if latest_eval else None,
        "scope": "collector_core",
    }


@router.get("/core/health/deep", response_model=DeepHealthStatus)
async def deep_health(state: AppState = Depends(get_state), session: AsyncSession = Depends(get_session)):
    details: dict = {}
    db_ok = False
    worker_ok = False
    vespa_ok = False
    try:
        await session.execute(sql_text("SELECT 1"))
        db_ok = True
    except Exception as exc:
        details["db_error"] = f"{type(exc).__name__}: {exc}"
    try:
        latest = (
            await session.execute(select(WorkerHeartbeat).order_by(WorkerHeartbeat.heartbeat_at.desc()).limit(1))
        ).scalar_one_or_none()
        heartbeat_at = _naive_utc(latest.heartbeat_at) if latest else None
        worker_ok = bool(state.auto_search.running or state.agent_runs.running or (heartbeat_at and heartbeat_at >= datetime.utcnow() - timedelta(minutes=3)))
        if latest:
            details["worker_heartbeat_at"] = latest.heartbeat_at.isoformat()
    except Exception as exc:
        details["worker_error"] = f"{type(exc).__name__}: {exc}"
    try:
        await state.vespa.health()
        vespa_ok = True
    except Exception as exc:
        details["vespa_error"] = f"{type(exc).__name__}: {exc}"
    llm_ok = bool(state.settings.llm_base_url and state.settings.llm_api_key)
    crawler_ok = bool(state.settings.user_agent)
    return DeepHealthStatus(
        ok=db_ok and worker_ok and crawler_ok,
        db=db_ok,
        worker=worker_ok,
        llm=llm_ok,
        vespa=vespa_ok,
        crawler=crawler_ok,
        details=details,
    )


@router.post("/tasks", response_model=SearchTaskStatus)
@router.post("/core/tasks", response_model=SearchTaskStatus)
async def upsert_task(payload: SearchTaskRequest, state: AppState = Depends(get_state)):
    return await state.auto_search.upsert_task(payload)


@router.get("/tasks", response_model=list[SearchTaskStatus])
@router.get("/core/tasks", response_model=list[SearchTaskStatus])
async def list_tasks(state: AppState = Depends(get_state)):
    return await state.auto_search.list_tasks()


@router.get("/tasks/{name}", response_model=SearchTaskStatus)
@router.get("/core/tasks/{name}", response_model=SearchTaskStatus)
async def get_task(name: str, state: AppState = Depends(get_state)):
    task = await state.auto_search.get_task(name)
    if not task:
        raise HTTPException(status_code=404, detail="task not found")
    return task


@router.post("/tasks/{name}/run", response_model=SearchTaskRunResponse)
@router.post("/core/tasks/{name}/run", response_model=SearchTaskRunResponse)
async def run_task(name: str, state: AppState = Depends(get_state)):
    result = await state.auto_search.run_task_now(name)
    if not result:
        raise HTTPException(status_code=404, detail="task not found")
    return result


@router.post("/engine/start")
@router.post("/core/engine/start")
async def start_engine(state: AppState = Depends(get_state)):
    state.auto_search.start()
    return {"running": state.auto_search.running}


@router.post("/engine/stop")
@router.post("/core/engine/stop")
async def stop_engine(state: AppState = Depends(get_state)):
    await state.auto_search.stop()
    return {"running": state.auto_search.running}


@router.get("/engine/status")
@router.get("/core/engine/status")
async def engine_status(state: AppState = Depends(get_state)):
    return {"running": state.auto_search.running, "active_tasks": state.auto_search.active_count}


@router.post("/agent/run", response_model=CollectionAgentResponse)
@router.post("/core/agent/run", response_model=CollectionAgentResponse)
async def run_agent(payload: CollectionAgentRequest, state: AppState = Depends(get_state)):
    return await state.agent.run(payload)


@router.post("/agent/runs", response_model=AgentRunStartResponse)
@router.post("/core/agent/runs", response_model=AgentRunStartResponse)
async def start_agent_run(payload: CollectionAgentRequest, state: AppState = Depends(get_state)):
    active = await state.agent_runs.active_count()
    limit = getattr(getattr(state, "settings", None), "api_max_concurrent_agent_runs", 1000000)
    if active >= limit:
        raise HTTPException(status_code=429, detail="too many queued or running agent runs")
    run = await state.agent_runs.start(payload, state.agent)
    return AgentRunStartResponse(run_id=run.run_id, status=run.status)


@router.get("/agent/runs/{run_id}", response_model=AgentRunStatusResponse)
@router.get("/core/agent/runs/{run_id}", response_model=AgentRunStatusResponse)
async def get_agent_run(run_id: str, state: AppState = Depends(get_state)):
    run = await state.agent_runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="agent run not found")
    events = await state.agent.orchestrator.state_store.events(run_id)
    steps = [CollectionAgentStep.model_validate(event.step.model_dump(mode="json")) for event in events]
    return AgentRunStatusResponse(
        run_id=run.run_id,
        status=run.status,
        final=run.final,
        error=run.error,
        started_at=run.started_at,
        finished_at=run.finished_at,
        steps=steps or run.steps,
    )


@router.post("/agent/runs/{run_id}/cancel", response_model=AgentRunStatusResponse)
@router.post("/core/agent/runs/{run_id}/cancel", response_model=AgentRunStatusResponse)
async def cancel_agent_run(run_id: str, state: AppState = Depends(get_state)):
    run = await state.agent_runs.cancel(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="agent run not found")
    return await get_agent_run(run_id, state)


@router.post("/agent/runs/{run_id}/retry", response_model=AgentRunStartResponse)
@router.post("/core/agent/runs/{run_id}/retry", response_model=AgentRunStartResponse)
async def retry_agent_run(run_id: str, state: AppState = Depends(get_state)):
    run = await state.agent_runs.retry(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="agent run not found")
    return AgentRunStartResponse(run_id=run.run_id, status=run.status)


@router.get("/agent/tools")
@router.get("/core/agent/tools")
async def agent_tools(state: AppState = Depends(get_state)):
    return [tool.model_dump(mode="json") for tool in state.agent.tool_registry.specs()]


@router.post("/core/agent/evals/run", response_model=AgentEvalRunStatus)
async def run_agent_eval(
    payload: AgentEvalRunRequest,
    state: AppState = Depends(get_state),
    session: AsyncSession = Depends(get_session),
):
    raw_cases = payload.cases or _baseline_eval_cases()
    cases = [AgentEvalCase.model_validate(item) for item in raw_cases]
    prompt_hash = hashlib.sha256(state.settings.agent_collection_system_prompt.encode("utf-8")).hexdigest()[:16]
    eval_row = AgentEvalRun(
        model=state.settings.llm_model,
        prompt_hash=prompt_hash,
        harness_config=payload.harness_config,
    )
    session.add(eval_row)
    await session.flush()
    runner = AgentEvalRunner(state.agent.orchestrator)
    results = await runner.run_cases(cases)
    eval_row.finished_at = datetime.utcnow()
    eval_row.score = sum(item.score for item in results) / max(len(results), 1)
    for case, result in zip(cases, results, strict=False):
        session.add(
            AgentEvalCaseRecord(
                eval_run_id=eval_row.id,
                case_name=result.name,
                request=case.request.model_dump(mode="json"),
                expected_tools=case.expected_tools or case.expected_tool_prefix or case.required_tools,
                actual_tools=result.tools,
                final=result.final,
                failures=result.failures,
            )
        )
    await session.flush()
    return await _eval_status(session, eval_row)


@router.get("/core/agent/evals", response_model=list[AgentEvalRunStatus])
async def list_agent_evals(session: AsyncSession = Depends(get_session)):
    rows = (await session.execute(select(AgentEvalRun).order_by(AgentEvalRun.started_at.desc()).limit(50))).scalars().all()
    return [await _eval_status(session, row) for row in rows]


@router.get("/core/agent/evals/{eval_id}", response_model=list[AgentEvalCaseStatus])
async def get_agent_eval(eval_id: int, session: AsyncSession = Depends(get_session)):
    run = await session.get(AgentEvalRun, eval_id)
    if not run:
        raise HTTPException(status_code=404, detail="eval run not found")
    rows = (
        await session.execute(select(AgentEvalCaseRecord).where(AgentEvalCaseRecord.eval_run_id == eval_id).order_by(AgentEvalCaseRecord.id))
    ).scalars().all()
    return [
        AgentEvalCaseStatus(
            id=row.id,
            eval_run_id=row.eval_run_id,
            case_name=row.case_name,
            request=row.request,
            expected_tools=row.expected_tools or [],
            actual_tools=row.actual_tools or [],
            final=row.final,
            failures=row.failures or [],
        )
        for row in rows
    ]


async def _eval_status(session: AsyncSession, row: AgentEvalRun) -> AgentEvalRunStatus:
    counts = (
        await session.execute(
            select(
                func.count(AgentEvalCaseRecord.id),
                func.count(AgentEvalCaseRecord.id).filter(func.cardinality(AgentEvalCaseRecord.failures) > 0),
            ).where(AgentEvalCaseRecord.eval_run_id == row.id)
        )
    ).one()
    return AgentEvalRunStatus(
        id=row.id,
        model=row.model,
        prompt_hash=row.prompt_hash,
        harness_config=row.harness_config or {},
        started_at=row.started_at,
        finished_at=row.finished_at,
        score=row.score or 0.0,
        cases=int(counts[0] or 0),
        failures=int(counts[1] or 0),
    )


def _baseline_eval_cases() -> list[dict]:
    return [
        {
            "name": "tool-sequence-status",
            "request": {"goal": "status", "mode": "status", "max_steps": 2},
            "expected_tool_prefix": ["core_status"],
        },
        {
            "name": "article-happy-path",
            "request": {
                "goal": "write article from verified evidence",
                "inputs": {"topic": "eval-baseline", "article_focus": "baseline article", "keywords": ["ai agents"]},
                "max_steps": 8,
            },
            "required_tools": ["review_evidence"],
            "forbidden_tools": ["final_article"],
        },
        {
            "name": "weak-evidence-refusal",
            "request": {
                "goal": "write article with weak evidence",
                "inputs": {"topic": "eval-empty-topic", "article_focus": "unsupported claim", "keywords": ["unlikely-eval-token"]},
                "max_steps": 4,
            },
            "forbidden_tools": ["final_article"],
        },
        {
            "name": "final-article-required-after-verify",
            "request": {
                "goal": "produce final article after claim verification",
                "inputs": {"topic": "eval-baseline", "article_focus": "baseline final", "keywords": ["ai agents"]},
                "max_steps": 12,
            },
            "required_tools": ["verify_claims", "final_article"],
        },
        {
            "name": "crawl-disabled-discovery",
            "request": {"goal": "discover only", "mode": "discover", "inputs": {"keywords": ["ai agents"], "crawl": False}, "max_steps": 2},
            "expected_tool_prefix": ["discover"],
        },
        {
            "name": "digest-freshness-default",
            "request": {"goal": "news digest for ai agents", "mode": "ingest", "inputs": {"keywords": ["ai agents"]}, "max_steps": 2},
            "expected_tool_prefix": ["ingest"],
        },
    ]


def _naive_utc(value: datetime) -> datetime:
    if value.tzinfo:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value
