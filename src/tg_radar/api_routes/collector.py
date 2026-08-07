from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from tg_radar.api_routes.deps import get_session, get_state
from tg_radar.app_state import AppState
from tg_radar.db import DiscoveredChannel, SearchTask, search_messages_db
from tg_radar.schemas import (
    DiscoverRequest,
    DiscoverResponse,
    ExpandRequest,
    ExpandResponse,
    GraphEdge,
    IngestRequest,
    IngestResponse,
    MonitorRequest,
    SearchFilters,
    SearchRequest,
    SearchResponse,
)
from tg_radar.service import graph_edges_from_db


router = APIRouter()


@router.post("/search", response_model=SearchResponse)
@router.post("/core/search", response_model=SearchResponse)
async def search(payload: SearchRequest, state: AppState = Depends(get_state), session: AsyncSession = Depends(get_session)):
    fetch_limit = min(max(payload.limit * 5, 50), 100)
    try:
        hits = await state.vespa.search(payload.query, payload.filters, fetch_limit, semantic=payload.semantic)
    except Exception:
        hits = await search_messages_db(session, payload.query, payload.filters, fetch_limit)
    if payload.rerank:
        hits = state.reranker.rerank(payload.query, hits, payload.limit)
    else:
        hits = hits[: payload.limit]
    return SearchResponse(query=payload.query, hits=hits)


@router.post("/discover", response_model=DiscoverResponse)
async def discover(payload: DiscoverRequest, state: AppState = Depends(get_state)):
    candidates = await state.discoverer.discover(payload.keywords, payload.seed_channels, payload.depth, payload.limit)
    return DiscoverResponse(candidates=candidates)


@router.post("/ingest", response_model=IngestResponse)
@router.post("/core/ingest", response_model=IngestResponse)
async def ingest(payload: IngestRequest, state: AppState = Depends(get_state)):
    return await state.ingesting.ingest(
        payload.keywords,
        payload.seed_channels,
        payload.depth,
        payload.limit,
        payload.pages_per_channel,
        payload.crawl,
        task_name=payload.topic or payload.task_name,
        max_live_crawl=payload.max_live_crawl,
        crawl_mode=payload.crawl_mode.value,
        freshness_days=payload.freshness_days,
        since=payload.since,
    )


@router.get("/candidates", response_model=DiscoverResponse)
async def candidates(limit: int = 100, session: AsyncSession = Depends(get_session)):
    rows = (
        await session.execute(
            select(DiscoveredChannel)
            .order_by(DiscoveredChannel.score.desc(), DiscoveredChannel.last_seen_at.desc())
            .limit(min(limit, 1000))
        )
    ).scalars().all()
    return DiscoverResponse(candidates=_candidate_rows(rows))


@router.get("/tasks/{name}/candidates", response_model=DiscoverResponse)
@router.get("/core/tasks/{name}/candidates", response_model=DiscoverResponse)
async def task_candidates(name: str, limit: int = 100, session: AsyncSession = Depends(get_session)):
    task = await session.scalar(select(SearchTask).where(SearchTask.name == name))
    seed_sources = []
    seed_usernames = []
    keyword_reasons = []
    if task:
        seed_usernames = [seed.strip("@") for seed in (task.seed_channels or [])]
        seed_sources = [f"{prefix}:{seed}" for seed in seed_usernames for prefix in ("linked", "telethon_recursive")]
        keyword_reasons = list(task.keywords or [])
    filters = [DiscoveredChannel.task_name == name]
    if seed_usernames:
        filters.append(DiscoveredChannel.username.in_(seed_usernames))
    if seed_sources:
        filters.append(DiscoveredChannel.source.in_(seed_sources))
    if keyword_reasons:
        filters.append(DiscoveredChannel.reason.in_(keyword_reasons))
    rows = (
        await session.execute(
            select(DiscoveredChannel)
            .where(or_(*filters))
            .order_by(DiscoveredChannel.score.desc(), DiscoveredChannel.last_seen_at.desc())
            .limit(min(limit, 1000))
        )
    ).scalars().all()
    return DiscoverResponse(candidates=_candidate_rows(rows))


@router.post("/expand", response_model=ExpandResponse)
async def expand(payload: ExpandRequest, state: AppState = Depends(get_state), session: AsyncSession = Depends(get_session)):
    edges = await graph_edges_from_db(session, payload.channel, payload.limit)
    if edges:
        return ExpandResponse(channel=payload.channel.strip("@"), edges=edges)
    candidates = await state.discoverer.expand_from_seeds([payload.channel], payload.depth, payload.limit)
    root = payload.channel.strip("@")
    return ExpandResponse(
        channel=root,
        edges=[
            GraphEdge(source=root, target=c.username, edge_type="link", message_url=None)
            for c in candidates
            if c.username != root
        ],
    )


@router.post("/monitor", response_model=SearchResponse)
async def monitor(payload: MonitorRequest, state: AppState = Depends(get_state), session: AsyncSession = Depends(get_session)):
    filters = SearchFilters(date_from=payload.since or datetime.fromtimestamp(0))
    request = SearchRequest(query=payload.query, filters=filters, limit=payload.limit)
    return await search(request, state, session)


def _candidate_rows(rows) -> list[dict]:
    seen = set()
    return [
        {
            "username": row.username,
            "source": row.source,
            "reason": row.reason,
            "score": row.score,
            "depth": row.depth,
            "task_name": row.task_name,
            "state": row.state,
            "quality_score": row.quality_score,
            "quality_reasons": row.quality_reasons or [],
            "operator_niche_category": row.operator_niche_category,
            "operator_niche_tags": row.operator_niche_tags or [],
            "operator_niche_confidence": row.operator_niche_confidence or 0.0,
        }
        for row in rows
        if not (row.username.lower() in seen or seen.add(row.username.lower()))
    ]
