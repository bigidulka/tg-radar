from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from tg_radar.api_routes.deps import get_session


router = APIRouter()


@router.get("/stats")
async def stats(session: AsyncSession = Depends(get_session)):
    totals = (
        await session.execute(
            sql_text(
                """
                SELECT
                  (SELECT count(DISTINCT lower(username)) FROM discovered_channels) AS unique_candidates,
                  (SELECT count(*) FROM discovered_channels) AS candidate_rows,
                  (SELECT count(*) FROM channels) AS confirmed_channels,
                  (SELECT count(DISTINCT channel_id) FROM messages) AS indexed_channels,
                  (SELECT count(*) FROM messages) AS messages,
                  (SELECT count(*) FROM vacancy_clusters) AS vacancy_clusters,
                  (SELECT count(*) FROM vacancy_clusters WHERE message_count > 1) AS duplicate_clusters,
                  (SELECT count(*) FROM edges) AS edges,
                  (SELECT count(DISTINCT lower(dst_username)) FROM edges) AS unique_edge_targets,
                  (SELECT count(DISTINCT source) FROM discovered_channels) AS sources,
                  (SELECT count(*) FROM search_tasks) AS tasks,
                  (SELECT count(*) FROM search_tasks WHERE enabled IS TRUE) AS enabled_tasks,
                  (SELECT count(*) FROM search_tasks WHERE running IS TRUE) AS running_tasks,
                  (SELECT count(*) FROM research_topics) AS topics,
                  (SELECT count(*) FROM topic_messages) AS topic_messages,
                  (SELECT count(*) FROM topic_messages WHERE pain_score >= 0.28) AS pain_items,
                  (SELECT count(*) FROM messages WHERE indexed_at IS NOT NULL) AS vespa_indexed_messages,
                  (SELECT count(*) FROM channels WHERE quality_score >= 0.65) AS high_quality_channels,
                  (SELECT count(*) FROM channels WHERE company_match IS TRUE) AS company_match_channels,
                  (SELECT count(*) FROM channels WHERE ai_llm_match IS TRUE) AS ai_llm_channels,
                  (SELECT count(*) FROM channels WHERE recent_jobs IS TRUE) AS recent_job_channels,
                  (SELECT count(*) FROM discovered_channels WHERE last_seen_at >= now() - interval '24 hours') AS candidates_24h,
                  (SELECT count(*) FROM messages WHERE posted_at >= now() - interval '24 hours') AS messages_24h
                """
            )
        )
    ).mappings().one()
    top_sources = (
        await session.execute(
            sql_text(
                """
                SELECT source, count(DISTINCT lower(username)) AS count
                FROM discovered_channels
                GROUP BY source
                ORDER BY count DESC, source
                LIMIT 8
                """
            )
        )
    ).mappings().all()
    task_rows = (
        await session.execute(
            sql_text(
                """
                SELECT
                  name,
                  enabled,
                  running,
                  total_runs,
                  COALESCE(dc.candidates, total_candidates) AS total_candidates,
                  COALESCE(dc.valid_candidates, total_valid_candidates) AS total_valid_candidates,
                  COALESCE(dc.junk_candidates, total_junk_candidates) AS total_junk_candidates,
                  CASE
                    WHEN COALESCE(dc.candidates, total_candidates) > 0
                    THEN COALESCE(dc.valid_candidates, total_valid_candidates)::float / COALESCE(dc.candidates, total_candidates)
                    ELSE 0
                  END AS precision,
                  total_channels_crawled,
                  total_messages_saved,
                  last_candidates_found,
                  last_valid_candidates,
                  last_junk_candidates,
                  last_channels_crawled,
                  last_messages_saved,
                  last_run_at,
                  next_run_at
                FROM search_tasks st
                LEFT JOIN LATERAL (
                  SELECT
                    count(DISTINCT lower(username)) AS candidates,
                    count(DISTINCT lower(username)) FILTER (WHERE state IN ('validated_channel', 'indexed_channel')) AS valid_candidates,
                    count(DISTINCT lower(username)) FILTER (WHERE state IN ('rejected_user_or_bot', 'empty_public')) AS junk_candidates
                  FROM discovered_channels dc
                  WHERE dc.task_name = st.name
                ) dc ON true
                ORDER BY running DESC, enabled DESC, last_run_at DESC NULLS LAST, name
                LIMIT 8
                """
            )
        )
    ).mappings().all()
    states = (
        await session.execute(
            sql_text(
                """
                SELECT state, count(DISTINCT lower(username)) AS count
                FROM discovered_channels
                GROUP BY state
                ORDER BY count DESC, state
                """
            )
        )
    ).mappings().all()
    quality = (
        await session.execute(
            sql_text(
                """
                SELECT
                  avg(quality_score) AS avg_quality,
                  percentile_disc(0.95) WITHIN GROUP (ORDER BY quality_score) AS p95_quality,
                  count(*) FILTER (WHERE public_chat) AS public_chat,
                  count(*) FILTER (WHERE recent_jobs) AS recent_jobs,
                  count(*) FILTER (WHERE company_match) AS company_match,
                  count(*) FILTER (WHERE ai_llm_match) AS ai_llm_match,
                  count(*) FILTER (WHERE direct_contact) AS direct_contact
                FROM channels
                """
            )
        )
    ).mappings().one()
    return {
        "totals": dict(totals),
        "top_sources": [dict(row) for row in top_sources],
        "states": [dict(row) for row in states],
        "quality": dict(quality),
        "tasks": [dict(row) for row in task_rows],
    }


@router.get("/core/stats")
async def core_stats(session: AsyncSession = Depends(get_session)):
    return await stats(session)
