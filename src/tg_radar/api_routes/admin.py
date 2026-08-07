from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy import or_, select, text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from tg_radar.api_routes.deps import get_session, get_state
from tg_radar.app_state import AppState
from tg_radar.config import get_settings
from tg_radar.db import Channel, Message, VacancyCluster, maybe_update_operator_quality, update_channel_quality
from tg_radar.schemas import ParsedMessage
from tg_radar.scoring import hiring_signal, vacancy_key


router = APIRouter()


def message_to_parsed(channel: Channel, message: Message) -> ParsedMessage:
    return ParsedMessage(
        channel_username=channel.username,
        channel_title=channel.title,
        tg_msg_id=message.tg_msg_id,
        url=message.url,
        posted_at=message.posted_at,
        text=message.text,
        views=message.views,
        links=message.links or [],
        mentions=message.mentions or [],
        forward_from=message.forward_from,
        content_hash=message.hash,
    )


@router.get("/vespa/status")
async def vespa_status(state: AppState = Depends(get_state)):
    try:
        data = await state.vespa.health()
        return {"ok": True, "status": data}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


@router.post("/admin/vespa/reindex")
async def vespa_reindex(
    limit: int = 1000,
    state: AppState = Depends(get_state),
    session: AsyncSession = Depends(get_session),
):
    rows = (
        await session.execute(
            select(Message, Channel)
            .join(Channel, Channel.id == Message.channel_id)
            .where(Message.indexed_at.is_(None))
            .order_by(Message.posted_at.desc().nulls_last(), Message.id.desc())
            .limit(min(max(limit, 1), 10_000))
        )
    ).all()
    indexed = 0
    errors: list[str] = []
    for message, channel in rows:
        parsed = message_to_parsed(channel, message)
        try:
            await state.vespa.feed_message(parsed, quality_score=channel.quality_score or 0.0)
            message.indexed_at = datetime.utcnow()
            indexed += 1
        except Exception as exc:
            errors.append(f"{message.url}: {type(exc).__name__}: {exc}")
            if len(errors) >= 5:
                break
    return {"requested": len(rows), "indexed": indexed, "errors": errors}


@router.post("/admin/backfill/quality")
async def backfill_quality(limit: int = 5000, session: AsyncSession = Depends(get_session)):
    rows = (
        await session.execute(
            select(Message, Channel)
            .join(Channel, Channel.id == Message.channel_id)
            .where((Message.vacancy_key.is_(None)) | (Message.detector_reasons.is_(None)))
            .order_by(Message.id)
            .limit(min(max(limit, 1), 20_000))
        )
    ).all()
    touched_channels: set[int] = set()
    clusters: dict[str, VacancyCluster] = {}
    for message, channel in rows:
        key = vacancy_key(message.text, message.url)
        signal = hiring_signal(" ".join([message.text or "", *(message.links or []), *(message.mentions or [])]))
        cluster = clusters.get(key)
        if cluster is None:
            cluster = await session.scalar(select(VacancyCluster).where(VacancyCluster.vacancy_key == key))
            if cluster is None:
                cluster = VacancyCluster(vacancy_key=key, canonical_url=message.url, message_count=0)
                session.add(cluster)
                await session.flush()
            clusters[key] = cluster
        message.vacancy_key = key
        message.detector_score = signal.score
        message.detector_reasons = signal.reasons
        message.cluster_id = cluster.id
        touched_channels.add(channel.id)
    await session.execute(
        sql_text(
            """
            UPDATE vacancy_clusters vc
            SET message_count = counted.count,
                last_seen_at = now()
            FROM (
                SELECT cluster_id, count(*) AS count
                FROM messages
                WHERE cluster_id IS NOT NULL
                GROUP BY cluster_id
            ) counted
            WHERE vc.id = counted.cluster_id
            """
        )
    )
    channel_filter = or_(Channel.quality_score == 0, Channel.id.in_(touched_channels)) if touched_channels else Channel.quality_score == 0
    channel_rows = (
        await session.execute(
            select(Channel)
            .where(channel_filter)
            .order_by(Channel.id)
            .limit(1000)
        )
    ).scalars().all()
    settings = get_settings()
    operator_quality_budget = settings.operator_quality_max_per_backfill
    quality_updated = 0
    for channel in channel_rows:
        msg_rows = (
            await session.execute(
                select(Message)
                .where(Message.channel_id == channel.id)
                .order_by(Message.posted_at.desc().nulls_last(), Message.id.desc())
                .limit(50)
            )
        ).scalars().all()
        parsed = [message_to_parsed(channel, message) for message in msg_rows]
        if parsed:
            await update_channel_quality(session, channel, parsed, channel.source)
            if operator_quality_budget > 0:
                await maybe_update_operator_quality(session, channel, parsed, settings)
                operator_quality_budget -= 1
            quality_updated += 1
    return {"messages_backfilled": len(rows), "clusters_touched": len(clusters), "channels_quality_updated": quality_updated}
