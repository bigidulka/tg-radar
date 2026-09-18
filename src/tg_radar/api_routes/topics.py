from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from tg_radar.api_routes.deps import get_session, get_state
from tg_radar.app_state import AppState
from tg_radar.db import (
    Channel,
    Message,
    ResearchTopic,
    TopicChannel,
    TopicMessage,
    SearchTask,
    link_topic_channel,
    link_topic_message,
    list_research_topics,
    upsert_research_topic,
)
from tg_radar.schemas import (
    PainInsight,
    PainReport,
    ResearchTopicRequest,
    ResearchTopicStatus,
    SearchTaskRequest,
    StoryMatchReport,
    TopicFeedItem,
    TopicFeedResponse,
)
from tg_radar.story_match import StoryMatchError, match_topic_stories
from tg_radar.topical import classify_pain, topic_relevance


router = APIRouter()


@router.post("/topics", response_model=ResearchTopicStatus)
@router.post("/core/topics", response_model=ResearchTopicStatus)
async def upsert_topic(
    payload: ResearchTopicRequest,
    state: AppState = Depends(get_state),
    session: AsyncSession = Depends(get_session),
):
    topic = await upsert_research_topic(session, payload)
    await state.auto_search.upsert_task(
        SearchTaskRequest(
            name=topic.slug,
            keywords=topic.keywords or [],
            seed_channels=topic.seed_channels or [],
            auto_tune=True,
            depth=1,
            limit=80,
            pages_per_channel=1,
            crawl=True,
            interval_seconds=180,
            enabled=topic.enabled,
        )
    )
    for item in await list_research_topics(session):
        if item.slug == topic.slug:
            return item
    return ResearchTopicStatus(slug=topic.slug, title=topic.title, keywords=topic.keywords or [], enabled=topic.enabled)


@router.get("/topics", response_model=list[ResearchTopicStatus])
@router.get("/core/topics", response_model=list[ResearchTopicStatus])
async def topics(session: AsyncSession = Depends(get_session)):
    return await list_research_topics(session)


@router.post("/topics/{slug}/backfill")
@router.post("/core/topics/{slug}/backfill")
async def topic_backfill(slug: str, limit: int = 5000, session: AsyncSession = Depends(get_session)):
    running_tasks = await _running_tasks(session)
    if running_tasks:
        raise HTTPException(status_code=409, detail=f"backfill requires idle engine; {running_tasks} tasks still running")
    topic = await session.scalar(select(ResearchTopic).where(ResearchTopic.slug == slug))
    if not topic:
        raise HTTPException(status_code=404, detail="topic not found")
    rows = (
        await session.execute(
            select(Message, Channel)
            .join(Channel, Channel.id == Message.channel_id)
            .order_by(Message.posted_at.desc().nulls_last(), Message.id.desc())
            .limit(min(max(limit, 1), 50_000))
        )
    ).all()
    linked_channels: set[int] = set()
    linked_messages = 0
    for message, channel in rows:
        pain = classify_pain(message.text or "", message.links or [], message.mentions or [])
        if not message.pain_reasons:
            message.pain_score = pain.score
            message.pain_type = pain.pain_type
            message.intent = pain.intent
            message.pain_reasons = pain.reasons
            message.discussion_key = pain.discussion_key
        relevance = topic_relevance(
            " ".join([message.text or "", *(message.links or []), *(message.mentions or []), channel.title or ""]),
            topic.keywords or [],
            topic.negative_keywords or [],
        )
        if relevance <= 0 and (message.pain_score or 0.0) < 0.28:
            continue
        await link_topic_channel(session, topic, channel, max(relevance, channel.quality_score or 0.0), ["backfill"])
        await link_topic_message(session, topic, message, relevance)
        linked_channels.add(channel.id)
        linked_messages += 1
    topic.updated_at = datetime.utcnow()
    return {"topic": slug, "scanned": len(rows), "channels": len(linked_channels), "messages": linked_messages}


@router.get("/topics/{slug}/channels")
@router.get("/core/topics/{slug}/channels")
async def topic_channels(slug: str, limit: int = 200, session: AsyncSession = Depends(get_session)):
    topic = await session.scalar(select(ResearchTopic).where(ResearchTopic.slug == slug))
    if not topic:
        raise HTTPException(status_code=404, detail="topic not found")
    rows = (
        await session.execute(
            select(Channel, TopicChannel)
            .join(TopicChannel, TopicChannel.channel_id == Channel.id)
            .where(TopicChannel.topic_id == topic.id)
            .order_by(TopicChannel.score.desc(), TopicChannel.last_seen_at.desc())
            .limit(min(max(limit, 1), 1000))
        )
    ).all()
    return {
        "topic": slug,
        "channels": [
            {
                "username": channel.username,
                "title": channel.title,
                "score": topic_channel.score,
                "reasons": topic_channel.reasons or [],
                "quality_score": channel.quality_score,
                "public_chat": channel.public_chat,
                "last_seen_at": topic_channel.last_seen_at,
            }
            for channel, topic_channel in rows
        ],
    }


@router.get("/topics/{slug}/feed", response_model=TopicFeedResponse)
@router.get("/core/topics/{slug}/feed", response_model=TopicFeedResponse)
async def topic_feed(slug: str, after_id: int = 0, limit: int = 100, session: AsyncSession = Depends(get_session)):
    topic = await session.scalar(select(ResearchTopic).where(ResearchTopic.slug == slug))
    if not topic:
        raise HTTPException(status_code=404, detail="topic not found")
    page_size = min(max(limit, 1), 500)
    rows = (
        await session.execute(
            select(Message, Channel)
            .join(TopicMessage, TopicMessage.message_id == Message.id)
            .join(Channel, Channel.id == Message.channel_id)
            .where(
                TopicMessage.topic_id == topic.id,
                Message.id > after_id,
                Message.deleted_or_missing.is_(False),
            )
            .order_by(Message.id.asc())
            .limit(page_size + 1)
        )
    ).all()
    items = [
        TopicFeedItem(
            id=message.id,
            channel=channel.username,
            url=message.url,
            text=message.text,
            posted_at=message.posted_at,
            views=message.views,
            pain_score=message.pain_score,
            tags=message.pain_reasons or [],
        )
        for message, channel in rows[:page_size]
    ]
    return TopicFeedResponse(topic=slug, items=items, next_after_id=items[-1].id if len(rows) > page_size else None)


@router.get("/topics/{slug}/pain", response_model=PainReport)
@router.get("/core/topics/{slug}/pain", response_model=PainReport)
async def topic_pain(slug: str, limit: int = 100, session: AsyncSession = Depends(get_session)):
    topic = await session.scalar(select(ResearchTopic).where(ResearchTopic.slug == slug))
    if not topic:
        raise HTTPException(status_code=404, detail="topic not found")
    rows = (
        await session.execute(
            select(Message, Channel, TopicMessage)
            .join(TopicMessage, TopicMessage.message_id == Message.id)
            .join(Channel, Channel.id == Message.channel_id)
            .where(TopicMessage.topic_id == topic.id, TopicMessage.pain_score >= 0.28)
            .order_by(TopicMessage.pain_score.desc(), Message.posted_at.desc().nulls_last())
            .limit(min(max(limit, 1), 1000))
        )
    ).all()
    group_rows = (
        await session.execute(
            select(TopicMessage.pain_type, func.count())
            .where(TopicMessage.topic_id == topic.id, TopicMessage.pain_score >= 0.28)
            .group_by(TopicMessage.pain_type)
            .order_by(func.count().desc())
        )
    ).all()
    items = [
        PainInsight(
            message_id=f"{channel.username}:{message.tg_msg_id}",
            channel_username=channel.username,
            channel_title=channel.title,
            url=message.url,
            posted_at=message.posted_at,
            text=message.text,
            pain_score=message.pain_score or 0.0,
            pain_type=message.pain_type or "discussion",
            intent=message.intent or "discussion",
            reasons=message.pain_reasons or [],
            highlights=classify_pain(message.text or "", message.links or [], message.mentions or []).highlights,
            discussion_key=message.discussion_key or "",
            thread_id=message.thread_id,
            reply_to_msg_id=message.reply_to_msg_id,
        )
        for message, channel, _ in rows
    ]
    return PainReport(
        topic=slug,
        total=sum(int(row[1] or 0) for row in group_rows),
        groups=[{"name": row[0] or "discussion", "count": int(row[1] or 0)} for row in group_rows],
        items=items,
    )


@router.get("/topics/{slug}/story-matches", response_model=StoryMatchReport)
@router.get("/core/topics/{slug}/story-matches", response_model=StoryMatchReport)
async def topic_story_matches(
    slug: str,
    window_days: int | None = None,
    limit: int | None = None,
    min_confidence: str = "medium",
    state: AppState = Depends(get_state),
    session: AsyncSession = Depends(get_session),
):
    try:
        return await match_topic_stories(
            session,
            state.settings,
            slug,
            window_days=window_days,
            limit=limit,
            min_confidence=min_confidence,
        )
    except StoryMatchError as exc:
        raise HTTPException(status_code=503 if "llm" in str(exc) else 404, detail=str(exc)) from exc


async def _running_tasks(session: AsyncSession) -> int:
    return int(await session.scalar(select(func.count()).select_from(SearchTask).where(SearchTask.running.is_(True))) or 0)
