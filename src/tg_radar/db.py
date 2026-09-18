from datetime import datetime, timedelta, timezone
from typing import AsyncIterator

from sqlalchemy import BigInteger, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint, func, select, text as sql_text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from tg_radar.config import Settings
from tg_radar.matching import word_tokens
from tg_radar.operator_quality import OperatorQualityAnalyzer
from tg_radar.schemas import CandidateState, ChannelCandidate, ChannelStatus, EdgeType, ParsedMessage, ResearchTopicRequest, ResearchTopicStatus, SearchFilters, SearchHit, SearchTaskRequest, SearchTaskStatus
from tg_radar.scoring import channel_quality, hiring_signal, source_trust, vacancy_key
from tg_radar.topical import classify_pain


class Base(DeclarativeBase):
    pass


class Channel(Base):
    __tablename__ = "channels"
    __table_args__ = (Index("uq_channels_username_lower", sql_text("lower(username)"), unique=True),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    title: Mapped[str | None] = mapped_column(String(512))
    about: Mapped[str | None] = mapped_column(Text)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_crawled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    status: Mapped[str] = mapped_column(String(32), default=ChannelStatus.pending.value, index=True)
    source: Mapped[str | None] = mapped_column(String(256))
    quality_score: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    public_chat: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    recent_jobs: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    company_match: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    ai_llm_match: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    direct_contact: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    source_trust: Mapped[float] = mapped_column(Float, default=0.0)
    quality_reasons: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    crawl_priority: Mapped[int] = mapped_column(Integer, default=100, index=True)
    crawl_interval_seconds: Mapped[int] = mapped_column(Integer, default=1800)
    last_successful_crawl_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    last_error_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)
    operator_niche_category: Mapped[str | None] = mapped_column(String(32), index=True)
    operator_income_authenticity: Mapped[str | None] = mapped_column(String(32))
    operator_niche_tags: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    operator_niche_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    operator_niche_evidence: Mapped[dict] = mapped_column(JSONB, default=dict)
    operator_niche_notes: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    operator_niche_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)

    messages: Mapped[list["Message"]] = relationship(back_populates="channel")


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        UniqueConstraint("channel_id", "tg_msg_id", name="uq_messages_channel_msg"),
        UniqueConstraint("hash", name="uq_messages_hash"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    channel_id: Mapped[int] = mapped_column(ForeignKey("channels.id", ondelete="CASCADE"), index=True)
    tg_msg_id: Mapped[int] = mapped_column(BigInteger)
    url: Mapped[str] = mapped_column(String(512), unique=True)
    posted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    text: Mapped[str] = mapped_column(Text)
    hash: Mapped[str] = mapped_column(String(64), index=True)
    views: Mapped[int | None] = mapped_column(Integer)
    links: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    mentions: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    forward_from: Mapped[str | None] = mapped_column(String(512))
    sender_id: Mapped[int | None] = mapped_column(BigInteger)
    sender_name: Mapped[str | None] = mapped_column(String(512))
    thread_id: Mapped[int | None] = mapped_column(BigInteger, index=True)
    reply_to_msg_id: Mapped[int | None] = mapped_column(BigInteger, index=True)
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    vacancy_key: Mapped[str | None] = mapped_column(String(64), index=True)
    detector_score: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    detector_reasons: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    cluster_id: Mapped[int | None] = mapped_column(ForeignKey("vacancy_clusters.id", ondelete="SET NULL"), index=True)
    pain_score: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    pain_type: Mapped[str | None] = mapped_column(String(64), index=True)
    intent: Mapped[str | None] = mapped_column(String(64), index=True)
    pain_reasons: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    discussion_key: Mapped[str | None] = mapped_column(String(64), index=True)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    edited_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    deleted_or_missing: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    last_refreshed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    source_snapshot_hash: Mapped[str | None] = mapped_column(String(64), index=True)

    channel: Mapped[Channel] = relationship(back_populates="messages")


class Edge(Base):
    __tablename__ = "edges"
    __table_args__ = (UniqueConstraint("src_channel_id", "dst_username", "edge_type", "msg_id", name="uq_edges"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    src_channel_id: Mapped[int] = mapped_column(ForeignKey("channels.id", ondelete="CASCADE"), index=True)
    dst_username: Mapped[str] = mapped_column(String(32), index=True)
    edge_type: Mapped[str] = mapped_column(String(32), index=True)
    msg_id: Mapped[int | None] = mapped_column(ForeignKey("messages.id", ondelete="CASCADE"))
    discovered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class CrawlJob(Base):
    __tablename__ = "crawl_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    channel_id: Mapped[int] = mapped_column(ForeignKey("channels.id", ondelete="CASCADE"), unique=True)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    priority: Mapped[int] = mapped_column(Integer, default=100, index=True)
    next_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    error: Mapped[str | None] = mapped_column(Text)


class DiscoveredChannel(Base):
    __tablename__ = "discovered_channels"
    __table_args__ = (UniqueConstraint("username", "source", "reason", name="uq_discovered_channel_source_reason"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(32), index=True)
    source: Mapped[str] = mapped_column(String(128), index=True)
    reason: Mapped[str] = mapped_column(Text)
    score: Mapped[float] = mapped_column(Float, default=1.0)
    depth: Mapped[int] = mapped_column(Integer, default=0, index=True)
    task_name: Mapped[str | None] = mapped_column(String(128), index=True)
    state: Mapped[str] = mapped_column(String(32), default=CandidateState.raw_candidate.value, index=True)
    quality_score: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    quality_reasons: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    operator_niche_category: Mapped[str | None] = mapped_column(String(32), index=True)
    operator_niche_tags: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    operator_niche_confidence: Mapped[float] = mapped_column(Float, default=0.0)


class SearchTask(Base):
    __tablename__ = "search_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    topic_slug: Mapped[str | None] = mapped_column(String(128), index=True)
    keywords: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    seed_channels: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    auto_tune: Mapped[bool] = mapped_column(Boolean, default=True)
    depth: Mapped[int] = mapped_column(Integer, default=1)
    limit: Mapped[int] = mapped_column(Integer, default=50)
    pages_per_channel: Mapped[int] = mapped_column(Integer, default=1)
    crawl: Mapped[bool] = mapped_column(Boolean, default=True)
    crawl_mode: Mapped[str] = mapped_column(String(32), default="backfill", index=True)
    freshness_days: Mapped[int | None] = mapped_column(Integer)
    interval_seconds: Mapped[int] = mapped_column(Integer, default=300)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    running: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    running_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    next_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    last_error: Mapped[str | None] = mapped_column(Text)
    total_runs: Mapped[int] = mapped_column(Integer, default=0)
    total_candidates: Mapped[int] = mapped_column(Integer, default=0)
    total_valid_candidates: Mapped[int] = mapped_column(Integer, default=0)
    total_junk_candidates: Mapped[int] = mapped_column(Integer, default=0)
    total_channels_crawled: Mapped[int] = mapped_column(Integer, default=0)
    total_messages_saved: Mapped[int] = mapped_column(Integer, default=0)
    last_candidates_found: Mapped[int] = mapped_column(Integer, default=0)
    last_valid_candidates: Mapped[int] = mapped_column(Integer, default=0)
    last_junk_candidates: Mapped[int] = mapped_column(Integer, default=0)
    last_channels_crawled: Mapped[int] = mapped_column(Integer, default=0)
    last_messages_saved: Mapped[int] = mapped_column(Integer, default=0)
    last_edges_saved: Mapped[int] = mapped_column(Integer, default=0)


class VacancyCluster(Base):
    __tablename__ = "vacancy_clusters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    vacancy_key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    canonical_url: Mapped[str | None] = mapped_column(String(512))
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    message_count: Mapped[int] = mapped_column(Integer, default=0, index=True)


class TelethonEntityCache(Base):
    __tablename__ = "telethon_entity_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    is_chat: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    entity_type: Mapped[str | None] = mapped_column(String(32), index=True)
    title: Mapped[str | None] = mapped_column(String(512))
    checked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    error: Mapped[str | None] = mapped_column(Text)


class ResearchTopic(Base):
    __tablename__ = "research_topics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    slug: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    title: Mapped[str | None] = mapped_column(String(512))
    keywords: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    negative_keywords: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    seed_channels: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


class TopicChannel(Base):
    __tablename__ = "topic_channels"
    __table_args__ = (UniqueConstraint("topic_id", "channel_id", name="uq_topic_channel"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    topic_id: Mapped[int] = mapped_column(ForeignKey("research_topics.id", ondelete="CASCADE"), index=True)
    channel_id: Mapped[int] = mapped_column(ForeignKey("channels.id", ondelete="CASCADE"), index=True)
    score: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    reasons: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


class TopicMessage(Base):
    __tablename__ = "topic_messages"
    __table_args__ = (UniqueConstraint("topic_id", "message_id", name="uq_topic_message"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    topic_id: Mapped[int] = mapped_column(ForeignKey("research_topics.id", ondelete="CASCADE"), index=True)
    message_id: Mapped[int] = mapped_column(ForeignKey("messages.id", ondelete="CASCADE"), index=True)
    relevance_score: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    pain_score: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    pain_type: Mapped[str | None] = mapped_column(String(64), index=True)
    intent: Mapped[str | None] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


class ContentCard(Base):
    __tablename__ = "content_cards"
    __table_args__ = (UniqueConstraint("fingerprint", name="uq_content_card_fingerprint"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    topic_slug: Mapped[str] = mapped_column(String(128), index=True)
    card_type: Mapped[str] = mapped_column(String(32), index=True)
    title: Mapped[str] = mapped_column(String(512))
    summary: Mapped[str] = mapped_column(Text)
    body: Mapped[str] = mapped_column(Text)
    source_channel: Mapped[str | None] = mapped_column(String(32), index=True)
    source_url: Mapped[str | None] = mapped_column(String(512))
    score: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    status: Mapped[str] = mapped_column(String(32), default="new", index=True)
    tags: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)
    fingerprint: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


class ArticleSession(Base):
    __tablename__ = "article_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    topic_slug: Mapped[str] = mapped_column(String(128), index=True)
    title: Mapped[str] = mapped_column(String(512))
    audience: Mapped[str | None] = mapped_column(String(512))
    angle: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="outline", index=True)
    source_card_ids: Mapped[list[int]] = mapped_column(ARRAY(Integer), default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


class ArticleBlock(Base):
    __tablename__ = "article_blocks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("article_sessions.id", ondelete="CASCADE"), index=True)
    position: Mapped[int] = mapped_column(Integer, default=0, index=True)
    block_type: Mapped[str] = mapped_column(String(32), default="paragraph", index=True)
    title: Mapped[str | None] = mapped_column(String(512))
    text: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="draft", index=True)
    source_card_ids: Mapped[list[int]] = mapped_column(ARRAY(Integer), default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


class ArticleDraft(Base):
    __tablename__ = "article_drafts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("article_sessions.id", ondelete="CASCADE"), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1, index=True)
    status: Mapped[str] = mapped_column(String(32), default="draft", index=True)
    format: Mapped[str] = mapped_column(String(32), default="telegram_markdown", index=True)
    text: Mapped[str] = mapped_column(Text)
    rich_json: Mapped[dict] = mapped_column(JSONB, default=dict)
    score: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


class AgentEventLog(Base):
    __tablename__ = "agent_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(512), index=True)
    step_index: Mapped[int] = mapped_column(Integer, index=True)
    event: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


class AgentRun(Base):
    __tablename__ = "agent_runs"

    run_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    request_json: Mapped[dict] = mapped_column(JSONB, default=dict)
    final: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    worker_id: Mapped[str | None] = mapped_column(String(128), index=True)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False, index=True)


class TaskRun(Base):
    __tablename__ = "task_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    task_name: Mapped[str] = mapped_column(String(128), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    ingest_stats: Mapped[dict] = mapped_column(JSONB, default=dict)
    error: Mapped[str | None] = mapped_column(Text)


class AgentEvalRun(Base):
    __tablename__ = "agent_eval_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model: Mapped[str | None] = mapped_column(String(128), index=True)
    prompt_hash: Mapped[str | None] = mapped_column(String(128), index=True)
    harness_config: Mapped[dict] = mapped_column(JSONB, default=dict)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    score: Mapped[float] = mapped_column(Float, default=0.0, index=True)


class AgentEvalCaseRecord(Base):
    __tablename__ = "agent_eval_cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    eval_run_id: Mapped[int] = mapped_column(ForeignKey("agent_eval_runs.id", ondelete="CASCADE"), index=True)
    case_name: Mapped[str] = mapped_column(String(256), index=True)
    request: Mapped[dict] = mapped_column(JSONB, default=dict)
    expected_tools: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    actual_tools: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    final: Mapped[str | None] = mapped_column(Text)
    failures: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)


class WorkerHeartbeat(Base):
    __tablename__ = "worker_heartbeats"

    worker_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    role: Mapped[str] = mapped_column(String(64), default="worker", index=True)
    heartbeat_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)


def make_engine(settings: Settings) -> AsyncEngine:
    return create_async_engine(settings.database_url, pool_pre_ping=True)


def make_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_db(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _ensure_compat_schema(conn)


async def _ensure_compat_schema(conn) -> None:
    statements = [
        "ALTER TABLE channels ADD COLUMN IF NOT EXISTS crawl_priority INTEGER DEFAULT 100 NOT NULL",
        "ALTER TABLE channels ADD COLUMN IF NOT EXISTS crawl_interval_seconds INTEGER DEFAULT 1800 NOT NULL",
        "ALTER TABLE channels ADD COLUMN IF NOT EXISTS last_successful_crawl_at TIMESTAMP WITH TIME ZONE",
        "ALTER TABLE channels ADD COLUMN IF NOT EXISTS last_error_at TIMESTAMP WITH TIME ZONE",
        "ALTER TABLE channels ADD COLUMN IF NOT EXISTS last_error TEXT",
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS last_seen_at TIMESTAMP WITH TIME ZONE",
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS edited_at TIMESTAMP WITH TIME ZONE",
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS deleted_or_missing BOOLEAN DEFAULT false NOT NULL",
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS last_refreshed_at TIMESTAMP WITH TIME ZONE",
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS source_snapshot_hash VARCHAR(64)",
        "ALTER TABLE search_tasks ADD COLUMN IF NOT EXISTS crawl_mode VARCHAR(32) DEFAULT 'backfill' NOT NULL",
        "ALTER TABLE search_tasks ADD COLUMN IF NOT EXISTS freshness_days INTEGER",
        "ALTER TABLE search_tasks ADD COLUMN IF NOT EXISTS running_started_at TIMESTAMP WITH TIME ZONE",
        "ALTER TABLE search_tasks ADD COLUMN IF NOT EXISTS topic_slug VARCHAR(128)",
        "CREATE INDEX IF NOT EXISTS idx_search_tasks_topic_slug ON search_tasks (topic_slug)",
        """
        CREATE TABLE IF NOT EXISTS agent_runs (
            run_id VARCHAR(128) PRIMARY KEY,
            status VARCHAR(32) DEFAULT 'queued' NOT NULL,
            request_json JSONB DEFAULT '{}'::jsonb NOT NULL,
            final TEXT,
            error TEXT,
            started_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
            finished_at TIMESTAMP WITH TIME ZONE,
            locked_at TIMESTAMP WITH TIME ZONE,
            worker_id VARCHAR(128),
            cancel_requested BOOLEAN DEFAULT false NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS task_runs (
            id SERIAL PRIMARY KEY,
            task_name VARCHAR(128) NOT NULL,
            status VARCHAR(32) NOT NULL,
            started_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
            finished_at TIMESTAMP WITH TIME ZONE,
            ingest_stats JSONB DEFAULT '{}'::jsonb NOT NULL,
            error TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS agent_eval_runs (
            id SERIAL PRIMARY KEY,
            model VARCHAR(128),
            prompt_hash VARCHAR(128),
            harness_config JSONB DEFAULT '{}'::jsonb NOT NULL,
            started_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
            finished_at TIMESTAMP WITH TIME ZONE,
            score DOUBLE PRECISION DEFAULT 0 NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS agent_eval_cases (
            id SERIAL PRIMARY KEY,
            eval_run_id INTEGER REFERENCES agent_eval_runs(id) ON DELETE CASCADE NOT NULL,
            case_name VARCHAR(256) NOT NULL,
            request JSONB DEFAULT '{}'::jsonb NOT NULL,
            expected_tools VARCHAR[] DEFAULT '{}' NOT NULL,
            actual_tools VARCHAR[] DEFAULT '{}' NOT NULL,
            final TEXT,
            failures VARCHAR[] DEFAULT '{}' NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS worker_heartbeats (
            worker_id VARCHAR(128) PRIMARY KEY,
            role VARCHAR(64) DEFAULT 'worker' NOT NULL,
            heartbeat_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
            payload JSONB DEFAULT '{}'::jsonb NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_channels_crawl_policy ON channels (crawl_priority, last_successful_crawl_at)",
        "CREATE INDEX IF NOT EXISTS idx_messages_last_seen_at ON messages (last_seen_at)",
        "CREATE INDEX IF NOT EXISTS idx_messages_deleted_or_missing ON messages (deleted_or_missing)",
        "CREATE INDEX IF NOT EXISTS idx_messages_last_refreshed_at ON messages (last_refreshed_at)",
        "CREATE INDEX IF NOT EXISTS idx_agent_runs_status_locked ON agent_runs (status, locked_at)",
        "CREATE INDEX IF NOT EXISTS idx_task_runs_task_started ON task_runs (task_name, started_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_agent_eval_runs_started ON agent_eval_runs (started_at DESC)",
    ]
    for statement in statements:
        await conn.execute(sql_text(statement))


async def session_scope(factory: async_sessionmaker[AsyncSession]) -> AsyncIterator[AsyncSession]:
    async with factory() as session:
        async with session.begin():
            yield session


async def upsert_channel(session: AsyncSession, username: str, title: str | None, source: str) -> Channel:
    username = username.strip("@")
    existing = await session.scalar(select(Channel).where(func.lower(Channel.username) == username.lower()))
    if existing:
        existing.title = title or existing.title
        existing.last_seen_at = datetime.utcnow()
        existing.status = ChannelStatus.active.value
        return existing
    channel = Channel(username=username, title=title, status=ChannelStatus.active.value, source=source)
    session.add(channel)
    await session.flush()
    return channel


async def update_channel_quality(session: AsyncSession, channel: Channel, messages: list[ParsedMessage], source: str | None = None) -> None:
    quality = channel_quality(messages, source or channel.source, public_chat=True)
    channel.quality_score = max(channel.quality_score or 0.0, quality.score)
    channel.public_chat = quality.public_chat
    channel.recent_jobs = channel.recent_jobs or quality.recent_jobs
    channel.company_match = channel.company_match or quality.company_match
    channel.ai_llm_match = channel.ai_llm_match or quality.ai_llm_match
    channel.direct_contact = channel.direct_contact or quality.direct_contact
    channel.source_trust = max(channel.source_trust or 0.0, quality.source_trust)
    channel.quality_reasons = sorted(set((channel.quality_reasons or []) + quality.reasons))
    await session.execute(
        sql_text(
            """
            UPDATE discovered_channels
            SET state = CASE WHEN :indexed THEN 'indexed_channel' ELSE 'validated_channel' END,
                quality_score = GREATEST(quality_score, :quality_score),
                quality_reasons = (
                    SELECT ARRAY(SELECT DISTINCT unnest(coalesce(quality_reasons, '{}') || :quality_reasons))
                ),
                last_seen_at = now()
            WHERE lower(username) = lower(:username)
            """
        ),
        {
            "username": channel.username,
            "indexed": bool(messages),
            "quality_score": quality.score,
            "quality_reasons": quality.reasons,
        },
    )


async def maybe_update_operator_quality(
    session: AsyncSession,
    channel: Channel,
    messages: list[ParsedMessage],
    settings: Settings,
    analyzer: OperatorQualityAnalyzer | None = None,
) -> None:
    if not settings.operator_quality_enabled:
        return
    if len(messages) < settings.operator_quality_min_messages:
        return
    updated_at = channel.operator_niche_updated_at
    if updated_at:
        if not updated_at.tzinfo:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        threshold = datetime.now(timezone.utc) - timedelta(days=settings.operator_quality_recheck_interval_days)
        if updated_at > threshold:
            return
    analyzer = analyzer or OperatorQualityAnalyzer(settings)
    try:
        result = await analyzer.analyze(channel.username, messages)
    except Exception:
        return
    channel.operator_niche_category = result["niche_category"]
    channel.operator_income_authenticity = result["income_authenticity"]
    channel.operator_niche_tags = result["niche_tags"]
    channel.operator_niche_confidence = result["confidence"]
    channel.operator_niche_evidence = {"items": result["evidence"]}
    channel.operator_niche_notes = result["notes"]
    channel.operator_niche_updated_at = datetime.now(timezone.utc)
    await session.execute(
        sql_text(
            """
            UPDATE discovered_channels
            SET operator_niche_category = :niche_category,
                operator_niche_tags = :niche_tags,
                operator_niche_confidence = :niche_confidence
            WHERE lower(username) = lower(:username)
            """
        ),
        {
            "username": channel.username,
            "niche_category": result["niche_category"],
            "niche_tags": result["niche_tags"],
            "niche_confidence": result["confidence"],
        },
    )


async def upsert_vacancy_cluster(session: AsyncSession, key: str, url: str) -> VacancyCluster:
    cluster = await session.scalar(select(VacancyCluster).where(VacancyCluster.vacancy_key == key))
    if cluster:
        cluster.last_seen_at = datetime.utcnow()
        cluster.message_count = (cluster.message_count or 0) + 1
        cluster.canonical_url = cluster.canonical_url or url
        return cluster
    cluster = VacancyCluster(vacancy_key=key, canonical_url=url, message_count=1)
    session.add(cluster)
    await session.flush()
    return cluster


async def upsert_message(session: AsyncSession, channel: Channel, parsed: ParsedMessage) -> tuple[Message, bool, bool]:
    now = datetime.utcnow()
    existing = await session.scalar(
        select(Message).where(
            ((Message.channel_id == channel.id) & (Message.tg_msg_id == parsed.tg_msg_id))
            | (Message.hash == parsed.content_hash)
            | (Message.url == parsed.url)
        )
    )
    if existing:
        changed = (
            existing.text != parsed.text
            or existing.hash != parsed.content_hash
            or existing.links != parsed.links
            or existing.mentions != parsed.mentions
        )
        existing.url = parsed.url or existing.url
        existing.posted_at = parsed.posted_at or existing.posted_at
        existing.views = parsed.views
        existing.links = parsed.links
        existing.mentions = parsed.mentions
        existing.forward_from = parsed.forward_from or existing.forward_from
        existing.sender_id = parsed.sender_id or existing.sender_id
        existing.sender_name = parsed.sender_name or existing.sender_name
        existing.thread_id = parsed.thread_id or existing.thread_id
        existing.reply_to_msg_id = parsed.reply_to_msg_id or existing.reply_to_msg_id
        existing.last_seen_at = now
        existing.last_refreshed_at = now
        existing.deleted_or_missing = False
        existing.source_snapshot_hash = parsed.content_hash
        if changed:
            existing.text = parsed.text
            existing.hash = parsed.content_hash
            existing.edited_at = now
            existing.indexed_at = None
            existing.vacancy_key = vacancy_key(existing.text, existing.url)
            signal = hiring_signal(" ".join([existing.text, *existing.links, *existing.mentions]))
            pain = classify_pain(existing.text, existing.links, existing.mentions)
            existing.detector_score = signal.score
            existing.detector_reasons = signal.reasons
            existing.pain_score = pain.score
            existing.pain_type = pain.pain_type
            existing.intent = pain.intent
            existing.pain_reasons = pain.reasons
            existing.discussion_key = pain.discussion_key
        else:
            pain = classify_pain(existing.text, existing.links, existing.mentions)
            if not existing.vacancy_key:
                existing.vacancy_key = vacancy_key(existing.text, existing.url)
            if not existing.detector_reasons:
                signal = hiring_signal(existing.text)
                existing.detector_score = signal.score
                existing.detector_reasons = signal.reasons
            if not existing.pain_reasons:
                existing.pain_score = pain.score
                existing.pain_type = pain.pain_type
                existing.intent = pain.intent
                existing.pain_reasons = pain.reasons
                existing.discussion_key = pain.discussion_key
        return existing, False, changed
    key = vacancy_key(parsed.text, parsed.url)
    signal = hiring_signal(" ".join([parsed.text, *parsed.links, *parsed.mentions]))
    pain = classify_pain(parsed.text, parsed.links, parsed.mentions)
    cluster = await upsert_vacancy_cluster(session, key, parsed.url)
    message = Message(
        channel_id=channel.id,
        tg_msg_id=parsed.tg_msg_id,
        url=parsed.url,
        posted_at=parsed.posted_at,
        text=parsed.text,
        hash=parsed.content_hash,
        views=parsed.views,
        links=parsed.links,
        mentions=parsed.mentions,
        forward_from=parsed.forward_from,
        sender_id=parsed.sender_id,
        sender_name=parsed.sender_name,
        thread_id=parsed.thread_id,
        reply_to_msg_id=parsed.reply_to_msg_id,
        vacancy_key=key,
        detector_score=signal.score,
        detector_reasons=signal.reasons,
        cluster_id=cluster.id,
        pain_score=pain.score,
        pain_type=pain.pain_type,
        intent=pain.intent,
        pain_reasons=pain.reasons,
        discussion_key=pain.discussion_key,
        last_seen_at=now,
        last_refreshed_at=now,
        deleted_or_missing=False,
        source_snapshot_hash=parsed.content_hash,
    )
    session.add(message)
    await session.flush()
    return message, True, True


async def mark_refresh_missing_messages(
    session: AsyncSession,
    channel: Channel,
    seen_tg_msg_ids: set[int],
    latest_limit: int,
) -> int:
    if not seen_tg_msg_ids:
        return 0
    rows = (
        await session.execute(
            select(Message)
            .where(Message.channel_id == channel.id, Message.deleted_or_missing.is_(False))
            .order_by(Message.posted_at.desc().nulls_last(), Message.id.desc())
            .limit(max(latest_limit, 1))
        )
    ).scalars().all()
    now = datetime.utcnow()
    marked = 0
    for message in rows:
        if message.tg_msg_id in seen_tg_msg_ids:
            continue
        message.deleted_or_missing = True
        message.last_refreshed_at = now
        marked += 1
    return marked


async def channel_last_crawled_map(session: AsyncSession, usernames: list[str]) -> dict[str, datetime | None]:
    cleaned = [username.strip("@").lower() for username in usernames if username]
    if not cleaned:
        return {}
    rows = await session.execute(
        select(Channel.username, Channel.last_crawled_at).where(func.lower(Channel.username).in_(cleaned))
    )
    return {row[0].lower(): row[1] for row in rows}


async def channel_crawl_policy_map(session: AsyncSession, usernames: list[str]) -> dict[str, tuple[datetime | None, int]]:
    cleaned = [username.strip("@").lower() for username in usernames if username]
    if not cleaned:
        return {}
    rows = await session.execute(
        select(Channel.username, Channel.last_crawled_at, Channel.crawl_interval_seconds).where(
            func.lower(Channel.username).in_(cleaned)
        )
    )
    return {row[0].lower(): (row[1], int(row[2] or 0)) for row in rows}


async def add_edges_for_message(session: AsyncSession, channel: Channel, message: Message, mentions: list[str]) -> None:
    for username in mentions:
        existing = await session.scalar(
            select(Edge).where(
                Edge.src_channel_id == channel.id,
                Edge.dst_username == username,
                Edge.edge_type == EdgeType.mention.value,
                Edge.msg_id == message.id,
            )
        )
        if not existing:
            edge = Edge(
                src_channel_id=channel.id,
                dst_username=username,
                edge_type=EdgeType.mention.value,
                msg_id=message.id,
            )
            session.add(edge)


async def add_channel_edge(session: AsyncSession, src_username: str, dst_username: str, edge_type: EdgeType = EdgeType.link) -> None:
    src = await session.scalar(select(Channel).where(func.lower(Channel.username) == src_username.strip("@").lower()))
    dst = dst_username.strip("@")
    if not src or not dst or src.username.lower() == dst.lower():
        return
    existing = await session.scalar(
        select(Edge).where(
            Edge.src_channel_id == src.id,
            Edge.dst_username == dst,
            Edge.edge_type == edge_type.value,
            Edge.msg_id.is_(None),
        )
    )
    if not existing:
        session.add(Edge(src_channel_id=src.id, dst_username=dst, edge_type=edge_type.value, msg_id=None))


async def upsert_discovered_channel(session: AsyncSession, candidate: ChannelCandidate, task_name: str | None = None) -> None:
    if not candidate.username:
        return
    resolved_task_name = task_name or candidate.task_name
    existing = await session.scalar(
        select(DiscoveredChannel).where(
            func.lower(DiscoveredChannel.username) == candidate.username.strip("@").lower(),
            DiscoveredChannel.source == candidate.source,
            DiscoveredChannel.reason == candidate.reason,
        )
    )
    if existing:
        existing.score = max(existing.score, candidate.score)
        existing.depth = min(existing.depth, candidate.depth)
        existing.task_name = resolved_task_name or existing.task_name
        existing.state = candidate.state.value if hasattr(candidate.state, "value") else str(candidate.state or existing.state)
        existing.quality_score = max(existing.quality_score or 0.0, candidate.quality_score or 0.0)
        existing.quality_reasons = sorted(set((existing.quality_reasons or []) + (candidate.quality_reasons or [])))
        existing.last_seen_at = datetime.utcnow()
        return
    session.add(
        DiscoveredChannel(
            username=candidate.username,
            source=candidate.source,
            reason=candidate.reason,
            score=candidate.score,
            depth=candidate.depth,
            task_name=resolved_task_name,
            state=candidate.state.value if hasattr(candidate.state, "value") else str(candidate.state),
            quality_score=candidate.quality_score,
            quality_reasons=candidate.quality_reasons,
        )
    )


async def mark_discovered_state(
    session: AsyncSession,
    username: str,
    state: CandidateState,
    quality_score: float = 0.0,
    quality_reasons: list[str] | None = None,
) -> None:
    await session.execute(
        sql_text(
            """
            UPDATE discovered_channels
            SET state = :state,
                quality_score = GREATEST(quality_score, :quality_score),
                quality_reasons = (
                    SELECT ARRAY(SELECT DISTINCT unnest(coalesce(quality_reasons, '{}') || :quality_reasons))
                ),
                last_seen_at = now()
            WHERE lower(username) = lower(:username)
            """
        ),
        {
            "username": username.strip("@"),
            "state": state.value,
            "quality_score": quality_score,
            "quality_reasons": quality_reasons or [],
        },
    )


async def get_telethon_entity_cache(session: AsyncSession, username: str) -> TelethonEntityCache | None:
    return await session.scalar(select(TelethonEntityCache).where(TelethonEntityCache.username == username.strip("@")))


async def upsert_telethon_entity_cache(
    session: AsyncSession,
    username: str,
    is_chat: bool,
    entity_type: str | None = None,
    title: str | None = None,
    error: str | None = None,
) -> TelethonEntityCache:
    clean = username.strip("@")
    existing = await get_telethon_entity_cache(session, clean)
    if existing:
        existing.is_chat = is_chat
        existing.entity_type = entity_type
        existing.title = title or existing.title
        existing.error = error
        existing.checked_at = datetime.utcnow()
        return existing
    row = TelethonEntityCache(username=clean, is_chat=is_chat, entity_type=entity_type, title=title, error=error)
    session.add(row)
    await session.flush()
    return row


async def upsert_crawl_job(session: AsyncSession, channel: Channel, priority: int = 100, status: str = "pending", error: str | None = None) -> None:
    existing = await session.scalar(select(CrawlJob).where(CrawlJob.channel_id == channel.id))
    if existing:
        existing.priority = min(existing.priority, priority)
        existing.status = status
        existing.error = error
        return
    session.add(CrawlJob(channel_id=channel.id, priority=priority, status=status, error=error))


def task_to_status(task: SearchTask) -> SearchTaskStatus:
    auto_tune = getattr(task, "auto_tune", True)
    total_candidates = task.total_candidates or 0
    total_valid = getattr(task, "total_valid_candidates", 0) or 0
    return SearchTaskStatus(
        name=task.name,
        topic_slug=getattr(task, "topic_slug", None),
        keywords=task.keywords or [],
        seed_channels=task.seed_channels or [],
        auto_tune=True if auto_tune is None else auto_tune,
        depth=task.depth,
        limit=task.limit,
        pages_per_channel=task.pages_per_channel,
        crawl=task.crawl,
        crawl_mode=getattr(task, "crawl_mode", "backfill") or "backfill",
        freshness_days=getattr(task, "freshness_days", None),
        interval_seconds=task.interval_seconds,
        enabled=task.enabled,
        running=task.running,
        last_run_at=task.last_run_at,
        next_run_at=task.next_run_at,
        running_started_at=getattr(task, "running_started_at", None),
        last_error=task.last_error,
        total_runs=task.total_runs,
        total_candidates=total_candidates,
        total_valid_candidates=total_valid,
        total_junk_candidates=getattr(task, "total_junk_candidates", 0) or 0,
        precision=(total_valid / total_candidates) if total_candidates else 0.0,
        total_channels_crawled=task.total_channels_crawled,
        total_messages_saved=task.total_messages_saved,
        last_candidates_found=getattr(task, "last_candidates_found", 0) or 0,
        last_valid_candidates=getattr(task, "last_valid_candidates", 0) or 0,
        last_junk_candidates=getattr(task, "last_junk_candidates", 0) or 0,
        last_channels_crawled=getattr(task, "last_channels_crawled", 0) or 0,
        last_messages_saved=getattr(task, "last_messages_saved", 0) or 0,
        last_edges_saved=getattr(task, "last_edges_saved", 0) or 0,
    )


async def upsert_search_task(session: AsyncSession, payload: SearchTaskRequest) -> SearchTask:
    now = datetime.utcnow()
    task = await session.scalar(select(SearchTask).where(SearchTask.name == payload.name))
    if task:
        task.topic_slug = payload.topic_slug
        task.keywords = payload.keywords
        task.seed_channels = payload.seed_channels
        task.auto_tune = payload.auto_tune
        task.depth = payload.depth
        task.limit = payload.limit
        task.pages_per_channel = payload.pages_per_channel
        task.crawl = payload.crawl
        task.crawl_mode = payload.crawl_mode.value if hasattr(payload.crawl_mode, "value") else str(payload.crawl_mode)
        task.freshness_days = payload.freshness_days
        task.interval_seconds = payload.interval_seconds
        task.enabled = payload.enabled
        task.next_run_at = now if payload.enabled else None
        task.last_error = None
        return task
    task = SearchTask(
        name=payload.name,
        topic_slug=payload.topic_slug,
        keywords=payload.keywords,
        seed_channels=payload.seed_channels,
        auto_tune=payload.auto_tune,
        depth=payload.depth,
        limit=payload.limit,
        pages_per_channel=payload.pages_per_channel,
        crawl=payload.crawl,
        crawl_mode=payload.crawl_mode.value if hasattr(payload.crawl_mode, "value") else str(payload.crawl_mode),
        freshness_days=payload.freshness_days,
        interval_seconds=payload.interval_seconds,
        enabled=payload.enabled,
        next_run_at=now if payload.enabled else None,
    )
    session.add(task)
    await session.flush()
    return task


async def upsert_research_topic(session: AsyncSession, payload: ResearchTopicRequest) -> ResearchTopic:
    slug = payload.slug.strip().lower().replace(" ", "-")
    topic = await session.scalar(select(ResearchTopic).where(ResearchTopic.slug == slug))
    if topic:
        topic.title = payload.title or topic.title
        topic.keywords = payload.keywords
        topic.negative_keywords = payload.negative_keywords
        topic.seed_channels = payload.seed_channels
        topic.enabled = payload.enabled
        topic.updated_at = datetime.utcnow()
        return topic
    topic = ResearchTopic(
        slug=slug,
        title=payload.title or slug,
        keywords=payload.keywords,
        negative_keywords=payload.negative_keywords,
        seed_channels=payload.seed_channels,
        enabled=payload.enabled,
    )
    session.add(topic)
    await session.flush()
    return topic


async def ensure_research_topic(session: AsyncSession, slug: str, keywords: list[str], seed_channels: list[str] | None = None) -> ResearchTopic:
    clean = slug.strip().lower().replace(" ", "-")
    topic = await session.scalar(select(ResearchTopic).where(ResearchTopic.slug == clean))
    if topic:
        if keywords and not topic.keywords:
            topic.keywords = keywords
        if seed_channels and not topic.seed_channels:
            topic.seed_channels = seed_channels
        topic.updated_at = datetime.utcnow()
        return topic
    topic = ResearchTopic(slug=clean, title=clean, keywords=keywords, seed_channels=seed_channels or [], enabled=True)
    session.add(topic)
    await session.flush()
    return topic


async def link_topic_channel(session: AsyncSession, topic: ResearchTopic, channel: Channel, score: float, reasons: list[str]) -> None:
    existing = await session.scalar(select(TopicChannel).where(TopicChannel.topic_id == topic.id, TopicChannel.channel_id == channel.id))
    if existing:
        existing.score = max(existing.score or 0.0, score)
        existing.reasons = sorted(set((existing.reasons or []) + reasons))
        existing.last_seen_at = datetime.utcnow()
        return
    session.add(TopicChannel(topic_id=topic.id, channel_id=channel.id, score=score, reasons=reasons))


async def link_topic_message(session: AsyncSession, topic: ResearchTopic, message: Message, relevance_score: float) -> None:
    existing = await session.scalar(select(TopicMessage).where(TopicMessage.topic_id == topic.id, TopicMessage.message_id == message.id))
    if existing:
        existing.relevance_score = max(existing.relevance_score or 0.0, relevance_score)
        existing.pain_score = max(existing.pain_score or 0.0, message.pain_score or 0.0)
        existing.pain_type = message.pain_type or existing.pain_type
        existing.intent = message.intent or existing.intent
        return
    session.add(
        TopicMessage(
            topic_id=topic.id,
            message_id=message.id,
            relevance_score=relevance_score,
            pain_score=message.pain_score or 0.0,
            pain_type=message.pain_type,
            intent=message.intent,
        )
    )


async def add_topic_seed_channels(session: AsyncSession, topic: ResearchTopic, usernames: list[str]) -> list[str]:
    """Grow the topic pool: the seed list is what the regular crawl task reads."""
    known = {name.lower() for name in (topic.seed_channels or [])}
    added: list[str] = []
    for username in usernames:
        clean = username.strip("@")
        if not clean or clean.lower() in known:
            continue
        known.add(clean.lower())
        added.append(clean)
    if added:
        topic.seed_channels = [*(topic.seed_channels or []), *added]
        topic.updated_at = datetime.utcnow()
    return added


async def list_research_topics(session: AsyncSession) -> list[ResearchTopicStatus]:
    rows = (
        await session.execute(
            sql_text(
                """
                WITH channel_stats AS (
                  SELECT topic_id, count(DISTINCT channel_id) AS channels
                  FROM topic_channels
                  GROUP BY topic_id
                ),
                message_stats AS (
                  SELECT
                    topic_id,
                    count(DISTINCT message_id) AS messages,
                    count(DISTINCT message_id) FILTER (WHERE pain_score >= 0.28) AS pain_items,
                    coalesce(avg(pain_score), 0) AS avg_pain_score
                  FROM topic_messages
                  GROUP BY topic_id
                )
                SELECT
                  rt.slug,
                  rt.title,
                  rt.keywords,
                  rt.negative_keywords,
                  rt.seed_channels,
                  rt.enabled,
                  rt.updated_at,
                  coalesce(cs.channels, 0) AS channels,
                  coalesce(ms.messages, 0) AS messages,
                  coalesce(ms.pain_items, 0) AS pain_items,
                  coalesce(ms.avg_pain_score, 0) AS avg_pain_score
                FROM research_topics rt
                LEFT JOIN channel_stats cs ON cs.topic_id = rt.id
                LEFT JOIN message_stats ms ON ms.topic_id = rt.id
                ORDER BY rt.enabled DESC, pain_items DESC, messages DESC, rt.slug
                """
            )
        )
    ).mappings().all()
    return [
        ResearchTopicStatus(
            slug=row["slug"],
            title=row["title"],
            keywords=row["keywords"] or [],
            negative_keywords=row["negative_keywords"] or [],
            seed_channels=row["seed_channels"] or [],
            enabled=row["enabled"],
            channels=int(row["channels"] or 0),
            messages=int(row["messages"] or 0),
            pain_items=int(row["pain_items"] or 0),
            avg_pain_score=float(row["avg_pain_score"] or 0.0),
            updated_at=row["updated_at"],
        )
        for row in rows
    ]


async def search_messages_db(session: AsyncSession, query: str, filters: SearchFilters, limit: int) -> list[SearchHit]:
    tokens = [token.replace("'", "") for token in word_tokens(query.lower(), min_length=2)]
    or_query = " | ".join(tokens[:12])
    if not or_query:
        or_query = query
    clauses = [
        "to_tsvector('simple', coalesce(m.text, '')) @@ to_tsquery('simple', :or_query)",
        "coalesce(m.deleted_or_missing, false) = false",
    ]
    params: dict[str, object] = {"query": query, "or_query": or_query, "limit": limit}
    if filters.channel_usernames:
        clauses.append("c.username = ANY(:channels)")
        params["channels"] = [x.strip("@") for x in filters.channel_usernames]
    if filters.date_from:
        clauses.append("m.posted_at >= :date_from")
        params["date_from"] = filters.date_from
    if filters.date_to:
        clauses.append("m.posted_at <= :date_to")
        params["date_to"] = filters.date_to
    where = " AND ".join(clauses)
    stmt = sql_text(
        f"""
        SELECT
            concat(c.username, ':', m.tg_msg_id) AS message_id,
            c.username AS channel_username,
            c.title AS channel_title,
            m.url AS url,
            m.posted_at AS posted_at,
            m.text AS text,
            m.vacancy_key AS vacancy_key,
            coalesce(vc.message_count, 1) AS cluster_size,
            m.detector_score AS detector_score,
            m.detector_reasons AS detector_reasons,
            m.pain_score AS pain_score,
            m.pain_type AS pain_type,
            m.intent AS intent,
            m.pain_reasons AS pain_reasons,
            m.discussion_key AS discussion_key,
            c.quality_score AS quality_score,
            c.quality_reasons AS quality_reasons,
            ts_rank_cd(to_tsvector('simple', coalesce(m.text, '')), to_tsquery('simple', :or_query)) AS fts_score,
            (
              ts_rank_cd(to_tsvector('simple', coalesce(m.text, '')), to_tsquery('simple', :or_query))
              + 0.35 * coalesce(m.detector_score, 0)
              + 0.3 * coalesce(m.pain_score, 0)
              + 0.15 * coalesce(c.quality_score, 0)
              + 0.05 * ln(1 + coalesce(vc.message_count, 1))
            ) AS score
        FROM messages m
        JOIN channels c ON c.id = m.channel_id
        LEFT JOIN vacancy_clusters vc ON vc.id = m.cluster_id
        WHERE {where}
        ORDER BY score DESC, m.posted_at DESC NULLS LAST
        LIMIT :limit
        """
    )
    rows = (await session.execute(stmt, params)).mappings().all()
    hits: list[SearchHit] = []
    for row in rows:
        signal = hiring_signal(row["text"] or "")
        reasons = ["postgres_fts"]
        reasons.extend(row["detector_reasons"] or [])
        reasons.extend(row["pain_reasons"] or [])
        reasons.extend(f"channel_{reason}" for reason in (row["quality_reasons"] or [])[:4])
        if row["cluster_size"] and int(row["cluster_size"]) > 1:
            reasons.append(f"duplicate_cluster:{int(row['cluster_size'])}")
        hits.append(
            SearchHit(
            message_id=row["message_id"],
            channel_username=row["channel_username"],
            channel_title=row["channel_title"],
            url=row["url"],
            posted_at=row["posted_at"],
            text=row["text"],
            score=float(row["score"] or 0.0),
                highlights=signal.highlights,
                score_reasons=sorted(set(reasons)),
                why_matched=sorted(set(reasons + signal.reasons)),
                cluster_key=row["vacancy_key"],
                cluster_size=int(row["cluster_size"] or 1),
                pain_score=float(row["pain_score"] or 0.0),
                pain_type=row["pain_type"],
                intent=row["intent"],
                discussion_key=row["discussion_key"],
            )
        )
    return hits
