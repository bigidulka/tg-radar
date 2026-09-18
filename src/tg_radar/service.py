from datetime import datetime, timedelta, timezone
import asyncio

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from tg_radar.config import get_settings
from tg_radar.crawler import TelegramPublicCrawler
from tg_radar.db import (
    Channel,
    DiscoveredChannel,
    Edge,
    Message,
    ResearchTopic,
    TopicChannel,
    add_channel_edge,
    add_edges_for_message,
    add_topic_seed_channels,
    channel_crawl_policy_map,
    ensure_research_topic,
    link_topic_channel,
    link_topic_message,
    mark_refresh_missing_messages,
    mark_discovered_state,
    maybe_update_operator_quality,
    update_channel_quality,
    upsert_channel,
    upsert_discovered_channel,
    upsert_message,
)
from tg_radar.discovery import Discoverer
from tg_radar.schemas import (
    CandidateState,
    ChannelCandidate,
    ChannelProfile,
    CrawlResult,
    EdgeType,
    GraphEdge,
    IngestResponse,
    ParsedMessage,
)
from tg_radar.text import extract_tme_usernames
from tg_radar.topical import TopicGate, topic_admission, topic_relevance
from tg_radar.vespa import VespaClient


class IndexingService:
    def __init__(self, crawler: TelegramPublicCrawler, vespa: VespaClient) -> None:
        self.crawler = crawler
        self.vespa = vespa

    async def crawl_and_index(
        self,
        session: AsyncSession,
        channel_username: str,
        pages: int = 1,
        crawl_mode: str = "backfill",
    ) -> CrawlResult:
        parsed_messages: list[ParsedMessage] = []
        all_messages: list[ParsedMessage] = []
        seen_tg_msg_ids: set[int] = set()
        deleted_or_missing_marked = 0
        pages_data = await self.crawler.crawl_channel(channel_username, pages=pages)
        if not pages_data:
            return CrawlResult()
        channel: Channel | None = None
        for page in pages_data:
            channel = await upsert_channel(session, page.channel_username, page.channel_title, "crawl")
            all_messages.extend(page.messages)
            for parsed in page.messages:
                seen_tg_msg_ids.add(parsed.tg_msg_id)
                message, created, changed = await upsert_message(session, channel, parsed)
                if not (created or changed):
                    continue
                await add_edges_for_message(session, channel, message, parsed.mentions)
                try:
                    await self.vespa.feed_message(parsed)
                    message.indexed_at = datetime.utcnow()
                except Exception:
                    # PostgreSQL remains searchable even if Vespa is down.
                    message.indexed_at = None
                parsed_messages.append(parsed)
        if channel:
            channel.last_crawled_at = datetime.now(timezone.utc)
            channel.last_successful_crawl_at = channel.last_crawled_at
            channel.last_error = None
            channel.last_error_at = None
            await update_channel_quality(session, channel, all_messages, channel.source)
            await maybe_update_operator_quality(session, channel, all_messages, get_settings())
            if crawl_mode == "refresh":
                deleted_or_missing_marked = await mark_refresh_missing_messages(
                    session,
                    channel,
                    seen_tg_msg_ids,
                    latest_limit=max(pages, 1) * 20,
                )
        return CrawlResult(saved=parsed_messages, seen=all_messages, deleted_or_missing_marked=deleted_or_missing_marked)


class IngestService:
    def __init__(
        self,
        discoverer: Discoverer,
        indexing: IndexingService,
        sessionmaker: async_sessionmaker[AsyncSession],
        concurrency: int,
        crawl_cooldown_seconds: int,
        topic_gate: TopicGate | None = None,
        max_new_channels_per_run: int = 5,
    ) -> None:
        self.discoverer = discoverer
        self.indexing = indexing
        self.sessionmaker = sessionmaker
        self.concurrency = concurrency
        self.crawl_cooldown_seconds = crawl_cooldown_seconds
        self.topic_gate = topic_gate or TopicGate()
        self.max_new_channels_per_run = max_new_channels_per_run

    async def ingest(
        self,
        keywords: list[str],
        seed_channels: list[str],
        depth: int,
        limit: int,
        pages_per_channel: int,
        crawl: bool,
        task_name: str | None = None,
        max_live_crawl: int | None = None,
        crawl_mode: str = "backfill",
        freshness_days: int | None = None,
        since: datetime | None = None,
        topic_slug: str | None = None,
    ) -> IngestResponse:
        topic_name = topic_slug or task_name
        if max_live_crawl == 0 and task_name:
            async with self.sessionmaker() as session:
                rows = (
                    await session.execute(
                        select(DiscoveredChannel)
                        .where(DiscoveredChannel.task_name == task_name)
                        .order_by(DiscoveredChannel.score.desc(), DiscoveredChannel.last_seen_at.desc())
                        .limit(limit)
                    )
                ).scalars().all()
            discovered = [
                ChannelCandidate(
                    username=row.username,
                    source=row.source,
                    reason=row.reason,
                    score=row.score,
                    depth=row.depth,
                    task_name=row.task_name,
                    state=CandidateState(row.state),
                    quality_score=row.quality_score or 0.0,
                    quality_reasons=row.quality_reasons or [],
                )
                for row in rows
            ]
        else:
            # At depth 0 the seed list is the pool, so the limit may never truncate it:
            # a growing pool would silently stop being crawled.
            if depth <= 0:
                discovery_limit = max(limit, len(seed_channels))
            else:
                discovery_limit = limit if seed_channels else max(10, min(limit, limit // 2))
            discovered = await self.discoverer.discover(keywords, seed_channels, depth, discovery_limit)
        diagnostics = [f"{candidate.source}: {candidate.reason}" for candidate in discovered if not candidate.username]
        candidates = [candidate for candidate in discovered if candidate.username]
        async with self.sessionmaker() as session:
            async with session.begin():
                for candidate in candidates:
                    candidate.task_name = task_name or candidate.task_name
                    if not (crawl and candidate.source.startswith("linked:")):
                        await upsert_discovered_channel(session, candidate, task_name)
                    for prefix in ("linked:", "telethon_recursive:"):
                        if candidate.source.startswith(prefix):
                            await add_channel_edge(session, candidate.source.removeprefix(prefix), candidate.username, EdgeType.link)

        if not crawl:
            return IngestResponse(candidates_found=len(candidates), channels_crawled=0, messages_saved=0, edges_saved=0, errors=diagnostics[:50], candidates=candidates)

        semaphore = asyncio.Semaphore(self.concurrency)
        errors: list[str] = []
        skipped: list[str] = []
        linked: dict[str, ChannelCandidate] = {}
        channels_crawled = 0
        messages_saved = 0
        cached_messages_used = 0
        edges_saved = 0
        deleted_or_missing_marked = 0
        newest_message_at: datetime | None = None
        oldest_included_message_at: datetime | None = None
        source_usernames: set[str] = set()
        valid_candidates: set[str] = set()
        junk_candidates: set[str] = set()
        admitted_channels: list[str] = []
        rejected_channels: list[str] = []
        expansion_budget = self.max_new_channels_per_run
        seen_candidates = {candidate.username.lower() for candidate in candidates}

        async def admit_to_topic(
            session: AsyncSession,
            topic: ResearchTopic,
            channel: Channel,
            candidate: ChannelCandidate,
            seen: int,
            matches: list[tuple[Message, float]],
        ) -> bool:
            """False keeps the candidate out of the pool; the reason stays on its row."""
            nonlocal expansion_budget
            if not await _needs_admission(session, topic, channel, candidate):
                return True
            admission = topic_admission(
                self.topic_gate,
                seen,
                [message.pain_score or 0.0 for message, _ in matches],
                channel.quality_score or 0.0,
            )
            deferred = admission.accepted and expansion_budget <= 0
            if not admission.accepted or deferred:
                gate_reason = "topic_gate_deferred:expansion budget" if deferred else admission.reason
                candidate.state = CandidateState.validated_channel if deferred else CandidateState.rejected_low_quality
                candidate.quality_reasons = sorted({*candidate.quality_reasons, gate_reason})
                await upsert_discovered_channel(session, candidate, task_name)
                rejected_channels.append(f"@{candidate.username}: {gate_reason}")
                return False
            expansion_budget -= 1
            admitted_channels.append(candidate.username)
            candidate.quality_reasons = sorted({*candidate.quality_reasons, admission.reason})
            return True

        async def use_cached_messages(candidate: ChannelCandidate, reason: str) -> bool:
            nonlocal cached_messages_used, newest_message_at, oldest_included_message_at
            if not topic_name:
                return False
            async with self.sessionmaker() as session:
                async with session.begin():
                    topic = await ensure_research_topic(session, topic_name, keywords, seed_channels)
                    channel = await session.scalar(select(Channel).where(func.lower(Channel.username) == candidate.username.strip("@").lower()))
                    if not channel:
                        return False
                    query = (
                        select(Message)
                        .where(Message.channel_id == channel.id)
                        .where(Message.deleted_or_missing.is_(False))
                    )
                    if since:
                        query = query.where(Message.posted_at >= since)
                    elif freshness_days:
                        query = query.where(Message.posted_at >= datetime.now(timezone.utc) - timedelta(days=freshness_days))
                    rows = (
                        await session.execute(
                            query
                            .order_by(Message.posted_at.desc().nulls_last(), Message.id.desc())
                            .limit(max(pages_per_channel, 1) * 20)
                        )
                    ).scalars().all()
                    if not rows:
                        return False
                    matches: list[tuple[Message, float]] = []
                    for message in rows:
                        if message.posted_at:
                            nonlocal_newest = newest_message_at
                            nonlocal_oldest = oldest_included_message_at
                            newest_message_at = max(nonlocal_newest, message.posted_at) if nonlocal_newest else message.posted_at
                            oldest_included_message_at = min(nonlocal_oldest, message.posted_at) if nonlocal_oldest else message.posted_at
                        relevance = topic_relevance(" ".join([message.text or "", *(message.links or []), *(message.mentions or [])]), keywords)
                        if relevance > 0 or (message.pain_score or 0.0) >= 0.28:
                            matches.append((message, relevance))
                    if not matches:
                        return False
                    # Skipping the crawl must not skip the gate, or every cooldown pass
                    # would let a rejected candidate into the pool through the back door.
                    if not await admit_to_topic(session, topic, channel, candidate, len(rows), matches):
                        return True
                    await link_topic_channel(session, topic, channel, candidate.score, [candidate.source, candidate.reason, reason])
                    for message, relevance in matches:
                        await link_topic_message(session, topic, message, relevance)
                    candidate.state = CandidateState.indexed_channel
                    await upsert_discovered_channel(session, candidate, task_name)
                    valid_candidates.add(candidate.username.lower())
                    source_usernames.add(candidate.username.lower())
                    cached_messages_used += len(matches)
                    return True

        async def filter_cooldown(items: list[ChannelCandidate]) -> list[ChannelCandidate]:
            if self.crawl_cooldown_seconds <= 0:
                return items
            async with self.sessionmaker() as session:
                crawl_policies = await channel_crawl_policy_map(session, [candidate.username for candidate in items])
            out: list[ChannelCandidate] = []
            for candidate in items:
                previous, interval = crawl_policies.get(candidate.username.lower(), (None, self.crawl_cooldown_seconds))
                cooldown_seconds = interval or self.crawl_cooldown_seconds
                threshold = datetime.now(timezone.utc) - timedelta(seconds=cooldown_seconds)
                previous_utc = previous.replace(tzinfo=timezone.utc) if previous and not previous.tzinfo else previous
                if previous_utc and previous_utc > threshold:
                    if not await use_cached_messages(candidate, "cached_after_crawl_cooldown"):
                        skipped.append(f"@{candidate.username}: skipped crawl cooldown")
                    continue
                out.append(candidate)
            return out

        async def crawl_one(candidate: ChannelCandidate) -> None:
            nonlocal channels_crawled, messages_saved, edges_saved, deleted_or_missing_marked, newest_message_at, oldest_included_message_at, expansion_budget
            if not candidate.username:
                return
            async with semaphore:
                try:
                    async with self.sessionmaker() as session:
                        async with session.begin():
                            topic = await ensure_research_topic(session, topic_name, keywords, seed_channels) if topic_name else None
                            result = await self.indexing.crawl_and_index(
                                session,
                                candidate.username,
                                pages=pages_per_channel,
                                crawl_mode=crawl_mode,
                            )
                            deleted_or_missing_marked += result.deleted_or_missing_marked
                            if not result.channel_reachable:
                                await mark_discovered_state(session, candidate.username, CandidateState.empty_public)
                                junk_candidates.add(candidate.username.lower())
                                skipped.append(f"@{candidate.username}: skipped empty or non-public channel/chat")
                                return
                            channel = await upsert_channel(session, candidate.username, None, candidate.source)
                            # The crawl happened and the messages are stored whatever the topic
                            # gate decides later, so the counters are settled here.
                            channels_crawled += 1
                            messages_saved += len(result.saved)
                            edges_saved += sum(len(message.mentions) for message in result.saved)
                            source_usernames.add(candidate.username.lower())
                            for prefix in ("linked:", "telethon_recursive:"):
                                if candidate.source.startswith(prefix):
                                    await add_channel_edge(session, candidate.source.removeprefix(prefix), candidate.username, EdgeType.link)
                            matches: list[tuple[Message, float]] = []
                            stored_messages = await _stored_messages(session, channel, result.seen)
                            for message in result.seen:
                                if message.posted_at:
                                    newest_message_at = max(newest_message_at, message.posted_at) if newest_message_at else message.posted_at
                                    oldest_included_message_at = min(oldest_included_message_at, message.posted_at) if oldest_included_message_at else message.posted_at
                                stored = stored_messages.get(message.tg_msg_id)
                                if topic and stored:
                                    relevance = topic_relevance(
                                        " ".join([message.text, *message.links, *message.mentions]),
                                        keywords,
                                    )
                                    if relevance > 0 or (stored.pain_score or 0.0) >= 0.28:
                                        matches.append((stored, relevance))
                                for username in extract_tme_usernames(" ".join([message.text, *message.links])):
                                    key = username.lower()
                                    if key in seen_candidates or key in linked:
                                        continue
                                    linked[key] = ChannelCandidate(
                                        username=username,
                                        source=f"linked:{candidate.username}",
                                        reason="auto graph expansion",
                                        score=max(candidate.score * 0.8, 0.3),
                                        depth=candidate.depth + 1,
                                    )
                            if topic and not await admit_to_topic(session, topic, channel, candidate, len(result.seen), matches):
                                return
                            if topic:
                                reasons = [candidate.source, candidate.reason]
                                await link_topic_channel(session, topic, channel, candidate.score, reasons)
                                for stored, relevance in matches:
                                    await link_topic_message(session, topic, stored, relevance)
                            candidate.state = CandidateState.indexed_channel
                            await upsert_discovered_channel(session, candidate, task_name)
                            valid_candidates.add(candidate.username.lower())
                except Exception as exc:
                    error_text = f"{type(exc).__name__}: {exc}"
                    errors.append(f"@{candidate.username}: {error_text}")
                    async with self.sessionmaker() as session:
                        async with session.begin():
                            channel = await session.scalar(select(Channel).where(func.lower(Channel.username) == candidate.username.strip("@").lower()))
                            if channel:
                                channel.last_error = error_text
                                channel.last_error_at = datetime.now(timezone.utc)

        async def crawl_all(items: list[ChannelCandidate]) -> None:
            """One unreachable channel must not abort the pass over the rest."""
            outcomes = await asyncio.gather(*(crawl_one(item) for item in items), return_exceptions=True)
            for item, outcome in zip(items, outcomes, strict=False):
                if isinstance(outcome, Exception):
                    errors.append(f"@{item.username}: {type(outcome).__name__}: {outcome}")

        crawl_candidates = await filter_cooldown(candidates)
        if max_live_crawl is not None and len(crawl_candidates) > max_live_crawl:
            skipped.extend(f"@{candidate.username}: skipped live crawl budget" for candidate in crawl_candidates[max_live_crawl:])
            crawl_candidates = crawl_candidates[:max_live_crawl]
        await crawl_all(crawl_candidates)
        if depth > 0 and len(candidates) < limit and linked:
            graph_candidates = list(linked.values())[: max(limit - len(candidates), 0)]
            async with self.sessionmaker() as session:
                async with session.begin():
                    for candidate in graph_candidates:
                        candidate.task_name = task_name or candidate.task_name
                        if not candidate.source.startswith("linked:"):
                            await upsert_discovered_channel(session, candidate, task_name)
                            for prefix in ("telethon_recursive:",):
                                if candidate.source.startswith(prefix):
                                    await add_channel_edge(session, candidate.source.removeprefix(prefix), candidate.username, EdgeType.link)
            candidates.extend(graph_candidates)
            crawl_graph_candidates = await filter_cooldown(graph_candidates)
            if max_live_crawl is not None and len(crawl_graph_candidates) > max_live_crawl:
                skipped.extend(f"@{candidate.username}: skipped live crawl budget" for candidate in crawl_graph_candidates[max_live_crawl:])
                crawl_graph_candidates = crawl_graph_candidates[:max_live_crawl]
            await crawl_all(crawl_graph_candidates)
        if topic_name and admitted_channels:
            async with self.sessionmaker() as session:
                async with session.begin():
                    topic = await ensure_research_topic(session, topic_name, keywords, seed_channels)
                    await add_topic_seed_channels(session, topic, admitted_channels)
        return IngestResponse(
            candidates_found=len(candidates),
            valid_candidates=len(valid_candidates),
            junk_candidates=len(junk_candidates),
            channels_crawled=channels_crawled,
            messages_saved=messages_saved,
            cached_messages_used=cached_messages_used,
            edges_saved=edges_saved,
            errors=(diagnostics + skipped + errors)[:50],
            candidates=candidates,
            freshness_days=freshness_days,
            newest_message_at=newest_message_at,
            oldest_included_message_at=oldest_included_message_at,
            source_count=len(source_usernames),
            message_count=messages_saved + cached_messages_used,
            deleted_or_missing_marked=deleted_or_missing_marked,
            admitted_channels=admitted_channels,
            rejected_channels=rejected_channels[:50],
        )


async def _stored_messages(session: AsyncSession, channel: Channel, parsed: list[ParsedMessage]) -> dict[int, Message]:
    tg_msg_ids = [message.tg_msg_id for message in parsed]
    if not tg_msg_ids:
        return {}
    rows = (
        await session.execute(
            select(Message).where(Message.channel_id == channel.id, Message.tg_msg_id.in_(tg_msg_ids))
        )
    ).scalars().all()
    return {row.tg_msg_id: row for row in rows}


async def _needs_admission(session: AsyncSession, topic: ResearchTopic, channel: Channel, candidate: ChannelCandidate) -> bool:
    """Curated seeds and channels already in the pool bypass the expansion gate."""
    if candidate.source == "seed":
        return False
    existing = await session.scalar(
        select(TopicChannel).where(TopicChannel.topic_id == topic.id, TopicChannel.channel_id == channel.id)
    )
    return existing is None


async def channel_profile(session: AsyncSession, username: str) -> ChannelProfile | None:
    username = username.strip("@")
    channel = await session.scalar(select(Channel).where(func.lower(Channel.username) == username.lower()))
    if not channel:
        return None
    message_count = await session.scalar(select(func.count(Message.id)).where(Message.channel_id == channel.id))
    neighbors_rows = await session.execute(select(Edge.dst_username).where(Edge.src_channel_id == channel.id).limit(50))
    return ChannelProfile(
        username=channel.username,
        title=channel.title,
        status=channel.status,
        first_seen_at=channel.first_seen_at,
        last_seen_at=channel.last_seen_at,
        message_count=int(message_count or 0),
        neighbors=sorted({row[0] for row in neighbors_rows}),
        operator_niche_category=channel.operator_niche_category,
        operator_income_authenticity=channel.operator_income_authenticity,
        operator_niche_tags=channel.operator_niche_tags or [],
        operator_niche_confidence=channel.operator_niche_confidence or 0.0,
        operator_niche_evidence=(channel.operator_niche_evidence or {}).get("items", []),
        operator_niche_notes=channel.operator_niche_notes or [],
        operator_niche_updated_at=channel.operator_niche_updated_at,
    )


async def graph_edges_from_db(session: AsyncSession, username: str, limit: int) -> list[GraphEdge]:
    channel = await session.scalar(select(Channel).where(func.lower(Channel.username) == username.strip("@").lower()))
    if not channel:
        return []
    rows = await session.execute(
        select(Edge.dst_username, Edge.edge_type, Message.url)
        .outerjoin(Message, Message.id == Edge.msg_id)
        .where(Edge.src_channel_id == channel.id)
        .limit(limit)
    )
    return [
        GraphEdge(
            source=channel.username,
            target=row[0],
            edge_type=EdgeType(row[1]),
            message_url=row[2],
        )
        for row in rows
    ]
