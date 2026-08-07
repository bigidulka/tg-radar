from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from tg_radar.agent import DataCollectionAgent
from tg_radar.agent_factory import build_collection_agent
from tg_radar.agent_runs import AgentRunRegistry
from tg_radar.auto_search import AutoSearchEngine
from tg_radar.config import Settings, get_settings
from tg_radar.crawler import TelegramPublicCrawler
from tg_radar.db import make_engine, make_sessionmaker
from tg_radar.discovery import Discoverer
from tg_radar.embeddings import Embedder
from tg_radar.reranker import Reranker
from tg_radar.search_providers import make_search_provider
from tg_radar.service import IndexingService, IngestService
from tg_radar.telethon_user import TelethonUserClient
from tg_radar.vespa import VespaClient


class AppState:
    settings: Settings
    engine: AsyncEngine
    sessionmaker: async_sessionmaker[AsyncSession]
    crawler: TelegramPublicCrawler
    discoverer: Discoverer
    vespa: VespaClient
    reranker: Reranker
    indexing: IndexingService
    ingesting: IngestService
    auto_search: AutoSearchEngine
    agent: DataCollectionAgent
    agent_runs: AgentRunRegistry
    telethon: TelethonUserClient


def build_state(settings: Settings | None = None) -> AppState:
    resolved = settings or get_settings()
    engine = make_engine(resolved)
    state = AppState()
    state.settings = resolved
    state.engine = engine
    state.sessionmaker = make_sessionmaker(engine)
    state.crawler = TelegramPublicCrawler(resolved)
    state.discoverer = Discoverer(
        state.crawler,
        make_search_provider(resolved),
        resolved.max_external_results_per_keyword,
        resolved.max_keyword_expansions,
        resolved.discovery_query_cache_ttl_seconds,
    )
    state.vespa = VespaClient(resolved.vespa_endpoint, Embedder(resolved))
    state.reranker = Reranker(resolved)
    state.indexing = IndexingService(state.crawler, state.vespa)
    state.ingesting = IngestService(
        state.discoverer,
        state.indexing,
        state.sessionmaker,
        resolved.ingest_concurrency,
        resolved.crawl_cooldown_seconds,
    )
    state.auto_search = AutoSearchEngine(
        state.sessionmaker,
        state.ingesting,
        resolved.auto_worker_interval_seconds,
        resolved.auto_task_timeout_seconds,
    )
    state.agent = build_collection_agent(
        resolved,
        state.sessionmaker,
        state.discoverer,
        state.ingesting,
        state.auto_search,
    )
    state.agent_runs = AgentRunRegistry(
        state.sessionmaker,
        timeout_seconds=resolved.agent_run_timeout_seconds,
        poll_seconds=resolved.agent_run_worker_interval_seconds,
    )
    state.telethon = TelethonUserClient(resolved)
    return state
