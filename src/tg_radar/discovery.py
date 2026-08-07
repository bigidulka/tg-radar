from collections import deque
import asyncio
import time

from tg_radar.crawler import TelegramPublicCrawler
from tg_radar.rule_loader import load_rules
from tg_radar.schemas import ChannelCandidate
from tg_radar.search_providers import ExternalSearchProvider
from tg_radar.text import clean_username, extract_mentions, extract_tme_usernames


DISCOVERY_RULES = load_rules("discovery_rules.json")


def expand_keyword_queries(keyword: str) -> list[str]:
    base = " ".join(keyword.split())
    if not base:
        return []
    lower = base.lower()
    roots = [base]
    for token in DISCOVERY_RULES["strip_tokens"]:
        if token in f" {lower} ":
            roots.insert(0, lower.replace(token.strip(), "").strip())
            break

    queries: list[str] = []
    seen: set[str] = set()
    for root in roots:
        root = root.strip()
        if not root:
            continue
        for suffix in DISCOVERY_RULES["suffixes"]:
            query = root if not suffix or suffix in root.lower() else f"{root} {suffix}"
            key = query.lower()
            if key not in seen:
                seen.add(key)
                queries.append(query)
    return queries[:12]


class Discoverer:
    def __init__(
        self,
        crawler: TelegramPublicCrawler,
        search_provider: ExternalSearchProvider,
        max_external_results_per_keyword: int,
        max_keyword_expansions: int = 6,
        query_cache_ttl_seconds: int = 3600,
    ) -> None:
        self.crawler = crawler
        self.search_provider = search_provider
        self.max_external_results_per_keyword = max_external_results_per_keyword
        self.max_keyword_expansions = max_keyword_expansions
        self.query_cache_ttl_seconds = query_cache_ttl_seconds
        self._query_cache: dict[str, tuple[float, list[ChannelCandidate]]] = {}
        self._inflight: dict[str, asyncio.Task[list[ChannelCandidate]]] = {}

    async def _search_cached(self, query: str, limit: int) -> list[ChannelCandidate]:
        key = query.lower()
        now = time.monotonic()
        cached = self._query_cache.get(key)
        if cached and (self.query_cache_ttl_seconds <= 0 or now - cached[0] <= self.query_cache_ttl_seconds):
            return cached[1][:limit]
        task = self._inflight.get(key)
        if task is None:
            task = asyncio.create_task(self.search_provider.search(query, limit))
            self._inflight[key] = task
        try:
            found = await task
        finally:
            self._inflight.pop(key, None)
        self._query_cache[key] = (now, found)
        return found

    async def from_keywords(self, keywords: list[str], limit: int) -> list[ChannelCandidate]:
        out: list[ChannelCandidate] = []
        seen: set[str] = set()
        queries: list[str] = []
        query_seen: set[str] = set()
        for keyword in keywords:
            for query in expand_keyword_queries(keyword)[: self.max_keyword_expansions]:
                key = query.lower()
                if key not in query_seen:
                    query_seen.add(key)
                    queries.append(query)

        async def run_query(query: str) -> list[ChannelCandidate]:
            try:
                return await self._search_cached(query, min(self.max_external_results_per_keyword, limit))
            except Exception as exc:
                return [
                    ChannelCandidate(
                        username="",
                        source="external_search_error",
                        reason=f"{query}: {type(exc).__name__}: {exc}",
                        score=0.0,
                    )
                ]

        def add_found(found: list[ChannelCandidate]) -> bool:
            for candidate in found:
                key = candidate.username.lower() if candidate.username else candidate.reason
                if key in seen:
                    continue
                seen.add(key)
                out.append(candidate)
                if len([item for item in out if item.username]) >= limit:
                    return True
            return False

        if queries and add_found(await run_query(queries[0])):
            return out

        semaphore = asyncio.Semaphore(max(1, min(4, self.max_keyword_expansions)))

        async def run_limited(query: str) -> list[ChannelCandidate]:
            async with semaphore:
                return await run_query(query)

        tasks = [asyncio.create_task(run_limited(query)) for query in queries[1:]]
        try:
            for task in asyncio.as_completed(tasks):
                if add_found(await task):
                    return out
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
        return out[:limit]

    async def expand_from_seeds(self, seeds: list[str], depth: int, limit: int, query: str | None = None) -> list[ChannelCandidate]:
        queue = deque((seed, 0, "seed", "seed/link graph", 1.0) for seed in seeds)
        seen: set[str] = set()
        candidates: list[ChannelCandidate] = []

        while queue and len(candidates) < limit:
            raw, current_depth, source, reason, score = queue.popleft()
            username = clean_username(raw)
            if not username or username.lower() in seen:
                continue
            seen.add(username.lower())
            candidates.append(ChannelCandidate(username=username, source=source, reason=reason, score=score, depth=current_depth))
            if current_depth >= depth:
                continue
            try:
                pages = await self.crawler.crawl_channel(username, pages=1)
            except Exception:
                pages = []
            linked: set[str] = set()
            for page in pages:
                for message in page.messages:
                    linked |= set(extract_tme_usernames(" ".join([message.text, *message.links])))
            for target in sorted(linked):
                if target.lower() not in seen:
                    queue.append((target, current_depth + 1, f"linked:{username}", "seed/link graph", 1.0))
            remaining = max(limit - len(candidates), 0)
            if remaining:
                try:
                    extra = await self.search_provider.expand(username, remaining, query)
                except Exception:
                    extra = []
                for candidate in extra:
                    if candidate.username.lower() not in seen:
                        queue.append((candidate.username, current_depth + 1, candidate.source, candidate.reason, candidate.score))
        return candidates[:limit]

    async def discover(self, keywords: list[str], seed_channels: list[str], depth: int, limit: int) -> list[ChannelCandidate]:
        query = " ".join(keywords[:3]) or None
        out = await self.expand_from_seeds(seed_channels, depth, limit, query)
        seen = {candidate.username.lower() for candidate in out if candidate.username}
        remaining = max(limit - len(out), 0)
        if remaining:
            for candidate in await self.from_keywords(keywords, remaining):
                if candidate.username and candidate.username.lower() in seen:
                    continue
                if candidate.username:
                    seen.add(candidate.username.lower())
                out.append(candidate)
        return out[:limit]
