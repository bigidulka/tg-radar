from urllib.parse import parse_qs, quote_plus, unquote, urlparse
import asyncio
import html

from curl_cffi.requests import AsyncSession
from selectolax.parser import HTMLParser

from tg_radar.config import Settings
from tg_radar.rule_loader import load_rules
from tg_radar.schemas import ChannelCandidate
from tg_radar.text import clean_username, extract_mentions, extract_tme_usernames

SEARCH_PROVIDER_RULES = load_rules("search_provider_rules.json")


def usernames_from_url(url: str) -> set[str]:
    parsed = urlparse(url)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        uddg = parse_qs(parsed.query).get("uddg", [None])[0]
        if uddg:
            url = unquote(uddg)
            parsed = urlparse(url)
    if parsed.netloc not in {"t.me", "telegram.me", "www.t.me"}:
        return set()
    parts = [part for part in parsed.path.split("/") if part]
    if not parts:
        return set()
    if parts[0] == "s" and len(parts) > 1:
        username = clean_username(parts[1])
    else:
        username = clean_username(parts[0])
    return {username} if username else set()


class ExternalSearchProvider:
    async def search(self, keyword: str, limit: int) -> list[ChannelCandidate]:
        raise NotImplementedError

    async def expand(self, seed: str, limit: int, query: str | None = None) -> list[ChannelCandidate]:
        return []


class DuckDuckGoSearchProvider(ExternalSearchProvider):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def search(self, keyword: str, limit: int) -> list[ChannelCandidate]:
        query = f'site:t.me/s "{keyword}"'
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        headers = {"user-agent": self.settings.user_agent}
        async with AsyncSession(timeout=self.settings.request_timeout_seconds, impersonate="chrome120") as session:
            response = await session.get(url, headers=headers)
        response.raise_for_status()
        tree = HTMLParser(response.text)
        candidates: list[ChannelCandidate] = []
        seen: set[str] = set()
        for node in tree.css("a.result__a, a[href]"):
            href = node.attributes.get("href", "")
            text = node.text(separator=" ")
            found = usernames_from_url(href) | set(extract_mentions(text))
            for username in sorted(found):
                if username.lower() in seen:
                    continue
                seen.add(username.lower())
                candidates.append(
                    ChannelCandidate(
                        username=username,
                        source="duckduckgo",
                        reason=query,
                        score=0.8,
                    )
                )
                if len(candidates) >= limit:
                    return candidates
        return candidates


class HtmlSearchProvider(ExternalSearchProvider):
    def __init__(self, settings: Settings, name: str, url_template: str) -> None:
        self.settings = settings
        self.name = name
        self.url_template = url_template

    async def search(self, keyword: str, limit: int) -> list[ChannelCandidate]:
        query = f"site:t.me/s {keyword}"
        url = self.url_template.format(query=quote_plus(query))
        headers = {"user-agent": self.settings.user_agent}
        async with AsyncSession(timeout=self.settings.request_timeout_seconds, impersonate="chrome120") as session:
            response = await session.get(url, headers=headers)
        response.raise_for_status()
        raw = html.unescape(response.text)
        candidates: list[ChannelCandidate] = []
        seen: set[str] = set()
        haystack = raw + " " + unquote(raw)
        for username in extract_tme_usernames(haystack):
            if username.lower() in seen:
                continue
            seen.add(username.lower())
            candidates.append(ChannelCandidate(username=username, source=self.name, reason=query, score=0.75))
            if len(candidates) >= limit:
                return candidates
        tree = HTMLParser(raw)
        for node in tree.css("a[href]"):
            found = usernames_from_url(node.attributes.get("href", "")) | set(extract_mentions(node.text(separator=" ")))
            for username in sorted(found):
                if username.lower() in seen:
                    continue
                seen.add(username.lower())
                candidates.append(ChannelCandidate(username=username, source=self.name, reason=query, score=0.7))
                if len(candidates) >= limit:
                    return candidates
        return candidates


class LyzemSearchProvider(ExternalSearchProvider):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def search(self, keyword: str, limit: int) -> list[ChannelCandidate]:
        url = f"https://lyzem.com/search?q={quote_plus(keyword)}"
        headers = {"user-agent": self.settings.user_agent}
        async with AsyncSession(timeout=self.settings.request_timeout_seconds, impersonate="chrome120") as session:
            response = await session.get(url, headers=headers)
        response.raise_for_status()
        tree = HTMLParser(response.text)
        candidates: list[ChannelCandidate] = []
        seen: set[str] = set()
        for node in tree.css(".search-result a[href], a[href]"):
            href = node.attributes.get("href", "")
            found = usernames_from_url(href) | set(extract_mentions(node.text(separator=" ")))
            for username in sorted(found):
                if username.lower() in seen:
                    continue
                seen.add(username.lower())
                candidates.append(ChannelCandidate(username=username, source="lyzem", reason=keyword, score=0.82))
                if len(candidates) >= limit:
                    return candidates
        return candidates


class TelemetrPublicProvider(ExternalSearchProvider):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def search(self, keyword: str, limit: int) -> list[ChannelCandidate]:
        headers = {"user-agent": self.settings.user_agent}
        candidates: list[ChannelCandidate] = []
        seen: set[str] = set()
        async with AsyncSession(timeout=self.settings.request_timeout_seconds, impersonate="chrome120") as session:
            for path, params, source, score in (
                ("/api/v1/channels", {"q": keyword, "limit": min(limit, 20)}, "telemetr_channels", 0.9),
                ("/api/v1/ads", {"q": keyword, "dest": "telegram", "limit": min(limit, 20)}, "telemetr_ads", 0.78),
            ):
                response = await session.get(f"https://tgadsspy.com{path}", params=params, headers=headers)
                response.raise_for_status()
                data = response.json()
                for item in data.get("data", []):
                    found = set()
                    username = item.get("username") or item.get("ctaTargetUsername")
                    if username:
                        clean = clean_username(username)
                        if clean:
                            found.add(clean)
                    found |= usernames_from_url(item.get("ctaUrl", "") or "")
                    found |= set(extract_mentions(" ".join(str(item.get(k, "")) for k in ("title", "text", "targetTitle"))))
                    for candidate_username in sorted(found):
                        if candidate_username.lower() in seen:
                            continue
                        seen.add(candidate_username.lower())
                        candidates.append(ChannelCandidate(username=candidate_username, source=source, reason=keyword, score=score))
                        if len(candidates) >= limit:
                            return candidates
        return candidates


def walk_values(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(walk_values(item))
        return out
    if isinstance(value, dict):
        out: list[str] = []
        for item in value.values():
            out.extend(walk_values(item))
        return out
    return []


class OpenWebSearchProvider(ExternalSearchProvider):
    def __init__(self, settings: Settings) -> None:
        if not settings.open_websearch_url:
            raise ValueError("open_websearch_url is required")
        self.settings = settings
        self.url = settings.open_websearch_url.rstrip("/")
        self.engines = [engine.strip() for engine in settings.open_websearch_engines.split(",") if engine.strip()]

    async def search(self, keyword: str, limit: int) -> list[ChannelCandidate]:
        candidates: list[ChannelCandidate] = []
        seen: set[str] = set()
        queries = [f"site:t.me {keyword}", f"site:t.me/s {keyword}", f'"t.me" {keyword} Telegram']
        async with AsyncSession(timeout=self.settings.request_timeout_seconds, impersonate="chrome120") as session:
            for query in queries:
                payload: dict[str, object] = {"query": query, "limit": min(limit, 20)}
                if self.engines:
                    payload["engines"] = self.engines
                response = await session.post(f"{self.url}/search", json=payload, headers={"user-agent": self.settings.user_agent})
                response.raise_for_status()
                envelope = response.json()
                data = envelope.get("data", envelope)
                values = " ".join(walk_values(data))
                for username in extract_tme_usernames(values + " " + unquote(values)):
                    if username.lower() in seen:
                        continue
                    seen.add(username.lower())
                    candidates.append(ChannelCandidate(username=username, source="open_websearch", reason=query, score=0.76))
                    if len(candidates) >= limit:
                        return candidates
                for username in extract_mentions(values):
                    if username.lower() in seen:
                        continue
                    seen.add(username.lower())
                    candidates.append(ChannelCandidate(username=username, source="open_websearch", reason=query, score=0.7))
                    if len(candidates) >= limit:
                        return candidates
                if candidates:
                    return candidates
        return candidates


class TelethonUserSearchProvider(ExternalSearchProvider):
    def __init__(self, settings: Settings) -> None:
        if not settings.telethon_enabled:
            raise ValueError("telethon_enabled is required")
        self.settings = settings

    async def search(self, keyword: str, limit: int) -> list[ChannelCandidate]:
        from tg_radar.telethon_user import TelethonUserClient

        client = TelethonUserClient(self.settings)
        seen: set[str] = set()
        candidates: list[ChannelCandidate] = []

        def add(items: list[ChannelCandidate]) -> bool:
            for candidate in items:
                if not candidate.username or candidate.username.lower() in seen:
                    continue
                seen.add(candidate.username.lower())
                candidates.append(candidate)
                if len(candidates) >= limit:
                    return True
            return False

        try:
            posts = await client.search_posts(keyword, limit)
        except Exception:
            posts = []
        if add(await client.filter_chat_candidates(client.candidates_from_messages(posts, "telethon_search_posts", keyword, limit), limit)):
            return candidates
        try:
            global_posts = await client.search_global(keyword, limit, broadcasts_only=True)
        except Exception:
            global_posts = []
        if add(await client.filter_chat_candidates(client.candidates_from_messages(global_posts, "telethon_search_global", keyword, limit), limit)):
            return candidates
        try:
            group_posts = await client.search_global(keyword, limit, groups_only=True)
        except Exception:
            group_posts = []
        if add(await client.filter_chat_candidates(client.candidates_from_messages(group_posts, "telethon_search_chats", keyword, limit), limit)):
            return candidates
        try:
            contacts = await client.search_contacts(keyword, limit)
        except Exception:
            contacts = {"chats": [], "users": []}
        if add(await client.filter_chat_candidates(client.candidates_from_entities(contacts.get("chats", []), "telethon_contacts_chats", keyword, limit, 0.9), limit)):
            return candidates
        for seed in list(candidates)[: min(5, limit)]:
            try:
                deep = await client.candidate_scan(seed.username, keyword, min(limit - len(candidates), self.settings.telethon_search_limit))
            except Exception:
                deep = []
            if add(deep):
                return candidates
        return candidates

    async def expand(self, seed: str, limit: int, query: str | None = None) -> list[ChannelCandidate]:
        from tg_radar.telethon_user import TelethonUserClient

        client = TelethonUserClient(self.settings)
        return await client.candidate_scan(seed, query, limit)


class JinaGoogleSearchProvider(ExternalSearchProvider):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def search(self, keyword: str, limit: int) -> list[ChannelCandidate]:
        candidates: list[ChannelCandidate] = []
        seen: set[str] = set()
        queries = [template.format(keyword=keyword) for template in SEARCH_PROVIDER_RULES["jina_google_expand_queries"]]
        async with AsyncSession(timeout=self.settings.request_timeout_seconds, impersonate="chrome120") as session:
            for query in queries:
                response = await session.get(
                    f"https://r.jina.ai/http://www.google.com/search?q={quote_plus(query)}",
                    headers={"user-agent": self.settings.user_agent},
                )
                response.raise_for_status()
                raw = html.unescape(response.text)
                for username in extract_tme_usernames(raw + " " + unquote(raw)):
                    if username.lower() in seen:
                        continue
                    seen.add(username.lower())
                    candidates.append(ChannelCandidate(username=username, source="jina_google", reason=query, score=0.85))
                    if len(candidates) >= limit:
                        return candidates
                if candidates:
                    return candidates
        return candidates


class MultiSearchProvider(ExternalSearchProvider):
    def __init__(self, providers: list[ExternalSearchProvider]) -> None:
        self.providers = providers

    async def search(self, keyword: str, limit: int) -> list[ChannelCandidate]:
        out: list[ChannelCandidate] = []
        seen: set[str] = set()
        for provider in self.providers:
            try:
                candidates = await provider.search(keyword, limit)
            except Exception:
                continue
            for candidate in candidates:
                if not candidate.username or candidate.username.lower() in seen:
                    continue
                seen.add(candidate.username.lower())
                out.append(candidate)
                if len(out) >= limit:
                    return out
        return out

    async def expand(self, seed: str, limit: int, query: str | None = None) -> list[ChannelCandidate]:
        out: list[ChannelCandidate] = []
        seen: set[str] = set()
        for provider in self.providers:
            try:
                candidates = await provider.expand(seed, limit, query)
            except Exception:
                continue
            for candidate in candidates:
                if not candidate.username or candidate.username.lower() in seen:
                    continue
                seen.add(candidate.username.lower())
                out.append(candidate)
                if len(out) >= limit:
                    return out
        return out


class TieredSearchProvider(ExternalSearchProvider):
    def __init__(self, tiers: list[list[ExternalSearchProvider]], concurrency: int) -> None:
        self.tiers = tiers
        self.concurrency = concurrency

    async def search(self, keyword: str, limit: int) -> list[ChannelCandidate]:
        out: list[ChannelCandidate] = []
        seen: set[str] = set()
        for tier in self.tiers:
            semaphore = asyncio.Semaphore(self.concurrency)

            async def run_provider(provider: ExternalSearchProvider) -> list[ChannelCandidate]:
                async with semaphore:
                    try:
                        return await asyncio.wait_for(provider.search(keyword, limit), timeout=30)
                    except Exception:
                        return []

            results = await asyncio.gather(*(run_provider(provider) for provider in tier))
            for candidates in results:
                for candidate in candidates:
                    if not candidate.username or candidate.username.lower() in seen:
                        continue
                    seen.add(candidate.username.lower())
                    out.append(candidate)
                    if len(out) >= limit:
                        return out
            if out:
                return out
        return out

    async def expand(self, seed: str, limit: int, query: str | None = None) -> list[ChannelCandidate]:
        out: list[ChannelCandidate] = []
        seen: set[str] = set()
        for tier in self.tiers:
            semaphore = asyncio.Semaphore(self.concurrency)

            async def run_provider(provider: ExternalSearchProvider) -> list[ChannelCandidate]:
                async with semaphore:
                    try:
                        return await asyncio.wait_for(provider.expand(seed, limit, query), timeout=30)
                    except Exception:
                        return []

            results = await asyncio.gather(*(run_provider(provider) for provider in tier))
            for candidates in results:
                for candidate in candidates:
                    if not candidate.username or candidate.username.lower() in seen:
                        continue
                    seen.add(candidate.username.lower())
                    out.append(candidate)
                    if len(out) >= limit:
                        return out
            if out:
                return out
        return out


class SearxngSearchProvider(ExternalSearchProvider):
    def __init__(self, settings: Settings) -> None:
        if not settings.searxng_endpoint:
            raise ValueError("searxng_endpoint is required")
        self.settings = settings
        self.endpoint = settings.searxng_endpoint.rstrip("/")

    async def search(self, keyword: str, limit: int) -> list[ChannelCandidate]:
        query = f'site:t.me/s "{keyword}"'
        params = {"q": query, "format": "json", "language": "ru-RU"}
        async with AsyncSession(timeout=self.settings.request_timeout_seconds, impersonate="chrome120") as session:
            response = await session.get(f"{self.endpoint}/search", params=params, headers={"user-agent": self.settings.user_agent})
        response.raise_for_status()
        data = response.json()
        candidates: list[ChannelCandidate] = []
        seen: set[str] = set()
        for result in data.get("results", []):
            found = usernames_from_url(result.get("url", "")) | set(extract_mentions(result.get("content", "")))
            for username in sorted(found):
                if username.lower() in seen:
                    continue
                seen.add(username.lower())
                candidates.append(ChannelCandidate(username=username, source="searxng", reason=query, score=0.9))
                if len(candidates) >= limit:
                    return candidates
        return candidates


class BraveSearchProvider(ExternalSearchProvider):
    def __init__(self, settings: Settings) -> None:
        if not settings.brave_api_key:
            raise ValueError("brave_api_key is required")
        self.settings = settings

    async def search(self, keyword: str, limit: int) -> list[ChannelCandidate]:
        query = f"site:t.me/s {keyword}"
        headers = {"x-subscription-token": self.settings.brave_api_key, "user-agent": self.settings.user_agent}
        params = {"q": query, "count": min(limit, 20), "search_lang": "ru"}
        async with AsyncSession(timeout=self.settings.request_timeout_seconds, impersonate="chrome120") as session:
            response = await session.get("https://api.search.brave.com/res/v1/web/search", params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        return candidates_from_search_items(data.get("web", {}).get("results", []), "brave", query, limit)


class SerpApiSearchProvider(ExternalSearchProvider):
    def __init__(self, settings: Settings) -> None:
        if not settings.serpapi_key:
            raise ValueError("serpapi_key is required")
        self.settings = settings

    async def search(self, keyword: str, limit: int) -> list[ChannelCandidate]:
        query = f"site:t.me/s {keyword}"
        params = {"engine": "google", "q": query, "api_key": self.settings.serpapi_key, "num": min(limit, 100)}
        async with AsyncSession(timeout=self.settings.request_timeout_seconds, impersonate="chrome120") as session:
            response = await session.get("https://serpapi.com/search.json", params=params, headers={"user-agent": self.settings.user_agent})
        response.raise_for_status()
        data = response.json()
        return candidates_from_search_items(data.get("organic_results", []), "serpapi", query, limit)


class TgstatSearchProvider(ExternalSearchProvider):
    def __init__(self, settings: Settings) -> None:
        if not settings.tgstat_token:
            raise ValueError("tgstat_token is required")
        self.settings = settings

    async def search(self, keyword: str, limit: int) -> list[ChannelCandidate]:
        params = {"token": self.settings.tgstat_token, "q": keyword, "limit": min(limit, 50), "extended": 1, "peerType": "all"}
        async with AsyncSession(timeout=self.settings.request_timeout_seconds, impersonate="chrome120") as session:
            response = await session.get("https://api.tgstat.ru/posts/search", params=params, headers={"user-agent": self.settings.user_agent})
        response.raise_for_status()
        data = response.json()
        channels = data.get("response", {}).get("channels", [])
        items = data.get("response", {}).get("items", [])
        return candidates_from_search_items(channels + items, "tgstat", keyword, limit)


def candidates_from_search_items(items: list[dict], source: str, reason: str, limit: int) -> list[ChannelCandidate]:
    candidates: list[ChannelCandidate] = []
    seen: set[str] = set()
    for item in items:
        found = usernames_from_url(item.get("url", "") or item.get("link", ""))
        username = clean_username(str(item.get("username", "")))
        if username:
            found.add(username)
        found |= set(extract_mentions(" ".join(str(item.get(k, "")) for k in ("title", "snippet", "description", "content"))))
        for username in sorted(found):
            if username.lower() in seen:
                continue
            seen.add(username.lower())
            candidates.append(ChannelCandidate(username=username, source=source, reason=reason, score=1.0))
            if len(candidates) >= limit:
                return candidates
    return candidates


class NullSearchProvider(ExternalSearchProvider):
    async def search(self, keyword: str, limit: int) -> list[ChannelCandidate]:
        return [
            ChannelCandidate(
                username="",
                source="external_search_disabled",
                reason=f'site:t.me/s "{keyword}"',
                score=0.0,
            )
        ][:limit]


def make_search_provider(settings: Settings) -> ExternalSearchProvider:
    if not settings.external_search_enabled:
        return NullSearchProvider()
    if settings.external_search_provider == "brave":
        return BraveSearchProvider(settings)
    if settings.external_search_provider == "serpapi":
        return SerpApiSearchProvider(settings)
    if settings.external_search_provider == "searxng":
        return SearxngSearchProvider(settings)
    if settings.external_search_provider == "tgstat":
        return TgstatSearchProvider(settings)
    if settings.external_search_provider == "telemetr_public":
        return TelemetrPublicProvider(settings)
    if settings.external_search_provider == "lyzem":
        return LyzemSearchProvider(settings)
    if settings.external_search_provider == "open_websearch":
        return OpenWebSearchProvider(settings)
    if settings.external_search_provider == "telethon":
        return TelethonUserSearchProvider(settings)
    fast: list[ExternalSearchProvider] = []
    if settings.telethon_enabled and settings.telethon_user_search_enabled:
        fast.append(TelethonUserSearchProvider(settings))
    if settings.brave_api_key:
        fast.append(BraveSearchProvider(settings))
    if settings.serpapi_key:
        fast.append(SerpApiSearchProvider(settings))
    if settings.searxng_endpoint:
        fast.append(SearxngSearchProvider(settings))
    if settings.tgstat_token:
        fast.append(TgstatSearchProvider(settings))
    if settings.telemetr_public_enabled:
        fast.append(TelemetrPublicProvider(settings))
    if settings.lyzem_enabled:
        fast.append(LyzemSearchProvider(settings))
    if settings.open_websearch_enabled and settings.open_websearch_url:
        fast.append(OpenWebSearchProvider(settings))
    fallback = [
        JinaGoogleSearchProvider(settings),
        DuckDuckGoSearchProvider(settings),
        HtmlSearchProvider(settings, "google", "https://www.google.com/search?q={query}"),
        HtmlSearchProvider(settings, "bing", "https://www.bing.com/search?q={query}"),
    ]
    return TieredSearchProvider([fast, fallback] if fast else [fallback], settings.external_search_tier_concurrency)
