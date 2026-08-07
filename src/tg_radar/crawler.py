import asyncio
import random
import time

from curl_cffi.requests import AsyncSession
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from tg_radar.config import Settings
from tg_radar.parser import parse_tme_page
from tg_radar.schemas import ParsedPage
from tg_radar.text import clean_username


class RateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        self.interval = 1.0 / requests_per_second
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            sleep_for = self.interval - (now - self._last)
            if sleep_for > 0:
                await asyncio.sleep(sleep_for + random.uniform(0, self.interval * 0.2))
            self._last = time.monotonic()


class TelegramPublicCrawler:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.limiter = RateLimiter(settings.requests_per_second)

    def page_url(self, channel: str, before: int | None = None) -> str:
        username = clean_username(channel)
        if not username:
            raise ValueError(f"invalid channel username: {channel}")
        url = f"https://t.me/s/{username}"
        if before:
            url += f"?before={before}"
        return url

    @retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=1, max=15))
    async def fetch_html(self, channel: str, before: int | None = None) -> str:
        await self.limiter.wait()
        headers = {"user-agent": self.settings.user_agent}
        async with AsyncSession(timeout=self.settings.request_timeout_seconds, impersonate="chrome120") as session:
            response = await session.get(self.page_url(channel, before), headers=headers)
        if response.status_code == 404:
            raise FileNotFoundError(channel)
        response.raise_for_status()
        return response.text

    async def crawl_page(self, channel: str, before: int | None = None) -> ParsedPage:
        html = await self.fetch_html(channel, before)
        username = clean_username(channel)
        if not username:
            raise ValueError(f"invalid channel username: {channel}")
        return parse_tme_page(html, username)

    async def crawl_channel(self, channel: str, pages: int = 1) -> list[ParsedPage]:
        before: int | None = None
        out: list[ParsedPage] = []
        for _ in range(max(pages, 1)):
            page = await self.crawl_page(channel, before)
            if not page.messages:
                break
            out.append(page)
            if page.next_before is None or page.next_before == before:
                break
            before = page.next_before
        return out
