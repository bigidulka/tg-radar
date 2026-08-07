from __future__ import annotations

from dataclasses import dataclass

from curl_cffi.requests import AsyncSession
from selectolax.parser import HTMLParser

from tg_radar.crawler import TelegramPublicCrawler
from tg_radar.text import clean_username, normalize_link


@dataclass(frozen=True)
class ChannelAvatar:
    channel: str
    url: str | None
    content_type: str | None = None
    data: bytes | None = None


class ChannelAvatarLoader:
    def __init__(self, crawler: TelegramPublicCrawler) -> None:
        self.crawler = crawler

    async def fetch_avatar_url(self, channel: str) -> ChannelAvatar:
        clean = clean_username(channel)
        if not clean:
            raise ValueError("invalid channel")
        html = await self.crawler.fetch_html(clean)
        return ChannelAvatar(channel=clean, url=parse_avatar_url(html))

    async def fetch_avatar(self, channel: str) -> ChannelAvatar:
        found = await self.fetch_avatar_url(channel)
        if not found.url:
            return found
        headers = {"user-agent": self.crawler.settings.user_agent}
        async with AsyncSession(timeout=self.crawler.settings.request_timeout_seconds, impersonate="chrome120") as session:
            response = await session.get(found.url, headers=headers)
        response.raise_for_status()
        return ChannelAvatar(
            channel=found.channel,
            url=found.url,
            content_type=response.headers.get("content-type"),
            data=response.content,
        )


def parse_avatar_url(html: str) -> str | None:
    tree = HTMLParser(html)
    selectors = (
        ".tgme_page_photo_image img",
        ".tgme_channel_info_header_photo img",
        "meta[property='og:image']",
        "meta[name='twitter:image']",
    )
    for selector in selectors:
        node = tree.css_first(selector)
        if not node:
            continue
        value = node.attributes.get("src") or node.attributes.get("content")
        parsed = normalize_link(value or "")
        if parsed:
            return parsed
    return None
