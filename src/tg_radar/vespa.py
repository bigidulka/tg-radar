from datetime import datetime
from typing import Any

import httpx

from tg_radar.embeddings import Embedder
from tg_radar.schemas import ParsedMessage, SearchFilters, SearchHit
from tg_radar.scoring import hiring_signal, vacancy_key


def unix_ts(value: datetime | None) -> int:
    return int(value.timestamp()) if value else 0


class VespaClient:
    def __init__(self, endpoint: str, embedder: Embedder) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.embedder = embedder

    async def feed_message(self, message: ParsedMessage, quality_score: float = 1.0) -> None:
        doc_id = f"{message.channel_username}:{message.tg_msg_id}"
        signal = hiring_signal(" ".join([message.text, *message.links, *message.mentions]))
        key = vacancy_key(message.text, message.url)
        payload = {
            "fields": {
                "message_id": doc_id,
                "channel_username": message.channel_username,
                "channel_title": message.channel_title or "",
                "posted_at": unix_ts(message.posted_at),
                "text": message.text,
                "links": message.links,
                "mentions": message.mentions,
                "embedding": {"values": self.embedder.embed(message.text)},
                "quality_score": quality_score,
                "vacancy_key": key,
                "detector_score": signal.score,
                "detector_reasons": signal.reasons,
            }
        }
        url = f"{self.endpoint}/document/v1/message/message/docid/{doc_id}"
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()

    async def search(self, query: str, filters: SearchFilters, limit: int, semantic: bool = True) -> list[SearchHit]:
        yql = "select * from message where userQuery()"
        filter_parts: list[str] = []
        if filters.channel_usernames:
            values = ", ".join(f'"{x.strip("@")}"' for x in filters.channel_usernames)
            filter_parts.append(f"channel_username in ({values})")
        if filters.date_from:
            filter_parts.append(f"posted_at >= {unix_ts(filters.date_from)}")
        if filters.date_to:
            filter_parts.append(f"posted_at <= {unix_ts(filters.date_to)}")
        if filter_parts:
            yql += " and " + " and ".join(filter_parts)

        body: dict[str, Any] = {
            "yql": yql,
            "query": query,
            "hits": limit,
            "ranking": "hybrid" if semantic else "bm25",
            "presentation.summary": "default",
        }
        if semantic:
            body["input.query(q_embedding)"] = self.embedder.embed(query)

        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(f"{self.endpoint}/search/", json=body)
            response.raise_for_status()
        data = response.json()
        hits: list[SearchHit] = []
        for child in data.get("root", {}).get("children", []):
            fields = child.get("fields", {})
            reasons = ["vespa_hybrid" if semantic else "vespa_bm25"]
            reasons.extend(fields.get("detector_reasons") or [])
            if fields.get("quality_score"):
                reasons.append(f"quality_score:{float(fields.get('quality_score') or 0):.2f}")
            hits.append(
                SearchHit(
                    message_id=fields.get("message_id", ""),
                    channel_username=fields.get("channel_username", ""),
                    channel_title=fields.get("channel_title") or None,
                    url=f"https://t.me/{fields.get('message_id', '').replace(':', '/')}",
                    posted_at=datetime.fromtimestamp(fields["posted_at"]) if fields.get("posted_at") else None,
                    text=fields.get("text", ""),
                    score=float(child.get("relevance", 0.0)),
                    highlights=hiring_signal(fields.get("text", "")).highlights,
                    score_reasons=sorted(set(reasons)),
                    why_matched=sorted(set(reasons)),
                    cluster_key=fields.get("vacancy_key"),
                    cluster_size=1,
                )
            )
        return hits

    async def health(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{self.endpoint}/ApplicationStatus")
            response.raise_for_status()
            return response.json()
