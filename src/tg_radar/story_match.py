from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tg_radar.agent_runtime.openai_transport import ChatCompletionsTransport, OpenAITransport, OpenAITransportConfig
from tg_radar.agent_runtime.prompt_loader import PromptLoader
from tg_radar.config import Settings
from tg_radar.db import Channel, Message, ResearchTopic, TopicMessage
from tg_radar.schemas import StoryCaseOut, StoryMatchPair, StoryMatchReport, StoryMatchSignalOut
from tg_radar.story_catalog import StoryCase, load_story_catalog, load_story_rules
from tg_radar.text import normalize_text


CONFIDENCE_ORDER = ("high", "medium", "low")
MIN_SIGNAL_CHARS = 120
MIN_YOU_ADD_CHARS = 40
SIGNAL_PROMPT_CHARS = 900
CASE_PROMPT_CHARS = 700


class StoryMatchError(RuntimeError):
    pass


@dataclass(frozen=True)
class StorySignal:
    message_id: str
    channel: str
    url: str
    posted_at: datetime | None
    text: str
    pain_type: str


async def collect_topic_signals(
    session: AsyncSession,
    topic_slug: str,
    window_days: int,
    limit: int,
) -> list[StorySignal]:
    topic = await session.scalar(select(ResearchTopic).where(ResearchTopic.slug == topic_slug))
    if not topic:
        raise StoryMatchError(f"topic not found: {topic_slug}")
    since = datetime.now(timezone.utc) - timedelta(days=max(window_days, 1))
    rows = (
        await session.execute(
            select(Message, Channel)
            .join(TopicMessage, TopicMessage.message_id == Message.id)
            .join(Channel, Channel.id == Message.channel_id)
            .where(
                TopicMessage.topic_id == topic.id,
                TopicMessage.pain_score >= 0.28,
                Message.posted_at.is_not(None),
                Message.posted_at >= since,
            )
            .order_by(Message.posted_at.desc(), Message.id.desc())
            .limit(min(max(limit, 1), 1000) * 3)
        )
    ).all()
    seen: set[str] = set()
    signals: list[StorySignal] = []
    for message, channel in rows:
        text = normalize_text(message.text or "")
        if len(text) < MIN_SIGNAL_CHARS:
            continue
        key = message.discussion_key or message.hash
        if key in seen:
            continue
        seen.add(key)
        signals.append(
            StorySignal(
                message_id=f"{channel.username}:{message.tg_msg_id}",
                channel=channel.username,
                url=message.url,
                posted_at=message.posted_at,
                text=text,
                pain_type=message.pain_type or "discussion",
            )
        )
        if len(signals) >= limit:
            break
    return signals


class StoryMatcher:
    def __init__(
        self,
        settings: Settings,
        cases: list[StoryCase] | None = None,
        transport: OpenAITransport | None = None,
        prompt_loader: PromptLoader | None = None,
    ) -> None:
        self.settings = settings
        self.cases = cases if cases is not None else load_story_catalog()
        self.transport = transport
        self.prompt_loader = prompt_loader or PromptLoader(override_dir=settings.content_prompt_dir)

    async def match(self, signals: list[StorySignal], min_confidence: str = "medium") -> list[StoryMatchPair]:
        if not self.cases:
            raise StoryMatchError("story catalog is empty")
        if not signals:
            return []
        transport = self.transport or self._build_transport()
        batch_size = max(self.settings.story_match_batch_size, 1)
        batches = [signals[index : index + batch_size] for index in range(0, len(signals), batch_size)]
        semaphore = asyncio.Semaphore(max(self.settings.story_match_concurrency, 1))

        async def run(batch: list[StorySignal]) -> list[dict[str, Any]]:
            async with semaphore:
                return await self._match_batch(transport, batch)

        results = await asyncio.gather(*(run(batch) for batch in batches))
        raw_pairs = [pair for batch_pairs in results for pair in batch_pairs]
        return self._group(raw_pairs, signals, min_confidence)

    def _build_transport(self) -> OpenAITransport:
        if not self.settings.llm_base_url or not self.settings.llm_api_key:
            raise StoryMatchError("llm is not configured; refusing to guess matches")
        return ChatCompletionsTransport(
            OpenAITransportConfig(
                base_url=self.settings.llm_base_url,
                api_key=self.settings.llm_api_key,
                model=self.settings.llm_model,
                timeout_seconds=self.settings.llm_timeout_seconds,
                temperature=0.0,
                max_tokens=self.settings.content_llm_max_tokens,
                json_response_format=False,
            )
        )

    async def _match_batch(self, transport: OpenAITransport, batch: list[StorySignal]) -> list[dict[str, Any]]:
        payload = {
            "catalog": [self._case_prompt(case) for case in self.cases],
            "signals": [self._signal_prompt(signal) for signal in batch],
        }
        try:
            content = await transport.complete_json(
                [
                    {"role": "system", "content": self.prompt_loader.load(self.settings.story_match_prompt)},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ]
            )
        except Exception as exc:
            raise StoryMatchError(f"llm request failed: {exc}") from exc
        return self._validate(content, batch)

    def _case_prompt(self, case: StoryCase) -> dict[str, Any]:
        return {
            "case_number": case.number,
            "title": case.title,
            "evidence": case.body[:CASE_PROMPT_CHARS],
            "takeaway": case.takeaway,
        }

    def _signal_prompt(self, signal: StorySignal) -> dict[str, Any]:
        return {
            "signal_url": signal.url,
            "channel": signal.channel,
            "posted_at": signal.posted_at.isoformat() if signal.posted_at else None,
            "text": signal.text[:SIGNAL_PROMPT_CHARS],
        }

    def _validate(self, content: str, batch: list[StorySignal]) -> list[dict[str, Any]]:
        data = _json_object(content)
        urls = {signal.url for signal in batch}
        numbers = {case.number for case in self.cases}
        validated: list[dict[str, Any]] = []
        claimed: set[str] = set()
        for item in data.get("pairs") or []:
            if not isinstance(item, dict):
                continue
            url = str(item.get("signal_url") or "")
            you_add = str(item.get("you_add") or "").strip()
            confidence = str(item.get("confidence") or "").strip().lower()
            try:
                case_number = int(item.get("case_number"))
            except (TypeError, ValueError):
                continue
            if url not in urls or url in claimed:
                continue
            if case_number not in numbers:
                continue
            if len(you_add) < MIN_YOU_ADD_CHARS:
                continue
            if confidence not in CONFIDENCE_ORDER:
                continue
            claimed.add(url)
            validated.append(
                {
                    "signal_url": url,
                    "case_number": case_number,
                    "signal_gist": str(item.get("signal_gist") or "").strip()[:300],
                    "you_add": you_add[:600],
                    "confidence": confidence,
                }
            )
        return validated

    def _group(self, pairs: list[dict[str, Any]], signals: list[StorySignal], min_confidence: str) -> list[StoryMatchPair]:
        floor = CONFIDENCE_ORDER.index(min_confidence) if min_confidence in CONFIDENCE_ORDER else len(CONFIDENCE_ORDER) - 1
        by_url = {signal.url: signal for signal in signals}
        by_case = {case.number: case for case in self.cases}
        grouped: dict[int, list[dict[str, Any]]] = {}
        for pair in pairs:
            if CONFIDENCE_ORDER.index(pair["confidence"]) > floor:
                continue
            grouped.setdefault(pair["case_number"], []).append(pair)
        out: list[StoryMatchPair] = []
        for case_number, items in grouped.items():
            items.sort(key=lambda item: (CONFIDENCE_ORDER.index(item["confidence"]), item["signal_url"]))
            case = by_case[case_number]
            matched = [
                StoryMatchSignalOut(
                    message_id=by_url[item["signal_url"]].message_id,
                    channel=by_url[item["signal_url"]].channel,
                    url=item["signal_url"],
                    posted_at=by_url[item["signal_url"]].posted_at,
                    pain_type=by_url[item["signal_url"]].pain_type,
                    gist=item["signal_gist"] or by_url[item["signal_url"]].text[:200],
                )
                for item in items
            ]
            channels = sorted({signal.channel for signal in matched})
            out.append(
                StoryMatchPair(
                    case=StoryCaseOut(
                        number=case.number,
                        title=case.title,
                        format=case.format,
                        section=case.section,
                        takeaway=case.takeaway,
                    ),
                    signals=matched,
                    you_add=items[0]["you_add"],
                    confidence=items[0]["confidence"],
                    channels=channels,
                    multi_channel=len(channels) >= 2,
                )
            )
        out.sort(
            key=lambda pair: (
                CONFIDENCE_ORDER.index(pair.confidence),
                -len(pair.signals),
                pair.case.number,
            )
        )
        return out


async def match_topic_stories(
    session: AsyncSession,
    settings: Settings,
    topic_slug: str,
    window_days: int | None = None,
    limit: int | None = None,
    min_confidence: str = "medium",
    matcher: StoryMatcher | None = None,
) -> StoryMatchReport:
    resolved_window = window_days or settings.story_match_window_days
    resolved_limit = limit or settings.story_match_signal_limit
    signals = await collect_topic_signals(session, topic_slug, resolved_window, resolved_limit)
    resolved_matcher = matcher or StoryMatcher(settings)
    pairs = await resolved_matcher.match(signals, min_confidence)
    return build_report(
        topic_slug,
        resolved_window,
        signals,
        pairs,
        len(resolved_matcher.cases),
        min_confidence,
    )


def build_report(
    topic_slug: str,
    window_days: int,
    signals: list[StorySignal],
    pairs: list[StoryMatchPair],
    cases_total: int,
    min_confidence: str,
) -> StoryMatchReport:
    templates = load_story_rules()["notes"]
    notes: list[str] = []
    if not signals:
        notes.append(str(templates["no_signals"]).format(window=window_days))
    elif not pairs:
        notes.append(str(templates["no_pairs"]))
    return StoryMatchReport(
        topic=topic_slug,
        window_days=window_days,
        generated_at=datetime.now(timezone.utc),
        signals_considered=len(signals),
        signals_matched=sum(len(pair.signals) for pair in pairs),
        cases_total=cases_total,
        min_confidence=min_confidence,
        pairs=pairs,
        notes=notes,
    )


def _json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise StoryMatchError("llm returned invalid json") from None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            raise StoryMatchError("llm returned invalid json") from None
    if not isinstance(data, dict):
        raise StoryMatchError("llm returned non-object json")
    return data
