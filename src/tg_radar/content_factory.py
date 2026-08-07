from __future__ import annotations

from collections import Counter
from datetime import datetime
import hashlib
import json

import httpx
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from tg_radar.config import Settings
from tg_radar.content_rules import load_content_rules
from tg_radar.db import ArticleBlock, ArticleDraft, ArticleSession, Channel, ContentCard, Message, TopicMessage, ResearchTopic
from tg_radar.agent_runtime.prompt_loader import PromptLoader
from tg_radar.schemas import (
    ArticleBlockOut,
    ArticleDraftOut,
    ArticleSessionOut,
    ContentCardOut,
    PainInsight,
)
from tg_radar.text import normalize_text


CONTENT_RULES = load_content_rules()
RESEARCH_BASIS: dict[str, str] = CONTENT_RULES["research_basis"]
STAGE_ORDER: dict[str, int] = CONTENT_RULES["stage_order"]
MIN_EVIDENCE_QUALITY = 0.45
DEFAULT_MIN_USABLE_EVIDENCE = 5
MIN_FOCUS_MATCH = 0.22

SIGNAL_RULES: dict[str, list[str]] = {
    "pricing_pain": ["price", "cost", "discount", "promo", "$", "\u20bd", "\u0446\u0435\u043d", "\u0434\u0435\u0448\u0435\u0432", "\u0441\u043a\u0438\u0434"],
    "payment_friction": ["payment", "pay", "card", "stripe", "\u043e\u043f\u043b\u0430\u0442", "\u043a\u0430\u0440\u0442"],
    "grey_market": ["account", "shared", "proxy", "\u0430\u043a\u043a\u0430\u0443\u043d\u0442", "\u0441\u0435\u0440\u044b", "\u043e\u0431\u0445\u043e\u0434"],
    "limits": ["limit", "quota", "rate", "\u043b\u0438\u043c\u0438\u0442"],
    "account_risk": ["ban", "risk", "verify", "access", "\u0431\u0430\u043d", "\u0440\u0438\u0441\u043a", "\u0434\u043e\u0441\u0442\u0443\u043f"],
    "workaround": ["workaround", "hack", "vpn", "\u043b\u0430\u0439\u0444\u0445\u0430\u043a", "\u043e\u0431\u0445\u043e\u0434"],
    "tool_comparison": ["cursor", "claude", "chatgpt", "gemini", "grok", "windsurf", "copilot"],
}

SOURCE_FLOOR = 0.35


def _quality_terms() -> dict[str, list[str]]:
    return {
        "subscription": [
            "\u043f\u043e\u0434\u043f\u0438\u0441",
            "subscription",
            "plus",
            "ultra",
        ],
        "money": [
            "\u0446\u0435\u043d",
            "\u0434\u0435\u0448\u0435\u0432",
            "\u0441\u043a\u0438\u0434",
            "\u043f\u0440\u043e\u043c\u043e\u043a\u043e\u0434",
            "\u043e\u043f\u043b\u0430\u0442",
            "\u043b\u0438\u043c\u0438\u0442",
            "\u0440\u0443\u0431",
            "\u20bd",
            "$",
        ],
        "ai": ["chatgpt", "claude", "cursor", "gemini", "grok", "openai", "anthropic", "\u043d\u0435\u0439\u0440\u043e"],
        "spam": [
            "\u0432\u0430\u043a\u0430\u043d\u0441",
            "\u0440\u0435\u0437\u044e\u043c",
            "\u0437\u0430\u0440\u043f\u043b\u0430\u0442",
            "\u0437/\u043f",
            "\u043a\u0430\u043d\u0434\u0438\u0434\u0430\u0442",
            "\u0443\u0434\u0430\u043b\u0435\u043d\u043d\u043e",
            "\u0443\u0432\u043e\u043b",
            "\u043c\u0435\u043d\u0435\u0434\u0436\u0435\u0440",
            "smm",
            "hr",
            "\u0441\u0432\u043e",
            "\u0433\u0440\u0443\u0437\u0438",
            "\u0443\u043a\u0440\u0430\u0438\u043d",
            "\u043c\u043e\u043b\u0434\u043e\u0432",
        ],
    }


def _tokens(value: str) -> list[str]:
    out: list[str] = []
    current: list[str] = []
    for char in value.lower():
        keep = char.isalnum() or char in {"_", "-"}
        if keep:
            current.append(char)
            continue
        if len(current) >= 3:
            out.append("".join(current))
        current = []
    if len(current) >= 3:
        out.append("".join(current))
    return out


def _focus_key(article_focus: str | None, focus_keywords: list[str] | None = None) -> str | None:
    terms = _focus_terms(article_focus, focus_keywords)
    if not terms:
        return None
    return hashlib.sha1(" ".join(terms[:24]).encode("utf-8")).hexdigest()[:12]


def _focus_terms(article_focus: str | None, focus_keywords: list[str] | None = None) -> list[str]:
    raw = " ".join([article_focus or "", " ".join(focus_keywords or [])]).lower()
    tokens = _tokens(raw)
    stop = {
        "\u0440\u0430\u0431\u043e\u0442\u0430\u0435\u043c",
        "\u0433\u043e\u0442\u043e\u0432\u043e\u043c\u0443",
        "\u0442\u043e\u043f\u0438\u043a\u0443",
        "topic",
        "telegram",
        "evidence",
        "article",
        "draft",
        "outline",
        "final",
        "write",
        "\u043d\u0430\u043f\u0438\u0448\u0438",
        "\u0441\u0442\u0430\u0442\u044c\u044e",
        "\u0444\u0438\u043d\u0430\u043b\u044c\u043d\u0443\u044e",
        "\u0442\u043e\u043b\u044c\u043a\u043e",
        "\u0447\u0435\u0440\u0435\u0437",
        "\u0444\u0430\u043a\u0442\u044b",
        "\u0441\u043d\u0430\u0447\u0430\u043b\u0430",
        "claims",
        "claim",
        "verify",
    }
    terms = [token for token in tokens if token not in stop and len(token) >= 4]
    aliases: list[str] = []
    if any(part in raw for part in ("\u044d\u043a\u043e\u043d\u043e\u043c", "\u0434\u0435\u0448\u0435\u0432", "\u0441\u043a\u0438\u0434", "\u0446\u0435\u043d", "\u043f\u043e\u043a\u0443\u043f", "\u043f\u043e\u0434\u043f\u0438\u0441", "subscription", "pricing", "price", "cost")):
        aliases.extend(
            [
                "\u044d\u043a\u043e\u043d\u043e\u043c",
                "\u0434\u0435\u0448\u0435\u0432",
                "\u0441\u043a\u0438\u0434",
                "\u0446\u0435\u043d",
                "\u043f\u043e\u0434\u043f\u0438\u0441\u043a",
                "subscription",
                "plus",
                "pro",
                "\u043e\u043f\u043b\u0430\u0442",
                "payment",
                "billing",
                "\u0440\u0443\u0431",
                "$",
            ]
        )
    if any(part in raw for part in ("\u0441\u0435\u0440", "gray", "grey", "\u043e\u0431\u0445\u043e\u0434", "vpn", "\u0430\u043a\u043a\u0430\u0443\u043d\u0442", "account", "access", "token", "\u043b\u0438\u043c\u0438\u0442", "limit")):
        aliases.extend(["\u0441\u0435\u0440", "\u043e\u0431\u0445\u043e\u0434", "vpn", "\u0430\u043a\u043a\u0430\u0443\u043d\u0442", "account", "access", "token", "\u043b\u0438\u043c\u0438\u0442", "limit", "ban", "risk"])
    if any(part in raw for part in ("\u043b\u043e\u043a\u0430\u043b\u044c", "free", "\u0431\u0435\u0441\u043f\u043b\u0430\u0442", "offline", "gemma", "local")):
        aliases.extend(["free", "\u0431\u0435\u0441\u043f\u043b\u0430\u0442", "\u043b\u043e\u043a\u0430\u043b\u044c", "offline", "gemma", "local"])
    return sorted(set([*terms, *aliases]))


def _focus_match_score(text: str, article_focus: str | None, focus_keywords: list[str] | None = None) -> tuple[float, list[str]]:
    terms = _focus_terms(article_focus, focus_keywords)
    if not terms:
        return 0.0, []
    lower = text.lower()
    text_tokens = set(_tokens(lower))
    hits = [term for term in terms if _focus_term_hit(term, lower, text_tokens)]
    core_terms = {"\u044d\u043a\u043e\u043d\u043e\u043c", "\u0434\u0435\u0448\u0435\u0432", "\u0441\u043a\u0438\u0434", "\u0446\u0435\u043d", "\u043f\u043e\u0434\u043f\u0438\u0441\u043a", "subscription", "\u043e\u043f\u043b\u0430\u0442", "payment", "billing", "\u0440\u0443\u0431", "$", "\u043b\u0438\u043c\u0438\u0442", "limit", "usage-based"}
    core_hits = [hit for hit in hits if hit in core_terms]
    score = min(1.0, len(core_hits) / 5 * 0.75 + (len(hits) - len(core_hits)) / 8 * 0.25)
    if len(core_hits) >= 2:
        score = max(score, 0.42)
    if len(core_hits) >= 4:
        score = max(score, 0.68)
    return score, hits[:12]


def _focus_term_hit(term: str, lower: str, text_tokens: set[str]) -> bool:
    if term == "$":
        return "$" in lower
    if term == "\u043f\u043e\u0434\u043f\u0438\u0441":
        return "\u043f\u043e\u0434\u043f\u0438\u0441\u043a" in lower
    if term == "pro":
        return "pro" in text_tokens
    if len(term) <= 3 and term.isascii():
        return term in text_tokens
    return term in lower


def card_to_out(row: ContentCard) -> ContentCardOut:
    return ContentCardOut(
        id=row.id,
        topic_slug=row.topic_slug,
        card_type=row.card_type,
        title=row.title,
        summary=row.summary,
        body=row.body,
        source_channel=row.source_channel,
        source_url=row.source_url,
        score=row.score or 0.0,
        status=row.status,
        tags=row.tags or [],
        payload=row.payload or {},
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def block_to_out(row: ArticleBlock) -> ArticleBlockOut:
    return ArticleBlockOut(
        id=row.id,
        position=row.position,
        block_type=row.block_type,
        title=row.title,
        text=row.text,
        status=row.status,
        source_card_ids=row.source_card_ids or [],
    )


def draft_to_out(row: ArticleDraft) -> ArticleDraftOut:
    return ArticleDraftOut(
        id=row.id,
        session_id=row.session_id,
        version=row.version,
        status=row.status,
        format=row.format,
        text=row.text,
        rich_json=row.rich_json or {},
        score=row.score or 0.0,
        created_at=row.created_at,
    )


async def article_to_out(session: AsyncSession, row: ArticleSession) -> ArticleSessionOut:
    blocks = (
        await session.execute(
            select(ArticleBlock)
            .where(ArticleBlock.session_id == row.id)
            .order_by(ArticleBlock.position, ArticleBlock.id)
        )
    ).scalars().all()
    drafts = (
        await session.execute(
            select(ArticleDraft)
            .where(ArticleDraft.session_id == row.id)
            .order_by(ArticleDraft.version.desc(), ArticleDraft.id.desc())
        )
    ).scalars().all()
    return ArticleSessionOut(
        id=row.id,
        topic_slug=row.topic_slug,
        title=row.title,
        audience=row.audience,
        angle=row.angle,
        status=row.status,
        source_card_ids=row.source_card_ids or [],
        blocks=[block_to_out(block) for block in blocks],
        drafts=[draft_to_out(draft) for draft in drafts],
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


async def list_cards(session: AsyncSession, topic_slug: str, limit: int = 120) -> list[ContentCardOut]:
    rows = (
        await session.execute(
            select(ContentCard)
            .where(ContentCard.topic_slug == topic_slug)
            .order_by(ContentCard.updated_at.desc())
            .limit(500)
        )
    ).scalars().all()
    status_rank = {"approved": 0, "saved": 1, "new": 2, "used": 3, "rejected": 9}
    ordered = sorted(
        rows,
        key=lambda row: (
            status_rank.get(row.status, 5),
            int((row.payload or {}).get("stage_order") or STAGE_ORDER.get(row.card_type, 999)),
            -(row.score or 0.0),
            row.id,
        ),
    )
    return [card_to_out(row) for row in ordered[: min(max(limit, 1), 500)]]


async def list_articles(session: AsyncSession, topic_slug: str | None = None, limit: int = 20) -> list[ArticleSessionOut]:
    stmt = select(ArticleSession).order_by(ArticleSession.updated_at.desc(), ArticleSession.id.desc()).limit(min(max(limit, 1), 100))
    if topic_slug:
        stmt = stmt.where(ArticleSession.topic_slug == topic_slug)
    rows = (await session.execute(stmt)).scalars().all()
    return [await article_to_out(session, row) for row in rows]


def _topic_match_score(text: str, keywords: list[str], relevance: float, pain: float) -> tuple[float, list[str], list[str]]:
    lower = text.lower()
    keyword_phrases = [item.lower().strip() for item in keywords if len(item.strip()) >= 3]
    keyword_tokens = sorted({token for item in keyword_phrases for token in _tokens(item) if len(token) >= 4})
    phrase_hits = [item for item in keyword_phrases if item in lower]
    token_hits = [item for item in keyword_tokens if item in lower]
    terms = _quality_terms()
    ai_hits = [item for item in terms["ai"] if item in lower]
    subscription_hits = [item for item in terms["subscription"] if item in lower]
    money_hits = [item for item in terms["money"] if item in lower]
    spam_hits = [item for item in terms["spam"] if item in lower]
    topic_text = " ".join(keyword_phrases)
    requires_subscription_anchor = any(item in topic_text for item in terms["subscription"])
    keyword_score = min(1.0, len(phrase_hits) * 0.22 + (len(token_hits) / max(len(keyword_tokens), 1)) * 0.55)
    context_bonus = 0.0
    if subscription_hits and (money_hits or ai_hits):
        context_bonus += 0.18
    if ai_hits and money_hits:
        context_bonus += 0.14
    if len(text) >= 180:
        context_bonus += 0.04
    spam_penalty = 0.0
    if spam_hits and not (ai_hits and subscription_hits and phrase_hits):
        spam_penalty += 0.28
    if len(token_hits) <= 1 and not phrase_hits and not (ai_hits and subscription_hits):
        spam_penalty += 0.22
    weak_anchor = False
    if requires_subscription_anchor and not subscription_hits and not phrase_hits:
        weak_anchor = True
        spam_penalty += 0.35
    if requires_subscription_anchor and spam_hits and not phrase_hits:
        weak_anchor = True
        spam_penalty += 0.45
    if not requires_subscription_anchor and not phrase_hits and len(token_hits) < 2 and not (ai_hits and subscription_hits):
        weak_anchor = True
        spam_penalty += 0.2
    score = max(0.0, min(1.0, relevance * 0.25 + pain * 0.18 + keyword_score * 0.5 + context_bonus - spam_penalty))
    reasons = []
    if phrase_hits:
        reasons.append("keyword_phrase")
    if token_hits:
        reasons.append("keyword_token")
    if ai_hits:
        reasons.append("ai_term")
    if subscription_hits:
        reasons.append("subscription_term")
    if money_hits:
        reasons.append("money_term")
    reject_reasons = []
    if score < MIN_EVIDENCE_QUALITY:
        reject_reasons.append("low_topic_match")
    if spam_penalty:
        reject_reasons.append("spam_or_adjacent_topic")
    if weak_anchor:
        reject_reasons.append("weak_topic_anchor")
    if len(text.strip()) < 80:
        reject_reasons.append("too_short")
    return score, reasons, reject_reasons


def _source_quality_score(channel: Channel) -> float:
    values = [channel.quality_score or 0.0, channel.source_trust or 0.0]
    if channel.ai_llm_match:
        values.append(0.72)
    if channel.public_chat:
        values.append(0.55)
    return max([*values, SOURCE_FLOOR])


def _evidence_quality_score(text: str, topic_match: float, pain: float, source_quality: float, novelty: float) -> float:
    length_bonus = 0.0
    if len(text) >= 220:
        length_bonus += 0.08
    if len(text) >= 600:
        length_bonus += 0.05
    return min(1.0, topic_match * 0.42 + source_quality * 0.2 + pain * 0.16 + novelty * 0.14 + length_bonus)


def _signal_name(text: str, pain_type: str | None, article_focus: str | None = None) -> str:
    lowered = text.lower()
    matches: list[tuple[str, int]] = []
    for name, tokens in SIGNAL_RULES.items():
        hits = sum(1 for token in tokens if token in lowered)
        if hits:
            matches.append((name, hits))
    if matches:
        matches.sort(key=lambda item: (-item[1], item[0]))
        return matches[0][0]
    return pain_type or "general_discussion"


def _confidence(frequency: int, avg_quality: float) -> str:
    if frequency >= 5 and avg_quality >= 0.62:
        return "high"
    if frequency >= 2 and avg_quality >= 0.5:
        return "medium"
    return "low"


async def review_topic_evidence(
    session: AsyncSession,
    topic_slug: str,
    limit: int = 80,
    min_quality: float = MIN_EVIDENCE_QUALITY,
    min_required: int = DEFAULT_MIN_USABLE_EVIDENCE,
    article_focus: str | None = None,
    focus_keywords: list[str] | None = None,
) -> dict:
    topic = await session.scalar(select(ResearchTopic).where(ResearchTopic.slug == topic_slug))
    if not topic:
        return {"topic": topic_slug, "ok": False, "reason": "topic not found", "usable": 0, "reviewed": 0, "items": [], "usable_message_ids": []}
    has_focus = bool(_focus_terms(article_focus, focus_keywords))
    fetch_limit = min(max(limit * (20 if has_focus else 5), 1200 if has_focus else 50), 1500 if has_focus else 1000)
    rows = (
        await session.execute(
            select(Message, Channel, TopicMessage)
            .join(TopicMessage, TopicMessage.message_id == Message.id)
            .join(Channel, Channel.id == Message.channel_id)
            .where(TopicMessage.topic_id == topic.id)
            .order_by(TopicMessage.relevance_score.desc(), TopicMessage.pain_score.desc(), Message.posted_at.desc().nulls_last())
            .limit(fetch_limit)
        )
    ).all()
    reviewed = []
    usable_ids = []
    seen_discussions: set[str] = set()
    for message, channel, topic_message in rows:
        text = message.text or ""
        discussion_key = message.discussion_key or message.hash
        duplicate = discussion_key in seen_discussions
        seen_discussions.add(discussion_key)
        quality, reasons, reject_reasons = _topic_match_score(
            text,
            topic.keywords or [],
            topic_message.relevance_score or 0.0,
            topic_message.pain_score or message.pain_score or 0.0,
        )
        focus_score, focus_hits = _focus_match_score(text, article_focus, focus_keywords)
        if _focus_terms(article_focus, focus_keywords):
            if focus_score < MIN_FOCUS_MATCH:
                reject_reasons.append("low_article_focus_match")
            else:
                reasons.append("article_focus")
                if focus_score >= 0.6:
                    reject_reasons = [reason for reason in reject_reasons if reason not in {"low_topic_match", "weak_topic_anchor"}]
                if focus_score >= 0.8:
                    reject_reasons = [reason for reason in reject_reasons if reason != "spam_or_adjacent_topic"]
        novelty = _novelty_score(text, topic.keywords or [])
        source_quality = _source_quality_score(channel)
        evidence_quality = _evidence_quality_score(
            text,
            quality,
            topic_message.pain_score or message.pain_score or 0.0,
            source_quality,
            novelty,
        )
        if _focus_terms(article_focus, focus_keywords):
            evidence_quality = min(1.0, evidence_quality * 0.55 + focus_score * 0.45)
        if duplicate:
            reject_reasons.append("duplicate_discussion")
        topic_gate = quality >= min_quality or (has_focus and focus_score >= 0.6)
        usable = topic_gate and evidence_quality >= min_quality and len(text.strip()) >= 80 and not reject_reasons
        if usable:
            usable_ids.append(message.id)
        reviewed.append(
            {
                "message_pk": message.id,
                "message_id": f"{channel.username}:{message.tg_msg_id}",
                "channel": channel.username,
                "url": message.url,
                "quality": round(evidence_quality, 3),
                "topic_match_score": round(quality, 3),
                "article_focus_score": round(focus_score, 3),
                "evidence_quality_score": round(evidence_quality, 3),
                "pain_score": round(topic_message.pain_score or message.pain_score or 0.0, 3),
                "novelty_score": round(novelty, 3),
                "source_quality_score": round(source_quality, 3),
                "relevance_score": round(topic_message.relevance_score or 0.0, 3),
                "signal_name": _signal_name(text, topic_message.pain_type or message.pain_type, article_focus),
                "posted_at": message.posted_at.isoformat() if message.posted_at else None,
                "usable": usable,
                "reasons": reasons,
                "reject_reasons": reject_reasons,
                "focus_hits": focus_hits,
                "text": normalize_text(text)[:500],
            }
        )
    usable_items = [item for item in reviewed if item["usable"]]
    rejected_items = [item for item in reviewed if not item["usable"]]
    usable_items.sort(key=lambda item: (-item["quality"], -item["pain_score"], item["message_id"]))
    rejected_items.sort(key=lambda item: (-item["quality"], item["message_id"]))
    item_limit = min(max(limit, 80) if has_focus else limit, 120 if has_focus else 50)
    usable_ids = [int(item["message_pk"]) for item in usable_items[:item_limit]]
    return {
        "topic": topic_slug,
        "article_focus": article_focus,
        "focus_key": _focus_key(article_focus, focus_keywords),
        "focus_terms": _focus_terms(article_focus, focus_keywords)[:24],
        "ok": len(usable_items) >= min_required,
        "reason": "enough usable evidence" if len(usable_items) >= min_required else "not enough usable evidence",
        "reviewed": len(reviewed),
        "usable": len(usable_items),
        "rejected": len(rejected_items),
        "min_required": min_required,
        "min_quality": min_quality,
        "usable_message_ids": usable_ids,
        "items": usable_items[:item_limit],
        "rejected_sample": rejected_items[:10],
    }


async def infer_topic_signal_taxonomy(
    session: AsyncSession,
    settings: Settings,
    topic_slug: str,
    limit: int = 80,
    min_required: int = DEFAULT_MIN_USABLE_EVIDENCE,
    article_focus: str | None = None,
    focus_keywords: list[str] | None = None,
    max_signals: int = 8,
) -> dict:
    review = await review_topic_evidence(
        session,
        topic_slug,
        limit,
        min_required=min_required,
        article_focus=article_focus,
        focus_keywords=focus_keywords,
    )
    if not review.get("ok"):
        return {**review, "taxonomy": [], "taxonomy_card": None}
    items = list(review.get("items") or [])[: min(max(limit, 40), 120)]
    taxonomy = await _llm_signal_taxonomy(settings, topic_slug, article_focus, items, max_signals)
    if not taxonomy:
        taxonomy = _fallback_signal_taxonomy(article_focus, items, max_signals)
    taxonomy = _normalize_taxonomy(taxonomy, max_signals)
    if not taxonomy:
        taxonomy = _normalize_taxonomy(_fallback_signal_taxonomy(article_focus, items, max_signals), max_signals)
    card = await _upsert_card(
        session,
        {
            "topic_slug": topic_slug,
            "card_type": "taxonomy",
            "title": f"{topic_slug} signal taxonomy",
            "summary": f"signals={len(taxonomy)} focus={_focus_key(article_focus, focus_keywords) or 'topic'}",
            "body": json.dumps(taxonomy, ensure_ascii=False),
            "source_channel": None,
            "source_url": None,
            "score": 0.7 if taxonomy else 0.0,
            "tags": ["taxonomy", topic_slug],
            "payload": {
                "stage": "taxonomy",
                "stage_order": 25,
                "artifact": "taxonomy",
                "article_focus": article_focus,
                "focus_key": _focus_key(article_focus, focus_keywords),
                "taxonomy": taxonomy,
            },
        },
    )
    return {
        **review,
        "ok": bool(taxonomy),
        "taxonomy": taxonomy,
        "taxonomy_card": card.id,
    }


async def _latest_taxonomy(session: AsyncSession, topic_slug: str, focus_key: str | None) -> list[dict] | None:
    rows = (
        await session.execute(
            select(ContentCard)
            .where(ContentCard.topic_slug == topic_slug, ContentCard.card_type == "taxonomy")
            .order_by(ContentCard.updated_at.desc(), ContentCard.id.desc())
            .limit(20)
        )
    ).scalars().all()
    for row in rows:
        payload = row.payload or {}
        if payload.get("focus_key") != focus_key:
            continue
        taxonomy = payload.get("taxonomy")
        if isinstance(taxonomy, list) and taxonomy:
            return _normalize_taxonomy([item for item in taxonomy if isinstance(item, dict)], 12)
    return None


async def cluster_topic_signals(
    session: AsyncSession,
    topic_slug: str,
    limit: int = 80,
    min_required: int = DEFAULT_MIN_USABLE_EVIDENCE,
    article_focus: str | None = None,
    focus_keywords: list[str] | None = None,
    settings: Settings | None = None,
    taxonomy: list[dict] | None = None,
) -> dict:
    review = await review_topic_evidence(session, topic_slug, limit, min_required=min_required, article_focus=article_focus, focus_keywords=focus_keywords)
    if not review.get("ok"):
        return {**review, "signals": [], "signal_cards": []}
    if taxonomy is None:
        taxonomy = await _latest_taxonomy(session, topic_slug, _focus_key(article_focus, focus_keywords))
    if taxonomy is None:
        taxonomy_data = (
            await infer_topic_signal_taxonomy(session, settings, topic_slug, limit, min_required, article_focus, focus_keywords)
            if settings
            else {"taxonomy": _fallback_signal_taxonomy(article_focus, list(review.get("items") or []), 8)}
        )
        taxonomy = list(taxonomy_data.get("taxonomy") or [])
    grouped: dict[str, list[dict]] = {}
    for item in review.get("items", []):
        signal_name = _cluster_signal_name(item, taxonomy)
        grouped.setdefault(signal_name, []).append(item)
    signals = []
    cards = []
    for name, items in sorted(grouped.items(), key=lambda pair: (-len(pair[1]), pair[0])):
        items = sorted(items, key=_representative_rank)
        avg_quality = sum(float(item.get("evidence_quality_score") or item.get("quality") or 0.0) for item in items) / max(len(items), 1)
        representatives = items[:5]
        signal = {
            "name": name,
            "summary": _signal_summary(name, representatives),
            "frequency": len(items),
            "representative_msgs": [item["message_id"] for item in representatives],
            "representative_urls": [item.get("url") for item in representatives if item.get("url")],
            "counterexamples": [],
            "confidence": _confidence(len(items), avg_quality),
            "avg_quality": round(avg_quality, 3),
            "evidence_ids": [int(item["message_pk"]) for item in items],
        }
        signals.append(signal)
        card = await _upsert_card(
            session,
            {
                "topic_slug": topic_slug,
                "card_type": "signal",
                "title": name,
                "summary": signal["summary"],
                "body": "\n".join(f"- {item['message_id']}: {item['text'][:220]}" for item in representatives),
                "source_channel": None,
                "source_url": representatives[0].get("url") if representatives else None,
                "score": min(1.0, avg_quality + min(len(items), 8) * 0.03),
                "tags": ["signal", name, signal["confidence"]],
                "payload": {
                    "stage": "signal",
                    "stage_order": 30,
                    "artifact": "signal",
                    "article_focus": article_focus,
                    "focus_key": _focus_key(article_focus, focus_keywords),
                    "frequency": len(items),
                    "confidence": signal["confidence"],
                    "evidence_ids": signal["evidence_ids"],
                    "representative_msgs": signal["representative_msgs"],
                },
            },
        )
        cards.append(card.id)
    return {
        "topic": topic_slug,
        "article_focus": article_focus,
        "focus_key": _focus_key(article_focus, focus_keywords),
        "ok": bool(signals),
        "reviewed": review.get("reviewed", 0),
        "usable": review.get("usable", 0),
        "rejected": review.get("rejected", 0),
        "signals": signals,
        "signal_cards": cards,
    }


def _representative_rank(item: dict) -> tuple[float, float, str]:
    text = str(item.get("text") or "").lower()
    score = 0.0
    if "#redigest" in text or "\u0434\u0430\u0439\u0434\u0436\u0435\u0441\u0442" in text or "\u043d\u0435\u0439\u0440\u043e\u0434\u0430\u0439\u0434\u0436\u0435\u0441\u0442" in text:
        score += 0.8
    if any(marker in text for marker in ("\u043e\u0444\u0438\u0446\u0438\u0430\u043b\u044c\u043d\u0430\u044f \u0446\u0435\u043d\u0430", "\u0444\u0438\u043b\u0438\u043f\u043f", "\u0435\u0433\u0438\u043f", "\u043a\u0430\u043a \u043e\u043f\u043b\u0430\u0447\u0438\u0432\u0430\u0442\u044c", "\u043f\u043e\u0434\u043f\u0438\u0441\u043a\u0438 \u043d\u0430 \u0438\u0438", "\u0443\u0441\u043b\u043e\u0432\u043d\u043e \u0431\u0435\u0441\u043f\u043b\u0430\u0442\u043d\u044b\u0439", "\u043b\u0438\u043c\u0438\u0442\u044b \u043f\u043e\u0434\u043f\u0438\u0441\u043e\u043a")):
        score -= 0.8
    return (score, -float(item.get("quality") or 0.0), str(item.get("message_id") or ""))


async def _llm_signal_taxonomy(settings: Settings, topic_slug: str, article_focus: str | None, items: list[dict], max_signals: int) -> list[dict]:
    if not settings.llm_base_url or not settings.llm_api_key:
        return []
    payload = {
        "task": "Infer a reusable signal taxonomy from Telegram evidence. Return JSON array only. Labels must describe mechanisms, pains, tradeoffs, or behavior patterns; avoid raw product names or single keywords unless unavoidable.",
        "topic": topic_slug,
        "article_focus": article_focus,
        "max_signals": max_signals,
        "evidence": [
            {
                "message_id": item.get("message_id"),
                "quality": item.get("quality"),
                "focus_hits": item.get("focus_hits") or [],
                "text": item.get("text"),
            }
            for item in items[:40]
        ],
        "schema": [
            {
                "name": "short_ascii_slug",
                "description": "what repeated pattern means",
                "inclusion_terms": ["terms or phrases that identify this signal"],
                "exclusion_terms": ["terms or phrases that should not belong"],
            }
        ],
    }
    data = await _llm_json(settings, payload)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict) and isinstance(data.get("taxonomy"), list):
        return [item for item in data["taxonomy"] if isinstance(item, dict)]
    return []


def _fallback_signal_taxonomy(article_focus: str | None, items: list[dict], max_signals: int) -> list[dict]:
    buckets: dict[str, Counter[str]] = {}
    stop_terms = _taxonomy_stop_terms()
    for item in items:
        terms = _taxonomy_terms(item, stop_terms)
        if not terms:
            continue
        roots = _taxonomy_roots(terms)
        for root in roots:
            key = _taxonomy_slug(root)
            if key:
                buckets.setdefault(key, Counter()).update(terms[:18])
    ranked = sorted(buckets.items(), key=lambda pair: (-sum(pair[1].values()), pair[0]))[:max_signals]
    return [
        {
            "name": name,
            "description": f"Repeated evidence pattern around {name.replace('_', ' ')}.",
            "inclusion_terms": [term for term, _ in counts.most_common(12)],
            "exclusion_terms": [],
        }
        for name, counts in ranked
    ]


def _taxonomy_terms(item: dict, stop_terms: set[str]) -> list[str]:
    hits = [str(term).lower().strip() for term in (item.get("focus_hits") or []) if str(term).strip()]
    text_terms = [token for token in _tokens(str(item.get("text") or "")) if len(token) >= 5]
    ordered: list[str] = []
    for term in [*hits, *text_terms[:30]]:
        if term in stop_terms or len(term) < 4:
            continue
        if any(term.startswith(existing[:5]) or existing.startswith(term[:5]) for existing in ordered):
            continue
        if term not in ordered:
            ordered.append(term)
    return ordered


def _taxonomy_roots(terms: list[str]) -> list[str]:
    if len(terms) >= 2:
        roots = [f"{terms[0]} {terms[1]}"]
        if len(terms) >= 4:
            roots.append(f"{terms[2]} {terms[3]}")
        return roots
    if terms:
        return [terms[0]]
    return []


def _taxonomy_stop_terms() -> set[str]:
    return {
        "openai",
        "claude",
        "codex",
        "cursor",
        "grok",
        "chatgpt",
        "gemini",
        "model",
        "models",
        "plus",
        "pro",
        "redigest",
        "digest",
        "news",
        "agent",
        "agents",
        "tool",
        "tools",
        "\u043c\u043e\u0434\u0435\u043b",
        "\u043c\u043e\u0434\u0435\u043b\u0438",
        "\u0430\u0433\u0435\u043d\u0442",
        "\u0441\u0435\u0440\u0432\u0438\u0441",
        "\u0438\u043d\u0441\u0442\u0440\u0443\u043c\u0435\u043d\u0442",
        "\u0434\u0430\u0439\u0434\u0436\u0435\u0441\u0442",
        "\u043d\u043e\u0432\u043e\u0441\u0442\u0438",
        "\u0440\u0443\u0431\u0440\u0438\u043a\u0443",
    }


def _normalize_taxonomy(taxonomy: list[dict], max_signals: int) -> list[dict]:
    normalized = []
    seen = set()
    for item in taxonomy:
        name = _taxonomy_slug(str(item.get("name") or item.get("title") or ""))
        if not name or name in seen:
            continue
        seen.add(name)
        inclusion = [str(term).lower().strip() for term in item.get("inclusion_terms") or item.get("terms") or [] if str(term).strip()]
        exclusion = [str(term).lower().strip() for term in item.get("exclusion_terms") or [] if str(term).strip()]
        normalized.append(
            {
                "name": name,
                "description": str(item.get("description") or item.get("summary") or name.replace("_", " ")),
                "inclusion_terms": inclusion[:20],
                "exclusion_terms": exclusion[:20],
            }
        )
        if len(normalized) >= max_signals:
            break
    return normalized


def _taxonomy_slug(value: str) -> str:
    tokens = _tokens(value)
    ascii_tokens = [token for token in tokens if token.isascii()]
    selected = ascii_tokens or tokens
    return "_".join(selected[:4])[:64]


def _assign_taxonomy_signal(item: dict, taxonomy: list[dict]) -> str | None:
    if not taxonomy:
        return None
    text = str(item.get("text") or "").lower()
    focus_hits = {str(term).lower() for term in (item.get("focus_hits") or [])}
    scored: list[tuple[int, str]] = []
    for signal in taxonomy:
        name = str(signal.get("name") or "")
        inclusion = [str(term).lower() for term in signal.get("inclusion_terms") or [] if str(term).strip()]
        exclusion = [str(term).lower() for term in signal.get("exclusion_terms") or [] if str(term).strip()]
        if exclusion and any(term in text for term in exclusion):
            continue
        score = sum(2 for term in inclusion if term in focus_hits) + sum(1 for term in inclusion if term and term in text)
        if score:
            scored.append((score, name))
    if not scored:
        return None
    scored.sort(key=lambda item_: (-item_[0], item_[1]))
    return scored[0][1]


def _cluster_signal_name(item: dict, taxonomy: list[dict]) -> str:
    assigned_signal = _assign_taxonomy_signal(item, taxonomy)
    if assigned_signal:
        return assigned_signal
    if taxonomy:
        return "uncategorized_evidence"
    return str(item.get("signal_name") or "general_discussion")


async def build_topic_evidence_map(
    session: AsyncSession,
    topic_slug: str,
    limit: int = 80,
    min_required: int = DEFAULT_MIN_USABLE_EVIDENCE,
    article_focus: str | None = None,
    focus_keywords: list[str] | None = None,
    settings: Settings | None = None,
    taxonomy: list[dict] | None = None,
) -> dict:
    clustered = await cluster_topic_signals(
        session,
        topic_slug,
        limit,
        min_required=min_required,
        article_focus=article_focus,
        focus_keywords=focus_keywords,
        settings=settings,
        taxonomy=taxonomy,
    )
    signals = list(clustered.get("signals") or [])
    strong = [
        signal
        for signal in signals
        if int(signal.get("frequency") or 0) >= 2 and str(signal.get("name") or "") != "uncategorized_evidence"
    ]
    if len(strong) < 1:
        return {**clustered, "ok": False, "reason": "not enough repeated signals", "claims": [], "thesis": None}
    claims = []
    claim_cards = []
    for index, signal in enumerate(strong[:6], start=1):
        evidence_ids = list(signal.get("evidence_ids") or [])[:8]
        claim = {
            "id": f"claim_{index}",
            "text": _claim_text(signal, article_focus),
            "signal": signal["name"],
            "evidence_ids": evidence_ids,
            "evidence_refs": list(signal.get("representative_msgs") or [])[:5],
            "confidence": signal.get("confidence") or "medium",
            "caveat": "Corpus is limited to collected Telegram messages.",
        }
        claims.append(claim)
        card = await _upsert_card(
            session,
            {
                "topic_slug": topic_slug,
                "card_type": "claim",
                "title": claim["text"][:160],
                "summary": f"{claim['signal']} evidence={len(evidence_ids)} confidence={claim['confidence']}",
                "body": claim["text"],
                "source_channel": None,
                "source_url": None,
                "score": 0.72 if claim["confidence"] == "high" else 0.62,
                "tags": ["claim", claim["signal"], claim["confidence"]],
                "payload": {
                    "stage": "claim",
                    "stage_order": 40,
                    "artifact": "claim",
                    "article_focus": article_focus,
                    "focus_key": _focus_key(article_focus, focus_keywords),
                    "claim_id": claim["id"],
                    "signal": claim["signal"],
                    "evidence_ids": evidence_ids,
                    "evidence_refs": claim["evidence_refs"],
                    "confidence": claim["confidence"],
                    "caveat": claim["caveat"],
                },
            },
        )
        claim_cards.append(card.id)
    thesis = _thesis_from_claims(topic_slug, claims, article_focus)
    evidence_map = {
        "thesis": thesis,
        "claims": claims,
        "counterpoints": _counterpoints_from_signals(signals),
        "signal_count": len(signals),
        "claim_cards": claim_cards,
    }
    map_card = await _upsert_card(
        session,
        {
            "topic_slug": topic_slug,
            "card_type": "evidence_map",
            "title": f"{topic_slug} evidence map",
            "summary": f"claims={len(claims)} signals={len(signals)}",
            "body": thesis,
            "source_channel": None,
            "source_url": None,
            "score": min(1.0, 0.55 + len(claims) * 0.06),
            "tags": ["evidence_map", topic_slug],
            "payload": {
                "stage": "evidence_map",
                "stage_order": 50,
                "artifact": "evidence_map",
                "article_focus": article_focus,
                "focus_key": _focus_key(article_focus, focus_keywords),
                **evidence_map,
            },
        },
    )
    return {
        "topic": topic_slug,
        "article_focus": article_focus,
        "focus_key": _focus_key(article_focus, focus_keywords),
        "ok": len(claims) >= 1,
        "reviewed": clustered.get("reviewed", 0),
        "usable": clustered.get("usable", 0),
        "rejected": clustered.get("rejected", 0),
        "signals": signals,
        "thesis": thesis,
        "claims": claims,
        "counterpoints": evidence_map["counterpoints"],
        "evidence_map_card": map_card.id,
        "claim_cards": claim_cards,
    }


async def propose_topic_angles(session: AsyncSession, topic_slug: str, limit: int = 80, article_focus: str | None = None, focus_keywords: list[str] | None = None, settings: Settings | None = None) -> dict:
    evidence_map = await _latest_evidence_map(session, topic_slug, _focus_key(article_focus, focus_keywords))
    if not evidence_map:
        evidence_map = await build_topic_evidence_map(session, topic_slug, limit, article_focus=article_focus, focus_keywords=focus_keywords, settings=settings)
    if not evidence_map.get("ok"):
        return {**evidence_map, "angles": []}
    signals = [claim.get("signal") for claim in evidence_map.get("claims", [])]
    thesis = str(evidence_map.get("thesis") or topic_slug)
    angles = [
        {
            "title": thesis,
            "why": "Best supported by repeated signals.",
            "claim_ids": [claim.get("id") for claim in evidence_map.get("claims", [])[:4]],
            "risk": "Medium: keep caveats visible.",
        },
        {
            "title": f"{article_focus or topic_slug}: the hidden cost is in {', '.join(signals[:3])}",
            "why": "Turns clustered pain into a concrete editorial line.",
            "claim_ids": [claim.get("id") for claim in evidence_map.get("claims", [])[:3]],
            "risk": "Low if every section cites evidence.",
        },
    ]
    card = await _upsert_card(
        session,
        {
            "topic_slug": topic_slug,
            "card_type": "angle",
            "title": angles[0]["title"][:160],
            "summary": angles[0]["why"],
            "body": json.dumps(angles, ensure_ascii=False),
            "source_channel": None,
            "source_url": None,
            "score": 0.75,
            "tags": ["angle", topic_slug],
            "payload": {"stage": "angle", "stage_order": 60, "artifact": "angles", "article_focus": article_focus, "focus_key": _focus_key(article_focus, focus_keywords), "angles": angles},
        },
    )
    return {"topic": topic_slug, "article_focus": article_focus, "focus_key": _focus_key(article_focus, focus_keywords), "ok": True, "angles": angles, "angle_card": card.id, "evidence_map": evidence_map}


async def build_topic_outline(session: AsyncSession, topic_slug: str, limit: int = 80, article_focus: str | None = None, focus_keywords: list[str] | None = None, settings: Settings | None = None) -> dict:
    evidence_map = await _latest_evidence_map(session, topic_slug, _focus_key(article_focus, focus_keywords))
    if not evidence_map:
        evidence_map = await build_topic_evidence_map(session, topic_slug, limit, article_focus=article_focus, focus_keywords=focus_keywords, settings=settings)
    if not evidence_map.get("ok"):
        return {**evidence_map, "outline_blocks": 0, "article_id": None}
    angle_data = await propose_topic_angles(session, topic_slug, limit, article_focus=article_focus, focus_keywords=focus_keywords, settings=settings)
    angle = (angle_data.get("angles") or [{}])[0].get("title") or evidence_map.get("thesis") or topic_slug
    claim_cards = await _claim_cards(session, topic_slug, evidence_map.get("claims", []), _focus_key(article_focus, focus_keywords))
    article = ArticleSession(
        topic_slug=topic_slug,
        title=str(angle)[:512],
        audience=f"readers interested in evidence-based Telegram analysis; focus:{_focus_key(article_focus, focus_keywords) or 'topic'}",
        angle=str(angle),
        status="outline",
        source_card_ids=[card.id for card in claim_cards],
    )
    session.add(article)
    await session.flush()
    blocks = _outline_specs(evidence_map)
    claim_card_by_id = {str((card.payload or {}).get("claim_id")): card.id for card in claim_cards}
    for position, block in enumerate(blocks):
        source_card_ids = []
        for claim_id in block.get("claim_ids", []):
            card_id = claim_card_by_id.get(str(claim_id))
            if card_id:
                source_card_ids.append(card_id)
        if not source_card_ids:
            source_card_ids = [card.id for card in claim_cards[:2]]
        session.add(
            ArticleBlock(
                session_id=article.id,
                position=position,
                block_type=block["block_type"],
                title=block["title"],
                text=block["text"],
                status="draft",
                source_card_ids=source_card_ids,
            )
        )
    await session.flush()
    return {
        "topic": topic_slug,
        "article_focus": article_focus,
        "focus_key": _focus_key(article_focus, focus_keywords),
        "ok": True,
        "article_id": article.id,
        "title": article.title,
        "angle": article.angle,
        "outline_blocks": len(blocks),
        "claim_cards": article.source_card_ids,
        "claims": evidence_map.get("claims", []),
    }


async def write_topic_draft(
    session: AsyncSession,
    settings: Settings,
    topic_slug: str,
    limit: int = 80,
    use_llm: bool = False,
    article_focus: str | None = None,
    focus_keywords: list[str] | None = None,
) -> dict:
    article = await _latest_article(session, topic_slug, statuses={"outline", "blocks", "draft"}, focus_key=_focus_key(article_focus, focus_keywords))
    if not article:
        outline = await build_topic_outline(session, topic_slug, limit, article_focus=article_focus, focus_keywords=focus_keywords, settings=settings)
        if not outline.get("ok"):
            return {**outline, "draft_id": None, "draft_chars": 0}
        article = await session.get(ArticleSession, int(outline["article_id"]))
    if not article:
        return {"topic": topic_slug, "ok": False, "reason": "article not found", "draft_id": None}
    if use_llm:
        draft = await generate_draft(session, settings, article.id, "evidence_first", "Use only cited claims and source refs.", True)
        if draft:
            return {"topic": topic_slug, "article_focus": article_focus, "focus_key": _focus_key(article_focus, focus_keywords), "ok": True, "article_id": article.id, "draft_id": draft.id, "draft_chars": len(draft.text), "draft_preview": draft.text[:1200]}
    text = await _evidence_draft_text(session, article)
    draft = ArticleDraft(
        session_id=article.id,
        version=(await session.scalar(select(func.max(ArticleDraft.version)).where(ArticleDraft.session_id == article.id)) or 0) + 1,
        status="draft",
        format="telegram_markdown",
        text=text,
        rich_json=_telegram_rich_json(text),
        score=_draft_score(text),
    )
    session.add(draft)
    article.status = "draft"
    article.updated_at = datetime.utcnow()
    await session.flush()
    return {"topic": topic_slug, "article_focus": article_focus, "focus_key": _focus_key(article_focus, focus_keywords), "ok": True, "article_id": article.id, "draft_id": draft.id, "draft_chars": len(text), "draft_preview": text[:1200]}


async def verify_topic_claims(session: AsyncSession, topic_slug: str, article_id: int | None = None, article_focus: str | None = None, focus_keywords: list[str] | None = None) -> dict:
    focus_key = _focus_key(article_focus, focus_keywords)
    article = await session.get(ArticleSession, article_id) if article_id else await _latest_article(session, topic_slug, statuses={"draft", "outline", "blocks", "final"}, focus_key=focus_key)
    if not article:
        return {"topic": topic_slug, "ok": False, "reason": "article not found", "checks": []}
    if article.topic_slug != topic_slug:
        return {
            "topic": topic_slug,
            "ok": False,
            "reason": "article topic mismatch",
            "article_id": article.id,
            "article_topic": article.topic_slug,
            "checks": [],
        }
    draft = await _latest_draft(session, article.id)
    claim_cards = await _cards_by_ids(session, article.source_card_ids or [])
    checks = []
    draft_text = draft.text if draft else ""
    for card in claim_cards:
        payload = card.payload or {}
        evidence_ids = list(payload.get("evidence_ids") or [])
        evidence_refs = [str(ref) for ref in payload.get("evidence_refs") or []]
        refs_present = any(ref in draft_text for ref in evidence_refs)
        focus_matches = int(sum(1 for ref in evidence_refs if ref in draft_text))
        card_focus_ok = not focus_key or payload.get("focus_key") == focus_key
        checks.append(
            {
                "claim_card_id": card.id,
                "claim": card.title,
                "focus_ok": card_focus_ok,
                "has_evidence": bool(evidence_ids),
                "evidence_count": len(evidence_ids),
                "focus_refs_in_draft": focus_matches,
                "refs_present_in_draft": refs_present,
                "generalization_supported": len(evidence_ids) >= 2,
                "confidence": payload.get("confidence") or "low",
            }
        )
    unsupported = [check for check in checks if not check["has_evidence"]]
    missing_refs = [check for check in checks if not check["refs_present_in_draft"]]
    focus_mismatch = [check for check in checks if not check["focus_ok"]]
    weak = [check for check in checks if not check["generalization_supported"]]
    draft_focus_score, draft_focus_hits = _focus_match_score(draft_text, article_focus, focus_keywords)
    draft_focus_ok = not focus_key or draft_focus_score >= MIN_FOCUS_MATCH
    block_count = await session.scalar(select(func.count(ArticleBlock.id)).where(ArticleBlock.session_id == article.id))
    ok = bool(draft) and bool(checks) and not unsupported and not missing_refs and not focus_mismatch and draft_focus_ok
    return {
        "topic": topic_slug,
        "article_focus": article_focus,
        "focus_key": focus_key,
        "ok": ok,
        "article_id": article.id,
        "draft_id": draft.id if draft else None,
        "checks": checks,
        "unsupported_claims": len(unsupported),
        "missing_draft_refs": len(missing_refs),
        "focus_mismatches": len(focus_mismatch),
        "draft_focus_score": round(draft_focus_score, 3),
        "draft_focus_hits": draft_focus_hits,
        "weak_generalizations": len(weak),
        "outline_blocks": int(block_count or 0),
        "reason": "claims verified" if ok else "draft or supported claims missing",
    }


async def finalize_topic_article(session: AsyncSession, topic_slug: str, article_id: int | None = None, article_focus: str | None = None, focus_keywords: list[str] | None = None) -> dict:
    verification = await verify_topic_claims(session, topic_slug, article_id, article_focus=article_focus, focus_keywords=focus_keywords)
    if not verification.get("ok"):
        return {**verification, "final": False}
    article = await session.get(ArticleSession, int(verification["article_id"]))
    draft = await _latest_draft(session, int(verification["article_id"]))
    if not article or not draft:
        return {**verification, "ok": False, "final": False, "reason": "article or draft not found"}
    draft.status = "final"
    article.status = "final"
    article.updated_at = datetime.utcnow()
    await session.flush()
    return {
        **verification,
        "ok": True,
        "final": True,
        "article_id": article.id,
        "draft_id": draft.id,
        "status": "final",
        "final_chars": len(draft.text),
        "final_preview": draft.text[:1200],
    }


async def generate_cards_from_topic(
    session: AsyncSession,
    topic_slug: str,
    limit: int = 60,
    usable_message_ids: set[int] | None = None,
    include_framework: bool = True,
) -> list[ContentCardOut]:
    topic = await session.scalar(select(ResearchTopic).where(ResearchTopic.slug == topic_slug))
    if not topic:
        return []
    stmt = (
        select(Message, Channel, TopicMessage)
        .join(TopicMessage, TopicMessage.message_id == Message.id)
        .join(Channel, Channel.id == Message.channel_id)
        .where(TopicMessage.topic_id == topic.id, TopicMessage.pain_score >= 0.28)
        .order_by(TopicMessage.pain_score.desc(), TopicMessage.relevance_score.desc(), Message.posted_at.desc().nulls_last())
        .limit(min(max(limit * 4, 20), 1000))
    )
    if usable_message_ids is not None:
        if not usable_message_ids:
            return []
        stmt = stmt.where(Message.id.in_(usable_message_ids))
    rows = (await session.execute(stmt)).all()
    seen_discussions: set[str] = set()
    created: list[ContentCard] = []
    for message, channel, topic_message in rows:
        discussion_key = message.discussion_key or hashlib.sha256((message.text or "").encode("utf-8")).hexdigest()
        if discussion_key in seen_discussions:
            continue
        seen_discussions.add(discussion_key)
        insight = PainInsight(
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
            highlights=[],
            discussion_key=discussion_key,
            thread_id=message.thread_id,
            reply_to_msg_id=message.reply_to_msg_id,
        )
        for spec in _cards_from_insight(topic_slug, insight, topic.keywords or [], topic_message.relevance_score or 0.0):
            card = await _upsert_card(session, spec)
            created.append(card)
        if len(created) >= limit:
            break
    if include_framework and created:
        for spec in _framework_cards(topic_slug, topic, rows):
            created.append(await _upsert_card(session, spec))
    return [card_to_out(row) for row in created[:limit]]


async def create_article(
    session: AsyncSession,
    topic_slug: str,
    title: str | None,
    audience: str | None,
    angle: str | None,
    card_ids: list[int],
) -> ArticleSessionOut:
    if not card_ids:
        rows = (
            await session.execute(
                select(ContentCard.id)
                .where(ContentCard.topic_slug == topic_slug, ContentCard.status.in_(["approved", "saved", "new"]))
                .order_by(ContentCard.status.asc(), ContentCard.score.desc())
                .limit(8)
            )
        ).scalars().all()
        card_ids = list(rows)
    cards = []
    if card_ids:
        cards = (await session.execute(select(ContentCard).where(ContentCard.id.in_(card_ids)))).scalars().all()
    resolved_title = title or (cards[0].title if cards else f"{topic_slug}: new article")
    resolved_angle = angle or _angle_from_cards(cards)
    row = ArticleSession(
        topic_slug=topic_slug,
        title=resolved_title[:512],
        audience=audience,
        angle=resolved_angle,
        status="outline",
        source_card_ids=card_ids,
    )
    session.add(row)
    await session.flush()
    return await article_to_out(session, row)


async def update_card_status(session: AsyncSession, card_id: int, action: str, note: str | None = None) -> ContentCardOut | None:
    card = await session.get(ContentCard, card_id)
    if not card:
        return None
    card.status = {"approve": "approved", "reject": "rejected", "save": "saved", "used": "used", "new": "new"}[action]
    payload = dict(card.payload or {})
    if note:
        payload.setdefault("notes", []).append({"at": datetime.utcnow().isoformat(), "text": note})
    card.payload = payload
    card.updated_at = datetime.utcnow()
    return card_to_out(card)


async def update_article(
    session: AsyncSession,
    article_id: int,
    title: str | None = None,
    audience: str | None = None,
    angle: str | None = None,
    status: str | None = None,
    source_card_ids: list[int] | None = None,
) -> ArticleSessionOut | None:
    article = await session.get(ArticleSession, article_id)
    if not article:
        return None
    if title is not None:
        article.title = title[:512]
    if audience is not None:
        article.audience = audience[:512]
    if angle is not None:
        article.angle = angle
    if status is not None:
        article.status = status[:32]
    if source_card_ids is not None:
        article.source_card_ids = source_card_ids
    article.updated_at = datetime.utcnow()
    return await article_to_out(session, article)


async def update_block(
    session: AsyncSession,
    block_id: int,
    title: str | None = None,
    text: str | None = None,
    block_type: str | None = None,
    status: str | None = None,
    position: int | None = None,
) -> ArticleSessionOut | None:
    block = await session.get(ArticleBlock, block_id)
    if not block:
        return None
    if title is not None:
        block.title = title[:512]
    if text is not None:
        block.text = text
    if block_type is not None:
        block.block_type = block_type[:32]
    if status is not None:
        block.status = status[:32]
    if position is not None:
        block.position = max(position, 0)
    block.updated_at = datetime.utcnow()
    article = await session.get(ArticleSession, block.session_id)
    if article:
        article.status = "blocks_edited"
        article.updated_at = datetime.utcnow()
        return await article_to_out(session, article)
    return None


async def update_draft(
    session: AsyncSession,
    draft_id: int,
    text: str | None = None,
    status: str | None = None,
    format_name: str | None = None,
) -> ArticleSessionOut | None:
    draft = await session.get(ArticleDraft, draft_id)
    if not draft:
        return None
    if text is not None:
        draft.text = _telegram_text(text) if (format_name or draft.format).startswith("telegram") else text
        draft.rich_json = _telegram_rich_json(draft.text)
        draft.score = _draft_score(draft.text)
    if status is not None:
        draft.status = status[:32]
    if format_name is not None:
        draft.format = format_name[:32]
    article = await session.get(ArticleSession, draft.session_id)
    if article:
        article.status = "approved" if draft.status == "approved" else "draft_edited"
        article.updated_at = datetime.utcnow()
        return await article_to_out(session, article)
    return None


async def generate_outline(session: AsyncSession, settings: Settings, article_id: int, style: str, instruction: str | None, use_llm: bool) -> ArticleSessionOut | None:
    article = await session.get(ArticleSession, article_id)
    if not article:
        return None
    cards = await _article_cards(session, article)
    await session.execute(delete(ArticleBlock).where(ArticleBlock.session_id == article.id))
    blocks = await _llm_outline(settings, article, cards, style, instruction) if use_llm else []
    if not blocks:
        blocks = _fallback_outline(article, cards)
    for pos, block in enumerate(blocks):
        session.add(
            ArticleBlock(
                session_id=article.id,
                position=pos,
                block_type=block.get("block_type", "paragraph"),
                title=block.get("title"),
                text=block.get("text", ""),
                source_card_ids=block.get("source_card_ids", article.source_card_ids or []),
            )
        )
    article.status = "blocks"
    article.updated_at = datetime.utcnow()
    await session.flush()
    return await article_to_out(session, article)


async def generate_draft(session: AsyncSession, settings: Settings, article_id: int, style: str, instruction: str | None, use_llm: bool) -> ArticleDraftOut | None:
    article = await session.get(ArticleSession, article_id)
    if not article:
        return None
    blocks = (
        await session.execute(
            select(ArticleBlock).where(ArticleBlock.session_id == article.id).order_by(ArticleBlock.position, ArticleBlock.id)
        )
    ).scalars().all()
    if not blocks:
        article_out = await generate_outline(session, settings, article_id, style, instruction, use_llm)
        if not article_out:
            return None
        blocks = (
            await session.execute(
                select(ArticleBlock).where(ArticleBlock.session_id == article.id).order_by(ArticleBlock.position, ArticleBlock.id)
            )
        ).scalars().all()
    cards = await _article_cards(session, article)
    text = await _llm_draft(settings, article, blocks, cards, style, instruction) if use_llm else ""
    if not text:
        text = _fallback_draft(article, blocks, cards)
    last_version = await session.scalar(select(func.max(ArticleDraft.version)).where(ArticleDraft.session_id == article.id))
    draft = ArticleDraft(
        session_id=article.id,
        version=(last_version or 0) + 1,
        status="draft",
        format="telegram_markdown",
        text=text,
        rich_json=_telegram_rich_json(text),
        score=_draft_score(text),
    )
    session.add(draft)
    article.status = "draft"
    article.updated_at = datetime.utcnow()
    await session.flush()
    return draft_to_out(draft)


async def format_telegram(session: AsyncSession, article_id: int) -> ArticleDraftOut | None:
    draft = (
        await session.execute(
            select(ArticleDraft)
            .where(ArticleDraft.session_id == article_id)
            .order_by(ArticleDraft.version.desc(), ArticleDraft.id.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if not draft:
        return None
    draft.text = _telegram_text(draft.text)
    draft.rich_json = _telegram_rich_json(draft.text)
    draft.format = "telegram_rich_preview"
    draft.status = "ready"
    return draft_to_out(draft)


def _cards_from_insight(topic_slug: str, item: PainInsight, keywords: list[str], relevance: float) -> list[dict]:
    text = normalize_text(item.text)
    title = _title_from_text(text, item.pain_type)
    rage = _rage_score(text)
    novelty = _novelty_score(text, keywords)
    base = min(1.0, item.pain_score * 0.45 + relevance * 0.2 + rage * 0.2 + novelty * 0.15)
    tags = sorted(set([item.pain_type, item.intent, *item.reasons[:4], *[k for k in keywords if k.lower() in text.lower()][:4]]))
    common = {
        "topic_slug": topic_slug,
        "source_channel": item.channel_username,
        "source_url": item.url,
        "tags": tags,
        "payload": {
            "message_id": item.message_id,
            "posted_at": item.posted_at.isoformat() if item.posted_at else None,
            "discussion_key": item.discussion_key,
            "pain_score": item.pain_score,
            "thread_id": item.thread_id,
            "reply_to_msg_id": item.reply_to_msg_id,
        },
    }
    hook = _hook_from_text(text, title)
    conflict = _conflict_from_text(text)
    thesis = _angle_title(text, title)
    evidence = _summary(text, 360)
    counterpoint = _counterpoint_from_text(text)
    takeaway = _takeaway_from_text(text)
    cta = _cta_from_text(text)
    hook_template = _card_template("hook")
    conflict_template = _card_template("conflict")
    thesis_template = _card_template("thesis")
    evidence_template = _card_template("evidence")
    counterpoint_template = _card_template("counterpoint")
    block_template = _card_template("block")
    takeaway_template = _card_template("takeaway")
    cta_template = _card_template("cta")
    return [
        {
            **common,
            "card_type": "topic",
            "title": title,
            "summary": _summary(text, 260),
            "body": text,
            "score": base,
            "payload": _payload(common, "topic", "topic", ["curiosity_gap", "scannability"]),
        },
        {
            **common,
            "card_type": "hook",
            "title": hook,
            "summary": hook_template["summary"],
            "body": hook,
            "score": min(1.0, base + 0.16 + novelty * 0.08),
            "payload": _payload(common, "hook", hook_template["stage"], hook_template["basis"]),
        },
        {
            **common,
            "card_type": "conflict",
            "title": conflict,
            "summary": conflict_template["summary"],
            "body": conflict_template["body_template"].format(title=conflict, evidence=evidence),
            "score": min(1.0, base + 0.14 + rage * 0.12),
            "payload": _payload(common, "conflict", conflict_template["stage"], conflict_template["basis"]),
        },
        {
            **common,
            "card_type": "thesis",
            "title": thesis,
            "summary": _angle_summary(text),
            "body": thesis_template["body_template"].format(title=thesis, summary=_summary(text, 420)),
            "score": min(1.0, base + 0.08 + rage * 0.08),
            "payload": _payload(common, "thesis", thesis_template["stage"], thesis_template["basis"]),
        },
        {
            **common,
            "card_type": "evidence",
            "title": evidence_template["title_template"].format(channel=item.channel_username),
            "summary": _summary(text, 220),
            "body": text,
            "score": min(1.0, base + 0.04),
            "payload": _payload(common, "evidence", evidence_template["stage"], evidence_template["basis"]),
        },
        {
            **common,
            "card_type": "counterpoint",
            "title": counterpoint,
            "summary": counterpoint_template["summary"],
            "body": counterpoint,
            "score": min(1.0, base + 0.03),
            "payload": _payload(common, "counterpoint", counterpoint_template["stage"], counterpoint_template["basis"]),
        },
        {
            **common,
            "card_type": "block",
            "title": block_template["title_template"].format(title=title),
            "summary": _summary(text, 260),
            "body": _block_from_text(text, title),
            "score": min(1.0, base + 0.05),
            "payload": _payload(common, "block", block_template["stage"], block_template["basis"]),
        },
        {
            **common,
            "card_type": "takeaway",
            "title": takeaway,
            "summary": takeaway_template["summary"],
            "body": takeaway,
            "score": min(1.0, base + 0.05),
            "payload": _payload(common, "takeaway", takeaway_template["stage"], takeaway_template["basis"]),
        },
        {
            **common,
            "card_type": "cta",
            "title": cta,
            "summary": cta_template["summary"],
            "body": cta,
            "score": min(1.0, base + 0.02),
            "payload": _payload(common, "cta", cta_template["stage"], cta_template["basis"]),
        },
    ]


def _framework_cards(topic_slug: str, topic: ResearchTopic, rows: list[tuple[Message, Channel, TopicMessage]]) -> list[dict]:
    pain_counts = Counter((message.pain_type or "discussion") for message, _, _ in rows)
    channels = Counter(channel.username for _, channel, _ in rows)
    main_pain = pain_counts.most_common(1)[0][0] if pain_counts else "discussion"
    top_channels = ", ".join(f"@{name}" for name, _ in channels.most_common(5))
    keyword_hint = ", ".join((topic.keywords or [])[:5])
    common = {
        "topic_slug": topic_slug,
        "source_channel": None,
        "source_url": None,
        "tags": ["framework", main_pain, *(topic.keywords or [])[:3]],
        "payload": {"topic_id": topic.id, "keywords": topic.keywords or [], "main_pain": main_pain, "top_channels": top_channels},
    }
    top_channels_or_fallback = top_channels or CONTENT_RULES["framework_fallbacks"]["top_channels"]
    return [
        {
            **common,
            "card_type": spec["card_type"],
            "title": spec["title"],
            "summary": spec["summary"],
            "body": spec.get("body") or spec["body_template"].format(
                keyword_hint=keyword_hint,
                top_channels_or_fallback=top_channels_or_fallback,
            ),
            "score": spec["score"],
            "payload": _payload(common, spec["card_type"], spec["card_type"], spec["basis"]),
        }
        for spec in CONTENT_RULES["framework_cards"]
    ]


async def _upsert_card(session: AsyncSession, spec: dict) -> ContentCard:
    payload = spec.get("payload") or {}
    fingerprint = hashlib.sha256(
        "|".join(
            [
                spec["topic_slug"],
                spec["card_type"],
                str(payload.get("focus_key") or ""),
                str(payload.get("artifact") or ""),
                str(payload.get("claim_id") or ""),
                str(payload.get("signal") or ""),
                spec.get("source_url") or "",
                spec["title"],
            ]
        ).encode("utf-8")
    ).hexdigest()
    row = await session.scalar(select(ContentCard).where(ContentCard.fingerprint == fingerprint))
    if row:
        row.score = max(row.score or 0.0, spec["score"])
        row.summary = spec["summary"]
        row.body = spec["body"]
        row.payload = {**(row.payload or {}), **(spec.get("payload") or {})}
        row.tags = sorted(set((row.tags or []) + (spec.get("tags") or [])))
        row.updated_at = datetime.utcnow()
        return row
    row = ContentCard(fingerprint=fingerprint, status="new", **spec)
    session.add(row)
    await session.flush()
    return row


async def _article_cards(session: AsyncSession, article: ArticleSession) -> list[ContentCard]:
    if article.source_card_ids:
        return (await session.execute(select(ContentCard).where(ContentCard.id.in_(article.source_card_ids)))).scalars().all()
    return (
        await session.execute(
            select(ContentCard)
            .where(ContentCard.topic_slug == article.topic_slug, ContentCard.status.in_(["approved", "saved"]))
            .order_by(ContentCard.score.desc())
            .limit(8)
        )
    ).scalars().all()


def _content_prompt(settings: Settings, name: str) -> str:
    return PromptLoader(override_dir=settings.content_prompt_dir).load(name)


async def _llm_outline(settings: Settings, article: ArticleSession, cards: list[ContentCard], style: str, instruction: str | None) -> list[dict]:
    prompt = {
        "task": _content_prompt(settings, settings.content_outline_task_prompt),
        "article": {"title": article.title, "audience": article.audience, "angle": article.angle, "style": style, "instruction": instruction},
        "cards": [_card_prompt(card) for card in cards],
        "schema": [{"block_type": "hook|thesis|evidence|analysis|counterpoint|takeaway|cta", "title": "short", "text": "draft block", "source_card_ids": [1]}],
    }
    data = await _llm_json(settings, prompt)
    return data if isinstance(data, list) else []


async def _llm_draft(
    settings: Settings,
    article: ArticleSession,
    blocks: list[ArticleBlock],
    cards: list[ContentCard],
    style: str,
    instruction: str | None,
) -> str:
    prompt = {
        "task": _content_prompt(settings, settings.content_draft_task_prompt),
        "article": {"title": article.title, "audience": article.audience, "angle": article.angle, "style": style, "instruction": instruction},
        "blocks": [{"type": b.block_type, "title": b.title, "text": b.text} for b in blocks],
        "evidence": [_card_prompt(card) for card in cards],
        "rules": [
            "Strong first line",
            "Concrete conflict",
            "Mention uncertainty if claim is speculative",
            "End with discussion question",
            "No sales pitch",
        ],
    }
    data = await _llm_text(settings, prompt)
    return data.strip()


async def _llm_json(settings: Settings, payload: dict) -> object | None:
    text = await _llm_text(settings, payload)
    if not text:
        return None
    text = _strip_code_fence(text)
    for candidate in _json_candidates(text):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _json_candidates(text: str) -> list[str]:
    candidates = [text.strip()]
    for left, right in (("[", "]"), ("{", "}")):
        start = text.find(left)
        end = text.rfind(right)
        if start >= 0 and end > start:
            candidate = text[start : end + 1].strip()
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def _strip_code_fence(value: str) -> str:
    text = value.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


async def _llm_text(settings: Settings, payload: dict) -> str:
    if not settings.llm_base_url or not settings.llm_api_key:
        return ""
    url = settings.llm_base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": _content_prompt(settings, settings.content_system_prompt)},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "temperature": settings.content_llm_temperature,
        "max_tokens": settings.content_llm_max_tokens,
    }
    try:
        async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
            response = await client.post(url, headers={"authorization": f"Bearer {settings.llm_api_key}"}, json=body)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"] or ""
    except Exception:
        return ""


def _fallback_outline(article: ArticleSession, cards: list[ContentCard]) -> list[dict]:
    ordered = sorted(cards, key=lambda card: ((card.payload or {}).get("stage_order") or STAGE_ORDER.get(card.card_type, 999), -(card.score or 0.0)))
    by_type: dict[str, list[ContentCard]] = {}
    for card in ordered:
        by_type.setdefault(card.card_type, []).append(card)
    evidence_cards = by_type.get("evidence", [])[:4] or ordered[:4]
    all_ids = [card.id for card in ordered[:10]]
    blocks = []
    fallback_values = {
        "article_angle_or_title": article.angle or article.title,
        "angle_from_cards": _angle_from_cards(ordered),
    }
    for card_type, fallback_title, fallback_text in CONTENT_RULES["fallback_outline"]["head"]:
        card = (by_type.get(card_type) or [None])[0]
        blocks.append(
            {
                "block_type": card_type,
                "title": card.title if card else fallback_title,
                "text": card.body if card else fallback_text.format(**fallback_values),
                "source_card_ids": [card.id] if card else all_ids[:3],
            }
        )
    blocks.extend(
        {
            "block_type": "evidence",
            "title": card.title,
            "text": card.body,
            "source_card_ids": [card.id],
        }
        for card in evidence_cards
    )
    for card_type, fallback_title, fallback_text in CONTENT_RULES["fallback_outline"]["tail"]:
        card = (by_type.get(card_type) or [None])[0]
        blocks.append(
            {
                "block_type": card_type,
                "title": card.title if card else fallback_title,
                "text": card.body if card else fallback_text,
                "source_card_ids": [card.id] if card else all_ids,
            }
        )
    return blocks


def _fallback_draft(article: ArticleSession, blocks: list[ArticleBlock], cards: list[ContentCard]) -> str:
    hook = blocks[0].text if blocks else article.title
    def evidence_line(card: ContentCard) -> str:
        source = f"@{card.source_channel}: " if card.source_channel else ""
        return f"- {source}{card.summary}"

    evidence = "\n".join(evidence_line(card) for card in cards[:5])
    draft_rules = CONTENT_RULES["fallback_draft"]
    parts = [
        f"**{hook}**",
        "",
        article.angle or _angle_from_cards(cards),
        "",
        draft_rules["discussion_label"],
        evidence,
        "",
        draft_rules["conclusion"],
        "",
        draft_rules["question"],
    ]
    return "\n".join(part for part in parts if part is not None)


def _telegram_text(text: str) -> str:
    clean = (text or "").replace("\\n", "\n").replace("\r\n", "\n").replace("\r", "\n")
    clean = "\n".join(normalize_text(line) for line in clean.split("\n"))
    clean = _collapse_blank_lines(clean)
    return clean[:4096]


def _collapse_blank_lines(text: str) -> str:
    lines = text.split("\n")
    out: list[str] = []
    blank = 0
    for line in lines:
        if line:
            blank = 0
            out.append(line)
            continue
        blank += 1
        if blank <= 2:
            out.append(line)
    return "\n".join(out)


def _telegram_rich_json(text: str) -> dict:
    clean = _telegram_text(text)
    blocks = [part.strip() for part in clean.split("\n\n") if part.strip()]
    return {
        "parse_mode": "Markdown",
        "text": clean,
        "blocks": [{"type": "paragraph", "text": block} for block in blocks],
        "length": len(clean),
    }


def _draft_score(text: str) -> float:
    score = 0.25
    if len(text) >= 900:
        score += 0.25
    if "?" in text:
        score += 0.15
    lowered = text.lower()
    if any(phrase in lowered for phrase in CONTENT_RULES["draft_score_phrases"]):
        score += 0.2
    if any(_looks_like_handle(token) for token in text.split()):
        score += 0.15
    return min(score, 1.0)


def _looks_like_handle(token: str) -> bool:
    value = token.strip(".,:;!?()[]{}<>")
    if not value.startswith("@") or not 6 <= len(value) <= 33:
        return False
    return all(char.isascii() and (char.isalnum() or char == "_") for char in value[1:])


def _signal_summary(name: str, items: list[dict]) -> str:
    refs = ", ".join(str(item.get("message_id")) for item in items[:3])
    return f"{name}: {len(items)} representative messages" + (f" ({refs})" if refs else "")


def _claim_text(signal: dict, article_focus: str | None = None) -> str:
    name = str(signal.get("name") or "signal")
    frequency = int(signal.get("frequency") or 0)
    confidence = str(signal.get("confidence") or "low")
    if article_focus:
        return f"For article focus '{article_focus}', Telegram evidence repeats the {name} signal across {frequency} usable messages; confidence is {confidence}."
    return f"Telegram evidence repeats the {name} signal across {frequency} usable messages; confidence is {confidence}."


def _thesis_from_claims(topic_slug: str, claims: list[dict], article_focus: str | None = None) -> str:
    signals = [str(claim.get("signal")) for claim in claims[:3] if claim.get("signal")]
    if article_focus and signals:
        return f"{article_focus}: Telegram evidence supports an angle around {', '.join(signals)}."
    if article_focus:
        return f"{article_focus}: the article angle must stay constrained by collected Telegram evidence."
    if signals:
        return f"{topic_slug}: the strongest angle is not generic advice, but repeated friction around {', '.join(signals)}."
    return f"{topic_slug}: the article angle must stay constrained by collected Telegram evidence."


def _counterpoints_from_signals(signals: list[dict]) -> list[dict]:
    weak = [signal for signal in signals if signal.get("confidence") == "low"]
    return [
        {
            "text": f"The {signal.get('name')} signal is weak and should be framed as a hypothesis.",
            "signal": signal.get("name"),
            "evidence_ids": list(signal.get("evidence_ids") or [])[:3],
        }
        for signal in weak[:3]
    ]


async def _latest_evidence_map(session: AsyncSession, topic_slug: str, focus_key: str | None = None) -> dict | None:
    rows = (
        await session.execute(
        select(ContentCard)
        .where(ContentCard.topic_slug == topic_slug, ContentCard.card_type == "evidence_map")
        .order_by(ContentCard.updated_at.desc(), ContentCard.id.desc())
        .limit(20)
        )
    ).scalars().all()
    card = None
    for row in rows:
        payload = row.payload or {}
        if focus_key and payload.get("focus_key") != focus_key:
            continue
        if not focus_key and payload.get("focus_key"):
            continue
        card = row
        break
    if not card:
        return None
    payload = dict(card.payload or {})
    return {
        "topic": topic_slug,
        "ok": bool(payload.get("claims")),
        "thesis": payload.get("thesis"),
        "claims": payload.get("claims") or [],
        "counterpoints": payload.get("counterpoints") or [],
        "article_focus": payload.get("article_focus"),
        "focus_key": payload.get("focus_key"),
        "evidence_map_card": card.id,
        "claim_cards": payload.get("claim_cards") or [],
    }


async def _claim_cards(session: AsyncSession, topic_slug: str, claims: list[dict], focus_key: str | None = None) -> list[ContentCard]:
    claim_ids = [str(claim.get("id")) for claim in claims if claim.get("id")]
    if not claim_ids:
        return []
    rows = (
        await session.execute(
            select(ContentCard)
            .where(ContentCard.topic_slug == topic_slug, ContentCard.card_type == "claim")
            .order_by(ContentCard.updated_at.desc(), ContentCard.id.desc())
        )
    ).scalars().all()
    by_claim_id: dict[str, ContentCard] = {}
    for row in rows:
        if focus_key and (row.payload or {}).get("focus_key") != focus_key:
            continue
        by_claim_id.setdefault(str((row.payload or {}).get("claim_id")), row)
    return [by_claim_id[item] for item in claim_ids if item in by_claim_id]


def _outline_specs(evidence_map: dict) -> list[dict]:
    claims = list(evidence_map.get("claims") or [])
    thesis = str(evidence_map.get("thesis") or "")
    blocks = [
        {
            "block_type": "hook",
            "title": "Hook",
            "text": thesis,
            "claim_ids": [claim.get("id") for claim in claims[:2]],
        },
        {
            "block_type": "thesis",
            "title": "Thesis",
            "text": thesis,
            "claim_ids": [claim.get("id") for claim in claims[:3]],
        },
    ]
    for claim in claims:
        refs = ", ".join(str(ref) for ref in claim.get("evidence_refs", [])[:4])
        blocks.append(
            {
                "block_type": "evidence",
                "title": str(claim.get("signal") or claim.get("id") or "claim"),
                "text": f"{claim.get('text')} Evidence: {refs}. Caveat: {claim.get('caveat')}",
                "claim_ids": [claim.get("id")],
            }
        )
    counterpoints = list(evidence_map.get("counterpoints") or [])
    if counterpoints:
        blocks.append(
            {
                "block_type": "counterpoint",
                "title": "Counterpoint",
                "text": " ".join(str(item.get("text")) for item in counterpoints[:2]),
                "claim_ids": [claim.get("id") for claim in claims[:2]],
            }
        )
    blocks.append(
        {
            "block_type": "takeaway",
            "title": "Takeaway",
            "text": "Keep only claims that have message-level evidence; downgrade weak claims to hypotheses.",
            "claim_ids": [claim.get("id") for claim in claims[:3]],
        }
    )
    return blocks


async def _latest_article(session: AsyncSession, topic_slug: str, statuses: set[str] | None = None, focus_key: str | None = None) -> ArticleSession | None:
    stmt = select(ArticleSession).where(ArticleSession.topic_slug == topic_slug)
    if statuses:
        stmt = stmt.where(ArticleSession.status.in_(sorted(statuses)))
    rows = (await session.execute(stmt.order_by(ArticleSession.updated_at.desc(), ArticleSession.id.desc()).limit(30))).scalars().all()
    for row in rows:
        marker = f"focus:{focus_key or 'topic'}"
        if focus_key and marker not in (row.audience or ""):
            continue
        if not focus_key and "focus:" in (row.audience or "") and "focus:topic" not in (row.audience or ""):
            continue
        return row
    return None


async def _latest_draft(session: AsyncSession, article_id: int) -> ArticleDraft | None:
    return await session.scalar(
        select(ArticleDraft)
        .where(ArticleDraft.session_id == article_id)
        .order_by(ArticleDraft.version.desc(), ArticleDraft.id.desc())
        .limit(1)
    )


async def _cards_by_ids(session: AsyncSession, card_ids: list[int]) -> list[ContentCard]:
    if not card_ids:
        return []
    return (await session.execute(select(ContentCard).where(ContentCard.id.in_(card_ids)))).scalars().all()


async def _evidence_draft_text(session: AsyncSession, article: ArticleSession) -> str:
    blocks = (
        await session.execute(
            select(ArticleBlock)
            .where(ArticleBlock.session_id == article.id)
            .order_by(ArticleBlock.position, ArticleBlock.id)
        )
    ).scalars().all()
    cards = {card.id: card for card in await _cards_by_ids(session, article.source_card_ids or [])}
    parts = [f"**{article.title}**", "", article.angle or article.title, ""]
    for block in blocks:
        parts.extend([f"### {block.title or block.block_type}", block.text])
        refs = []
        for card_id in block.source_card_ids or []:
            card = cards.get(card_id)
            payload = card.payload if card else {}
            refs.extend(str(ref) for ref in (payload or {}).get("evidence_refs", [])[:4])
        if refs:
            parts.append("Sources: " + ", ".join(sorted(set(refs))))
        parts.append("")
    ledger_refs = []
    for card in cards.values():
        payload = card.payload or {}
        if payload.get("artifact") != "claim":
            continue
        ledger_refs.extend(str(ref) for ref in (payload.get("evidence_refs") or [])[:4])
    if ledger_refs:
        parts.append("Evidence ledger: " + ", ".join(sorted(set(ledger_refs))))
        parts.append("")
    parts.append("Claims without sources should remain hypotheses, not facts.")
    return "\n".join(part for part in parts if part is not None)


def _card_prompt(card: ContentCard) -> dict:
    return {
        "id": card.id,
        "type": card.card_type,
        "title": card.title,
        "summary": card.summary,
        "source": f"@{card.source_channel}" if card.source_channel else None,
        "url": card.source_url,
        "score": card.score,
        "tags": card.tags or [],
        "stage": (card.payload or {}).get("stage"),
        "evidence_refs": (card.payload or {}).get("evidence_refs", []),
        "caveat": (card.payload or {}).get("caveat"),
        "research_basis": (card.payload or {}).get("research_basis", []),
    }


def _payload(common: dict, card_type: str, stage: str, basis_keys: list[str]) -> dict:
    payload = dict(common.get("payload") or {})
    payload.update(
        {
            "stage": stage,
            "stage_order": STAGE_ORDER.get(card_type, 999),
            "research_basis": [RESEARCH_BASIS[key] for key in basis_keys if key in RESEARCH_BASIS],
            "basis_keys": basis_keys,
        }
    )
    return payload


def _card_template(card_type: str) -> dict:
    return CONTENT_RULES["card_templates"][card_type]


def _matching_rule(text: str, rules: list[dict]) -> dict | None:
    lowered = text.lower()
    for rule in rules:
        if any(token in lowered for token in rule["tokens"]):
            return rule
    return None


def _hook_from_text(text: str, fallback: str) -> str:
    rule = _matching_rule(text, CONTENT_RULES["hook_rules"])
    if rule:
        return rule["text"]
    if "?" in text:
        return CONTENT_RULES["hook_question_template"].format(fallback=fallback)[:160]
    return CONTENT_RULES["hook_default_template"].format(fallback=fallback.lower())[:160]


def _conflict_from_text(text: str) -> str:
    rule = _matching_rule(text, CONTENT_RULES["conflict_rules"])
    return rule["text"] if rule else CONTENT_RULES["conflict_default"]


def _counterpoint_from_text(text: str) -> str:
    rule = _matching_rule(text, CONTENT_RULES["counterpoint_rules"])
    return rule["text"] if rule else CONTENT_RULES["counterpoint_default"]


def _block_from_text(text: str, title: str) -> str:
    return CONTENT_RULES["block_template"].format(title=title, summary=_summary(text, 380))


def _takeaway_from_text(text: str) -> str:
    rule = _matching_rule(text, CONTENT_RULES["takeaway_rules"])
    return rule["text"] if rule else CONTENT_RULES["takeaway_default"]


def _cta_from_text(text: str) -> str:
    rule = _matching_rule(text, CONTENT_RULES["cta_rules"])
    return rule["text"] if rule else CONTENT_RULES["cta_default"]


def _title_from_text(text: str, pain_type: str) -> str:
    sentence = _first_sentence(text)
    sentence = sentence.strip(" -—")
    if 18 <= len(sentence) <= 110:
        return sentence
    words = _word_tokens(text)[:12]
    prefixes = CONTENT_RULES["title_prefixes"]
    prefix = prefixes.get(pain_type, prefixes["default"])
    return f"{prefix}: {' '.join(words)}"[:140]


def _first_sentence(text: str) -> str:
    for index, char in enumerate(text or ""):
        if char in {".", "!", "?"}:
            return text[: index + 1]
    return text or ""


def _word_tokens(text: str) -> list[str]:
    out: list[str] = []
    current: list[str] = []
    extra = {"#", "+", ".", "-"}
    for char in text:
        if char.isalnum() or char in extra:
            current.append(char)
            continue
        if current:
            out.append("".join(current))
            current = []
    if current:
        out.append("".join(current))
    return out


def _angle_title(text: str, fallback: str) -> str:
    rule = _matching_rule(text, CONTENT_RULES["angle_title_rules"])
    if rule:
        return rule["text"]
    if "?" in text:
        return CONTENT_RULES["angle_question_template"].format(fallback=fallback)[:160]
    return CONTENT_RULES["angle_default_template"].format(fallback=fallback)[:160]


def _angle_summary(text: str) -> str:
    return CONTENT_RULES["angle_summary_template"].format(summary=_summary(text, 220))


def _angle_from_cards(cards: list[ContentCard]) -> str:
    if not cards:
        return CONTENT_RULES["empty_angle"]
    words = Counter()
    stop_words = set(CONTENT_RULES["angle_stop_words"])
    for card in cards:
        for word in _word_tokens((card.title + " " + card.summary).lower()):
            if len(word) < 4:
                continue
            if word not in stop_words:
                words[word] += 1
    common = ", ".join(word for word, _ in words.most_common(5))
    return CONTENT_RULES["angle_from_cards_template"].format(common=common)


def _summary(text: str, limit: int) -> str:
    clean = normalize_text(text)
    return clean if len(clean) <= limit else clean[: limit - 1].rstrip() + "…"


def _rage_score(text: str) -> float:
    lowered = text.lower()
    hits = sum(1 for token in CONTENT_RULES["rage_tokens"] if token in lowered)
    return min(1.0, hits / 4)


def _novelty_score(text: str, keywords: list[str]) -> float:
    lowered = text.lower()
    matched = sum(1 for keyword in keywords if keyword.lower() in lowered)
    rare = sum(1 for token in CONTENT_RULES["novelty_rare_tokens"] if token in lowered)
    return min(1.0, 0.12 * matched + 0.18 * rare)
