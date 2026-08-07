from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib

from tg_radar.matching import contains_any, has_email, has_handle_or_tme, matched_groups, strip_urls_and_handles
from tg_radar.rule_loader import load_rules
from tg_radar.schemas import ParsedMessage
from tg_radar.text import normalize_text


SCORING_RULES = load_rules("scoring_rules.json")
SOURCE_TRUST = SCORING_RULES["source_trust"]


@dataclass(frozen=True)
class HiringSignal:
    score: float
    company_match: bool
    ai_llm_match: bool
    hiring_match: bool
    direct_contact: bool
    companies: list[str]
    reasons: list[str]
    highlights: list[str]


@dataclass(frozen=True)
class ChannelQuality:
    score: float
    public_chat: bool
    recent_jobs: bool
    company_match: bool
    ai_llm_match: bool
    direct_contact: bool
    source_trust: float
    reasons: list[str]


def source_trust(source: str | None) -> float:
    value = (source or "").lower()
    for prefix, score in SOURCE_TRUST.items():
        if value == prefix or value.startswith(f"{prefix}:") or value.startswith(prefix):
            return score
    return 0.55


def hiring_signal(text: str) -> HiringSignal:
    normalized = normalize_text(text or "")
    companies = matched_groups(normalized, SCORING_RULES["company_terms"])
    company_match = bool(companies)
    ai_llm_match = contains_any(normalized, SCORING_RULES["ai_terms"])
    hiring_match = contains_any(normalized, SCORING_RULES["hiring_terms"])
    direct_contact = (
        contains_any(normalized, SCORING_RULES["contact_terms"])
        or has_handle_or_tme(normalized)
        or has_email(normalized)
    )
    score = 0.0
    reasons: list[str] = []
    if company_match:
        score += 0.28
        reasons.append("company_match:" + ",".join(companies))
    if ai_llm_match:
        score += 0.24
        reasons.append("ai_llm_match")
    if hiring_match:
        score += 0.24
        reasons.append("hiring_match")
    if direct_contact:
        score += 0.14
        reasons.append("direct_contact")
    if company_match and ai_llm_match and hiring_match:
        score += 0.1
        reasons.append("target_hiring_detector")
    highlights = _snippets(normalized, [*companies, *SCORING_RULES["highlight_terms"]])
    return HiringSignal(
        score=min(score, 1.0),
        company_match=company_match,
        ai_llm_match=ai_llm_match,
        hiring_match=hiring_match,
        direct_contact=direct_contact,
        companies=companies,
        reasons=reasons,
        highlights=highlights,
    )


def channel_quality(messages: list[ParsedMessage], source: str | None, public_chat: bool = True) -> ChannelQuality:
    now = datetime.now(timezone.utc)
    source_score = source_trust(source)
    recent_jobs = False
    company_match = False
    ai_llm_match = False
    direct_contact = False
    best_signal = 0.0
    for message in messages:
        posted_at = message.posted_at
        if posted_at and not posted_at.tzinfo:
            posted_at = posted_at.replace(tzinfo=timezone.utc)
        recent = not posted_at or posted_at >= now - timedelta(days=90)
        signal = hiring_signal(" ".join([message.text, *message.links, *(message.mentions or [])]))
        if recent and signal.hiring_match:
            recent_jobs = True
        company_match = company_match or signal.company_match
        ai_llm_match = ai_llm_match or signal.ai_llm_match
        direct_contact = direct_contact or signal.direct_contact
        best_signal = max(best_signal, signal.score)
    score = 0.0
    reasons: list[str] = []
    if public_chat:
        score += 0.25
        reasons.append("public_chat")
    if recent_jobs:
        score += 0.22
        reasons.append("recent_jobs")
    if company_match:
        score += 0.18
        reasons.append("company_match")
    if ai_llm_match:
        score += 0.16
        reasons.append("ai_llm_match")
    if direct_contact:
        score += 0.09
        reasons.append("direct_contact")
    score += 0.1 * source_score
    if source_score >= 0.8:
        reasons.append("source_trust")
    if best_signal >= 0.7:
        reasons.append("target_hiring_detector")
    return ChannelQuality(
        score=min(score, 1.0),
        public_chat=public_chat,
        recent_jobs=recent_jobs,
        company_match=company_match,
        ai_llm_match=ai_llm_match,
        direct_contact=direct_contact,
        source_trust=source_score,
        reasons=reasons,
    )


def vacancy_key(text: str, url: str | None = None) -> str:
    normalized = normalize_text(strip_urls_and_handles(text or "")).lower()
    if len(normalized) < 40 and url:
        normalized = url.lower().split("?")[0]
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _snippets(text: str, needles: list[str], radius: int = 72) -> list[str]:
    out: list[str] = []
    lowered = text.lower()
    for needle in needles:
        needle = needle.lower()
        if not needle:
            continue
        idx = lowered.find(needle)
        if idx < 0:
            continue
        start = max(idx - radius, 0)
        end = min(idx + len(needle) + radius, len(text))
        snippet = text[start:end].strip()
        if snippet and snippet not in out:
            out.append(snippet)
        if len(out) >= 3:
            break
    return out
