from __future__ import annotations

from dataclasses import dataclass
import hashlib

from tg_radar.matching import contains_any, strip_urls_and_handles, word_tokens
from tg_radar.rule_loader import load_rules
from tg_radar.text import normalize_text


TOPICAL_RULES = load_rules("topical_rules.json")


@dataclass(frozen=True)
class PainSignal:
    score: float
    pain_type: str
    intent: str
    reasons: list[str]
    highlights: list[str]
    discussion_key: str


@dataclass(frozen=True)
class TopicGate:
    min_messages: int = 10
    min_quality_score: float = 0.55
    min_match_rate: float = 0.55
    min_pain_score: float = 0.45


@dataclass(frozen=True)
class TopicAdmission:
    accepted: bool
    reason: str
    match_rate: float
    avg_pain_score: float


def topic_admission(gate: TopicGate, seen: int, matched_pain_scores: list[float], quality_score: float) -> TopicAdmission:
    """Decide whether a freshly crawled channel joins the topic pool.

    `matched_pain_scores` holds the pain score of every crawled message that the
    topic would link, so the two ratios come from the same rule the linking uses.
    """
    match_rate = len(matched_pain_scores) / seen if seen else 0.0
    avg_pain_score = sum(matched_pain_scores) / len(matched_pain_scores) if matched_pain_scores else 0.0
    checks = [
        (seen >= gate.min_messages, f"topic_gate_sample:{seen}<{gate.min_messages}"),
        (quality_score >= gate.min_quality_score, f"topic_gate_quality:{quality_score:.2f}<{gate.min_quality_score:.2f}"),
        (match_rate >= gate.min_match_rate, f"topic_gate_match_rate:{match_rate:.2f}<{gate.min_match_rate:.2f}"),
        (avg_pain_score >= gate.min_pain_score, f"topic_gate_pain:{avg_pain_score:.2f}<{gate.min_pain_score:.2f}"),
    ]
    for passed, reason in checks:
        if not passed:
            return TopicAdmission(accepted=False, reason=reason, match_rate=match_rate, avg_pain_score=avg_pain_score)
    return TopicAdmission(
        accepted=True,
        reason=f"topic_gate_pass:rate={match_rate:.2f},pain={avg_pain_score:.2f},quality={quality_score:.2f}",
        match_rate=match_rate,
        avg_pain_score=avg_pain_score,
    )


def classify_pain(text: str, links: list[str] | None = None, mentions: list[str] | None = None) -> PainSignal:
    normalized = normalize_text(" ".join([text or "", *(links or []), *(mentions or [])]))
    reasons: list[str] = []
    score = 0.0
    matched_types: list[str] = []
    for name, terms in TOPICAL_RULES["pain_terms"].items():
        if contains_any(normalized, terms):
            matched_types.append(name)
            reasons.append(f"pain_{name}")
            score += 0.18
    for name, terms in TOPICAL_RULES["context_terms"].items():
        if contains_any(normalized, terms):
            reasons.append(f"context_{name}")
            score += 0.12
    if len(normalized) >= 180:
        score += 0.08
        reasons.append("long_context")
    if "?" in normalized and len(normalized) >= 40:
        score += 0.08
        reasons.append("question_shape")
    if contains_any(normalized, TOPICAL_RULES["first_person_terms"]):
        score += 0.06
        reasons.append("first_person")
    pain_type = matched_types[0] if matched_types else "discussion"
    intent = _intent_from_types(matched_types)
    highlights = _snippets(normalized, [reason.removeprefix("pain_") for reason in reasons] + TOPICAL_RULES["highlight_terms"])
    return PainSignal(
        score=min(score, 1.0),
        pain_type=pain_type,
        intent=intent,
        reasons=sorted(set(reasons)),
        highlights=highlights,
        discussion_key=discussion_key(normalized),
    )


def discussion_key(text: str) -> str:
    normalized = strip_urls_and_handles(text.lower())
    tokens = word_tokens(normalized, min_length=4)[:80]
    return hashlib.sha256(" ".join(tokens).encode("utf-8")).hexdigest()


def topic_relevance(text: str, keywords: list[str], negative_keywords: list[str] | None = None) -> float:
    normalized = normalize_text(text or "").lower()
    if any(token and token.lower() in normalized for token in (negative_keywords or [])):
        return 0.0
    score = 0.0
    for keyword in keywords:
        key = normalize_text(keyword).lower()
        if not key:
            continue
        if key in normalized:
            score += 0.22
            continue
        words = word_tokens(key, min_length=3)
        if words:
            matched = sum(1 for word in words if word in normalized)
            score += 0.16 * (matched / len(words))
    return min(score, 1.0)


def _intent_from_types(types: list[str]) -> str:
    if "question" in types:
        return "needs_answer"
    if "complaint" in types:
        return "pain"
    if "workaround" in types:
        return "workaround"
    if "pricing" in types:
        return "budget"
    if "recommendation" in types:
        return "choice"
    if "comparison" in types:
        return "comparison"
    return "discussion"


def _snippets(text: str, needles: list[str], radius: int = 90) -> list[str]:
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
