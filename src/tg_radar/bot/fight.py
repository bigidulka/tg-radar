from __future__ import annotations

import asyncio
from html import escape
import json
from random import SystemRandom
from typing import Any

from tg_radar.agent_runtime.openai_transport import ChatCompletionsTransport, OpenAITransport, OpenAITransportConfig
from tg_radar.agent_runtime.prompt_loader import PromptLoader
from tg_radar.bot.store import BotFightRun, BotRun
from tg_radar.bot.texts import bot_texts, text
from tg_radar.config import Settings


AIRBOT_FIGHT_SCHEMA_VERSION = 1


class AirbotFightError(RuntimeError):
    pass


class AirbotFightArena:
    def __init__(
        self,
        settings: Settings,
        transport: OpenAITransport | None = None,
        prompt_loader: PromptLoader | None = None,
    ) -> None:
        self.settings = settings
        self.transport = transport
        self.prompt_loader = prompt_loader or PromptLoader(override_dir=settings.agent_prompt_dir)
        self._llm_semaphore = asyncio.Semaphore(settings.bot_fight_global_concurrency)
        self._random = SystemRandom()

    async def fight(self, left: BotRun, right: BotRun, topic: str) -> dict[str, Any]:
        if not left.result or not right.result:
            raise AirbotFightError("missing fighter result")
        clean_topic = self._topic(topic)
        if not clean_topic:
            raise AirbotFightError("missing fight topic")
        payload = {
            "schema_version": AIRBOT_FIGHT_SCHEMA_VERSION,
            "topic": clean_topic,
            "rules": self._rules(),
            "fighters": [
                self._fighter_payload("left", left),
                self._fighter_payload("right", right),
            ],
        }
        content = await self._complete_json(
            [
                {"role": "system", "content": self.prompt_loader.load("airbot_fight.md")},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
        )
        return self._normalize(self._json_object(content), left, right, clean_topic, payload)

    async def _complete_json(self, messages: list[dict[str, str]]) -> str:
        transport = self.transport or self._build_transport(self.settings)
        try:
            async with self._llm_semaphore:
                return await transport.complete_json(messages)
        except Exception as exc:
            raise AirbotFightError(f"llm request failed: {exc}") from exc

    def _build_transport(self, settings: Settings) -> OpenAITransport:
        if not settings.llm_base_url or not settings.llm_api_key:
            raise AirbotFightError("llm is not configured")
        return ChatCompletionsTransport(
            OpenAITransportConfig(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
                model=settings.llm_model,
                timeout_seconds=settings.llm_timeout_seconds,
                temperature=settings.agent_model_temperature,
                max_tokens=settings.bot_fight_max_tokens,
                json_response_format=False,
            )
        )

    def _fighter_payload(self, side: str, run: BotRun) -> dict[str, Any]:
        result = run.result or {}
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        tactics = result.get("tactics") if isinstance(result.get("tactics"), list) else []
        evidence = result.get("evidence") if isinstance(result.get("evidence"), list) else []
        detail_score = self._detail_score(metrics, tactics)
        return {
            "side": side,
            "channel": run.channel,
            "air_elo": self._int(result.get("air_elo"), 0, 4000),
            "rank": self._rank_name(result),
            "summary": str(result.get("summary") or "")[:600],
            "niche_tags": self._strings(result.get("niche_tags"), 8),
            "core_traits": self._strings(result.get("core_traits"), 8),
            "detail_score": detail_score,
            "random_rolls": {
                "chaos": self._random.randint(1, 100),
                "luck": self._random.randint(1, 100),
                "style": self._random.randint(1, 100),
            },
            "metrics": self._metric_rows(metrics, 18),
            "tactics": self._tactic_rows(tactics, 8),
            "evidence": self._evidence_rows(evidence, 5),
            "fun_metrics": self._fun_rows(result.get("fun_metrics"), 4),
        }

    def _detail_score(self, metrics: dict[str, Any], tactics: list[Any]) -> int:
        metric_scores = []
        for value in metrics.values():
            item = value if isinstance(value, dict) else {}
            metric_scores.append(self._int(item.get("score"), 0, 100))
        tactic_scores = []
        for value in tactics:
            item = value if isinstance(value, dict) else {}
            tactic_scores.append(self._int(item.get("intensity"), 0, 100))
        metric_avg = sum(metric_scores) / len(metric_scores) if metric_scores else 0.0
        tactic_avg = sum(tactic_scores[:6]) / min(len(tactic_scores), 6) if tactic_scores else 0.0
        chaos = self._random.randint(1, 100)
        return round(metric_avg * 0.62 + tactic_avg * 0.23 + chaos * 0.15)

    def _normalize(
        self,
        data: dict[str, Any],
        left: BotRun,
        right: BotRun,
        topic: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        rounds = data.get("rounds") if isinstance(data.get("rounds"), list) else []
        if not rounds:
            raise AirbotFightError("llm returned invalid fight json")
        final = data.get("final") if isinstance(data.get("final"), dict) else {}
        if not final:
            raise AirbotFightError("llm returned invalid fight json")
        out_rounds = [self._round(item, left, right, index + 1) for index, item in enumerate(rounds[:3])]
        winner = self._winner_id(final.get("winner"), left, right)
        winner_channel = self._winner_channel(winner, left, right)
        return {
            "schema_version": AIRBOT_FIGHT_SCHEMA_VERSION,
            "fight_id": "",
            "topic": topic,
            "fighters": [
                self._fighter_summary(payload["fighters"][0]),
                self._fighter_summary(payload["fighters"][1]),
            ],
            "rounds": out_rounds,
            "final": {
                "winner": winner,
                "channel": winner_channel,
                "decisive_round": self._int(final.get("decisive_round"), 0, 3),
                "method": str(final.get("method") or "")[:160],
                "verdict": str(final.get("verdict") or "")[:900],
                "why": str(final.get("why") or "")[:900],
            },
            "share_line": str(data.get("share_line") or "")[:240],
        }

    def _round(self, value: Any, left: BotRun, right: BotRun, number: int) -> dict[str, Any]:
        item = value if isinstance(value, dict) else {}
        winner = self._winner_id(item.get("winner"), left, right)
        return {
            "number": self._int(item.get("round"), 1, 3) or number,
            "title": str(item.get("title") or "")[:140],
            "winner": winner,
            "winner_channel": self._winner_channel(winner, left, right),
            "decisive": bool(item.get("decisive")),
            "snapshot": str(item.get("snapshot") or "")[:900],
            "logic": str(item.get("logic") or "")[:600],
        }

    def _fighter_summary(self, value: dict[str, Any]) -> dict[str, Any]:
        return {
            "side": value["side"],
            "channel": value["channel"],
            "air_elo": value["air_elo"],
            "rank": value["rank"],
            "detail_score": value["detail_score"],
            "random_rolls": value["random_rolls"],
        }

    def _rules(self) -> dict[str, Any]:
        rules = bot_texts().get("fight_rules")
        return rules if isinstance(rules, dict) else {}

    def _rank_name(self, result: dict[str, Any]) -> str:
        profile = result.get("meme_profile") if isinstance(result.get("meme_profile"), dict) else {}
        return str(profile.get("rank_name") or result.get("humorous_label") or "")[:160]

    def _metric_rows(self, metrics: dict[str, Any], limit: int) -> list[dict[str, Any]]:
        labels = bot_texts().get("metric_labels")
        label_map = labels if isinstance(labels, dict) else {}
        rows = []
        for key, value in metrics.items():
            item = value if isinstance(value, dict) else {}
            rows.append(
                {
                    "code": str(key),
                    "label": str(label_map.get(str(key)) or key),
                    "score": self._int(item.get("score"), 0, 100),
                    "why": str(item.get("why") or "")[:200],
                }
            )
        rows.sort(key=lambda item: item["score"], reverse=True)
        return rows[:limit]

    def _tactic_rows(self, tactics: list[Any], limit: int) -> list[dict[str, Any]]:
        rows = []
        for value in tactics[:limit]:
            item = value if isinstance(value, dict) else {}
            rows.append(
                {
                    "code": str(item.get("code") or "")[:80],
                    "name": str(item.get("name") or item.get("hint") or "")[:160],
                    "intensity": self._int(item.get("intensity"), 0, 100),
                    "hint": str(item.get("hint") or item.get("why_it_keeps_attention") or "")[:240],
                }
            )
        return rows

    def _evidence_rows(self, evidence: list[Any], limit: int) -> list[dict[str, str]]:
        rows = []
        for value in evidence[:limit]:
            item = value if isinstance(value, dict) else {}
            rows.append(
                {
                    "url": str(item.get("url") or "")[:220],
                    "snippet": str(item.get("snippet") or "")[:240],
                    "comment": str(item.get("comment") or "")[:200],
                }
            )
        return rows

    def _fun_rows(self, value: Any, limit: int) -> list[dict[str, str]]:
        items = value if isinstance(value, list) else []
        rows = []
        for raw in items[:limit]:
            item = raw if isinstance(raw, dict) else {}
            rows.append({"label": str(item.get("label") or "")[:80], "display": str(item.get("display") or "")[:80]})
        return rows

    def _winner_id(self, value: Any, left: BotRun, right: BotRun) -> str:
        raw = str(value or "").strip().lower().lstrip("@")
        if raw in {"left", "l", left.channel.lower()}:
            return "left"
        if raw in {"right", "r", right.channel.lower()}:
            return "right"
        if raw in {"draw", "tie", "none"}:
            return "draw"
        return ""

    def _winner_channel(self, winner: str, left: BotRun, right: BotRun) -> str:
        if winner == "left":
            return left.channel
        if winner == "right":
            return right.channel
        return ""

    def _json_object(self, content: str) -> dict[str, Any]:
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start < 0 or end <= start:
                raise AirbotFightError("llm returned invalid fight json") from None
            data = json.loads(content[start : end + 1])
        if not isinstance(data, dict):
            raise AirbotFightError("llm returned invalid fight json")
        return data

    def _topic(self, value: str) -> str:
        return " ".join(str(value or "").split())[:180]

    def _strings(self, value: Any, limit: int) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item)[:160] for item in value[:limit] if item is not None]

    def _int(self, value: Any, low: int, high: int) -> int:
        try:
            parsed = int(value)
        except Exception:
            parsed = low
        return min(max(parsed, low), high)


def fight_result_html(result: dict[str, Any]) -> str:
    fighters = result.get("fighters") if isinstance(result.get("fighters"), list) else []
    rounds = result.get("rounds") if isinstance(result.get("rounds"), list) else []
    final = result.get("final") if isinstance(result.get("final"), dict) else {}
    return text(
        "fight.result_html",
        topic=escape(str(result.get("topic") or "")),
        fighters="\n".join(_fighter_html(item, result) for item in fighters[:2]),
        rating=_rating_html(result),
        rounds="\n".join(_round_html(item) for item in rounds[:3]),
        winner=_winner_html(final),
        method=escape(str(final.get("method") or "")),
        verdict=escape(str(final.get("verdict") or "")),
        why=escape(str(final.get("why") or "")),
        share=escape(str(result.get("share_line") or "")),
    )


def fight_result_plain(result: dict[str, Any]) -> str:
    lines = [text("fight.result_title"), text("fight.topic_line", topic=str(result.get("topic") or ""))]
    for item in (result.get("rounds") if isinstance(result.get("rounds"), list) else [])[:3]:
        lines.append(text("fight.round_plain", number=item.get("number"), title=item.get("title"), winner=_winner_plain(item), body=item.get("snapshot")))
    final = result.get("final") if isinstance(result.get("final"), dict) else {}
    lines.append(text("fight.final_plain", winner=_winner_plain(final), verdict=str(final.get("verdict") or "")))
    rating = _rating_plain(result)
    if rating:
        lines.append(rating)
    return "\n\n".join(lines)


def fight_failed_public_error(exc: Exception) -> str:
    value = str(exc)
    if value == "llm is not configured":
        return text("llm_missing")
    if value.startswith("llm request failed"):
        return text("llm_unavailable")
    if value.startswith("llm returned"):
        return text("llm_bad_json")
    return value or type(exc).__name__


def fight_matches_html(channel: str, fights: list[BotFightRun], page: int) -> str:
    if not fights:
        return text("fight.matches_empty_html", channel=_channel_link(channel))
    rows = []
    for index, fight in enumerate(fights, start=1 + page * 5):
        rows.append(
            text(
                "fight.matches_row_html",
                index=escape(str(index)),
                opponent=_channel_link(_opponent_channel(fight, channel)),
                topic=escape(fight.topic),
                outcome=escape(_outcome_label(fight, channel)),
                delta=escape(_signed(_rating_delta_for_channel(fight.result or {}, channel))),
            )
        )
    return text(
        "fight.matches_html",
        channel=_channel_link(channel),
        page=escape(str(page + 1)),
        rows="\n".join(rows),
    )


def fight_matches_plain(channel: str, fights: list[BotFightRun], page: int) -> str:
    if not fights:
        return text("fight.matches_empty", channel=f"@{channel}")
    lines = [text("fight.matches_title", channel=f"@{channel}", page=page + 1)]
    for index, fight in enumerate(fights, start=1 + page * 5):
        lines.append(
            text(
                "fight.matches_row_plain",
                index=index,
                opponent=f"@{_opponent_channel(fight, channel)}",
                topic=fight.topic,
                outcome=_outcome_label(fight, channel),
                delta=_signed(_rating_delta_for_channel(fight.result or {}, channel)),
            )
        )
    return "\n".join(lines)


def fight_match_button_label(fight: BotFightRun, channel: str, index: int) -> str:
    return text(
        "buttons.fight_match_item",
        index=index,
        opponent=_opponent_channel(fight, channel),
        outcome=_outcome_label(fight, channel),
        delta=_signed(_rating_delta_for_channel(fight.result or {}, channel)),
    )


def _fighter_html(item: Any, result: dict[str, Any]) -> str:
    value = item if isinstance(item, dict) else {}
    channel = str(value.get("channel") or "")
    rating = _rating_side(result, str(value.get("side") or ""), channel)
    return text(
        "fight.fighter_html",
        channel=_channel_link(channel),
        elo=escape(str(value.get("air_elo") or "")),
        delta=escape(_signed(_plain_int(rating.get("delta")))),
        after=escape(str(rating.get("after") or "")),
        detail=escape(str(value.get("detail_score") or "")),
        rank=escape(str(value.get("rank") or "")),
    )


def _round_html(item: Any) -> str:
    value = item if isinstance(item, dict) else {}
    return text(
        "fight.round_html",
        number=escape(str(value.get("number") or "")),
        title=escape(str(value.get("title") or "")),
        winner=_winner_html(value),
        snapshot=escape(str(value.get("snapshot") or "")),
        logic=escape(str(value.get("logic") or "")),
    )


def _winner_html(final: dict[str, Any]) -> str:
    channel = str(final.get("winner_channel") or final.get("channel") or "")
    if channel:
        return _channel_link(channel)
    return escape(text("fight.draw"))


def _winner_plain(item: dict[str, Any]) -> str:
    channel = str(item.get("winner_channel") or item.get("channel") or "")
    if channel:
        return f"@{channel}"
    return text("fight.draw")


def _opponent_channel(fight: BotFightRun, channel: str) -> str:
    clean = channel.lower().strip().lstrip("@")
    if fight.left_channel == clean:
        return fight.right_channel
    return fight.left_channel


def _outcome_label(fight: BotFightRun, channel: str) -> str:
    result = fight.result or {}
    final = result.get("final") if isinstance(result.get("final"), dict) else {}
    winner = str(final.get("channel") or "").lower()
    clean = channel.lower().strip().lstrip("@")
    if not winner:
        return text("fight.outcome_draw")
    if winner == clean:
        return text("fight.outcome_win")
    return text("fight.outcome_loss")


def _rating_plain(result: dict[str, Any]) -> str:
    rating = result.get("rating_change") if isinstance(result.get("rating_change"), dict) else {}
    left = rating.get("left") if isinstance(rating, dict) else None
    right = rating.get("right") if isinstance(rating, dict) else None
    if not isinstance(left, dict) or not isinstance(right, dict):
        return ""
    return text(
        "fight.rating_plain",
        left=f"@{left.get('channel') or ''}",
        left_delta=_signed(_plain_int(left.get("delta"))),
        left_after=str(left.get("after") or ""),
        right=f"@{right.get('channel') or ''}",
        right_delta=_signed(_plain_int(right.get("delta"))),
        right_after=str(right.get("after") or ""),
        gap=str(rating.get("gap") or ""),
        k=str(rating.get("k_factor") or ""),
        multiplier=str(rating.get("gap_multiplier") or ""),
    )


def _rating_html(result: dict[str, Any]) -> str:
    rating = result.get("rating_change") if isinstance(result.get("rating_change"), dict) else {}
    if not rating:
        return ""
    return text(
        "fight.rating_html",
        gap=escape(str(rating.get("gap") or "")),
        k=escape(str(rating.get("k_factor") or "")),
        multiplier=escape(str(rating.get("gap_multiplier") or "")),
    )


def _rating_side(result: dict[str, Any], side: str, channel: str) -> dict[str, Any]:
    rating = result.get("rating_change") if isinstance(result.get("rating_change"), dict) else {}
    if side in {"left", "right"}:
        item = rating.get(side) if isinstance(rating, dict) else None
        return item if isinstance(item, dict) else {}
    clean = channel.lower().strip().lstrip("@")
    for key in ("left", "right"):
        item = rating.get(key) if isinstance(rating, dict) else None
        data = item if isinstance(item, dict) else {}
        if str(data.get("channel") or "").lower().lstrip("@") == clean:
            return data
    return {}


def _rating_delta_for_channel(result: dict[str, Any], channel: str) -> int:
    clean = channel.lower().strip().lstrip("@")
    for side in ("left", "right"):
        data = _rating_side(result, side, "")
        if str(data.get("channel") or "").lower().lstrip("@") == clean:
            return _plain_int(data.get("delta"))
    return 0


def _signed(value: int) -> str:
    if value > 0:
        return f"+{value}"
    return str(value)


def _plain_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _channel_link(channel: str) -> str:
    if not channel:
        return ""
    value = escape(channel)
    url = escape(f"https://t.me/{channel}", quote=True)
    return f"<a href=\"{url}\">@{value}</a>"
