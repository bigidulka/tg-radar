from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import json
from typing import Any

from tg_radar.agent_runtime.openai_transport import ChatCompletionsTransport, OpenAITransport, OpenAITransportConfig
from tg_radar.agent_runtime.prompt_loader import PromptLoader
from tg_radar.bot.ratings import air_elo, elo_rank, elo_visual_score
from tg_radar.bot.texts import bot_texts
from tg_radar.config import Settings


ProgressCallback = Callable[[int, int], Awaitable[None]]
AIRBOT_RESULT_SCHEMA_VERSION = 4


class AirbotAnalysisError(RuntimeError):
    pass


class AirbotAnalyzer:
    def __init__(
        self,
        settings: Settings,
        transport: OpenAITransport | None = None,
        prompt_loader: PromptLoader | None = None,
    ) -> None:
        self.settings = settings
        self.prompt_loader = prompt_loader or PromptLoader(override_dir=settings.agent_prompt_dir)
        self.transport = transport
        self._llm_semaphore = asyncio.Semaphore(settings.bot_llm_global_concurrency)

    async def analyze(
        self,
        channel: str,
        messages: list[dict[str, Any]],
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        if not messages:
            raise AirbotAnalysisError("no messages")
        transport = self.transport or self._build_transport(self.settings)
        chunk_size = max(1, self.settings.bot_llm_chunk_size)
        chunks = [messages[index : index + chunk_size] for index in range(0, len(messages), chunk_size)]
        chunk_results = await self._analyze_chunks(channel, chunks, transport, progress)
        merged = self._merge_locally(channel, chunk_results)
        merged["channel"] = channel
        merged["message_count"] = len(messages)
        return self._normalize_result(merged)

    async def _analyze_chunks(
        self,
        channel: str,
        chunks: list[list[dict[str, Any]]],
        transport: OpenAITransport,
        progress: ProgressCallback | None,
    ) -> list[dict[str, Any]]:
        if len(chunks) <= 1:
            result = [await self._analyze_chunk(channel, chunks[0], transport)]
            if progress:
                await progress(1, 1)
            return result
        semaphore = asyncio.Semaphore(self.settings.bot_llm_chunk_concurrency)
        results: list[dict[str, Any] | None] = [None] * len(chunks)
        done = 0

        async def run_chunk(index: int, chunk: list[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
            async with semaphore:
                return index, await self._analyze_chunk(channel, chunk, transport)

        tasks = [asyncio.create_task(run_chunk(index, chunk)) for index, chunk in enumerate(chunks)]
        try:
            for task in asyncio.as_completed(tasks):
                index, result = await task
                results[index] = result
                done += 1
                if progress:
                    await progress(done, len(chunks))
        except Exception:
            for task in tasks:
                task.cancel()
            raise
        return [item for item in results if item is not None]

    def _build_transport(self, settings: Settings) -> OpenAITransport:
        if not settings.llm_base_url or not settings.llm_api_key:
            raise AirbotAnalysisError("llm is not configured")
        return ChatCompletionsTransport(
            OpenAITransportConfig(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
                model=settings.llm_model,
                timeout_seconds=settings.llm_timeout_seconds,
                temperature=settings.agent_model_temperature,
                max_tokens=settings.agent_model_max_tokens,
                json_response_format=False,
            )
        )

    async def _analyze_chunk(self, channel: str, messages: list[dict[str, Any]], transport: OpenAITransport) -> dict[str, Any]:
        content = await self._complete_json(
            transport,
            [
                {"role": "system", "content": self.prompt_loader.load("airbot_chunk_analysis.md")},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "channel": channel,
                            "messages": [self._compact_message(message) for message in messages],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        )
        parsed = self._json_object(content)
        parsed["message_count"] = parsed.get("message_count") or len(messages)
        return parsed

    async def _merge(self, channel: str, chunk_results: list[dict[str, Any]], transport: OpenAITransport) -> dict[str, Any]:
        content = await self._complete_json(
            transport,
            [
                {"role": "system", "content": self.prompt_loader.load("airbot_merge_analysis.md")},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "channel": channel,
                            "chunk_results": [self._compact_chunk_result(chunk) for chunk in chunk_results],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        )
        return self._json_object(content)

    def _merge_locally(self, channel: str, chunk_results: list[dict[str, Any]]) -> dict[str, Any]:
        chunks = [self._compact_chunk_result(chunk) for chunk in chunk_results]
        message_count = sum(self._clamp_int(chunk.get("message_count"), 0, 1000000) for chunk in chunks)
        metrics = self._merge_metrics(chunks)
        tactics = self._merge_tactics(chunks)
        evidence = self._merge_evidence(chunks)
        niche_tags = self._top_strings([tag for chunk in chunks for tag in chunk.get("niche_tags", [])], 8)
        core_traits = self._top_strings([trait for chunk in chunks for trait in chunk.get("core_traits", [])], 8)
        counter_signals = self._merge_counter_signals(chunks)
        meme_profile = self._local_meme_profile(metrics, tactics, chunks)
        summary = self._local_summary(core_traits, tactics, counter_signals)
        return {
            "schema_version": AIRBOT_RESULT_SCHEMA_VERSION,
            "channel": channel,
            "message_count": message_count,
            "niche_tags": niche_tags,
            "core_traits": core_traits,
            "confidence": 0.72 if message_count >= 30 else 0.58,
            "humorous_label": meme_profile["air_title"],
            "summary": summary,
            "metrics": metrics,
            "tactics": tactics,
            "evidence": evidence,
            "counter_signals": counter_signals,
            "text_generation_hooks": self._local_text_generation_hooks(tactics, chunks),
            "meme_profile": meme_profile,
            "disclaimer": self._local_text("disclaimer"),
        }

    def _merge_metrics(self, chunks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        totals: dict[str, int] = {}
        weights: dict[str, int] = {}
        reasons: dict[str, list[str]] = {}
        for chunk in chunks:
            weight = max(1, self._clamp_int(chunk.get("message_count"), 1, 1000000))
            for name, score in (chunk.get("metric_scores") or {}).items():
                parsed = self._clamp_int(score, 0, 100)
                totals[name] = totals.get(name, 0) + parsed * weight
                weights[name] = weights.get(name, 0) + weight
        for tactic in self._merge_tactics(chunks):
            code = tactic.get("code") or ""
            if code:
                reasons.setdefault(code, []).append(str(tactic.get("hint") or tactic.get("name") or ""))
        return {
            name: {
                "score": round(totals.get(name, 0) / max(1, weights.get(name, 0))),
                "why": "; ".join([value for value in reasons.get(name, []) if value][:2]) or self._local_text("metric_average"),
            }
            for name in self._metric_names()
        }

    def _merge_tactics(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_code: dict[str, dict[str, Any]] = {}
        for chunk in chunks:
            for tactic in chunk.get("detected_tactics") or []:
                code = str(tactic.get("code") or "")
                if not code:
                    continue
                current = by_code.get(code)
                intensity = self._clamp_int(tactic.get("strength"), 0, 100)
                if current is None or intensity > current["intensity"]:
                    by_code[code] = {
                        "code": code,
                        "name": str(tactic.get("hint") or code)[:80],
                        "intensity": intensity,
                        "hint": str(tactic.get("hint") or "")[:180],
                        "why_it_keeps_attention": str(tactic.get("hint") or self._local_text("attention_hook"))[:220],
                        "evidence_urls": self._strings(tactic.get("evidence_urls"), 5),
                    }
        return sorted(by_code.values(), key=lambda item: item["intensity"], reverse=True)[:10]

    def _merge_evidence(self, chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        seen: set[str] = set()
        for chunk in chunks:
            for quote in chunk.get("notable_quotes") or []:
                url = str(quote.get("url") or "")
                if url in seen:
                    continue
                seen.add(url)
                out.append(
                    {
                        "url": url,
                        "metric": str(quote.get("metric") or ""),
                        "snippet": str(quote.get("snippet") or "")[:180],
                        "comment": str(quote.get("why") or "")[:160],
                    }
                )
                if len(out) >= 12:
                    return out
        return out

    def _merge_counter_signals(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        seen: set[str] = set()
        for chunk in chunks:
            for signal in chunk.get("counter_signals") or []:
                name = str(signal.get("name") or "")
                if not name or name in seen:
                    continue
                seen.add(name)
                out.append({"name": name, "comment": name, "evidence_urls": self._strings(signal.get("evidence_urls"), 5)})
                if len(out) >= 8:
                    return out
        return out

    def _local_meme_profile(self, metrics: dict[str, dict[str, Any]], tactics: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> dict[str, str]:
        top = tactics[0] if tactics else {}
        top_name = str(top.get("name") or top.get("code") or self._local_text("default_top_name"))
        observations = [chunk.get("meme_observations") or {} for chunk in chunks]
        metaphors = [value for item in observations for value in item.get("weather_metaphors", [])]
        gaps = [value for item in observations for value in item.get("missing_action_patterns", [])]
        return {
            "air_title": top_name[:80],
            "custom_tag": top_name[:80],
            "weather_metaphor": (metaphors[0] if metaphors else self._local_text("weather_metaphor"))[:180],
            "one_liner": self._local_text("one_liner", top_name=top_name)[:180],
            "promise_gap": (gaps[0] if gaps else self._local_text("promise_gap"))[:180],
            "safe_roast": self._local_text("safe_roast"),
        }

    def _local_summary(self, traits: list[str], tactics: list[dict[str, Any]], counter_signals: list[dict[str, Any]]) -> str:
        main = ", ".join(traits[:3]) or self._local_text("summary_main")
        tactic = str((tactics[0] or {}).get("name") or (tactics[0] or {}).get("code") or self._local_text("summary_tactic")) if tactics else self._local_text("summary_tactic")
        counter = self._local_text("summary_counter", counter=counter_signals[0]["name"]) if counter_signals else ""
        return self._local_text("summary", main=main, tactic=tactic, counter=counter)

    def _local_text_generation_hooks(self, tactics: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> dict[str, list[str]]:
        jokes = [str(tactic.get("name") or tactic.get("code") or "") for tactic in tactics[:5]]
        observations = [chunk.get("meme_observations") or {} for chunk in chunks]
        angles = [value for item in observations for value in item.get("safe_joke_angles", [])][:5]
        return {
            "roast_angles": angles or jokes,
            "running_jokes": [self._local_text("running_joke", joke=joke) for joke in jokes[:5] if joke],
            "safe_phrases": self._local_list("safe_phrases"),
        }

    def _local_text(self, name: str, **values: object) -> str:
        texts = bot_texts().get("local_analysis")
        value = texts.get(name) if isinstance(texts, dict) else None
        if not isinstance(value, str):
            return name
        return value.format(**values)

    def _local_list(self, name: str) -> list[str]:
        texts = bot_texts().get("local_analysis")
        value = texts.get(name) if isinstance(texts, dict) else None
        return [str(item) for item in value] if isinstance(value, list) else []

    def _top_strings(self, values: list[str], limit: int) -> list[str]:
        counts: dict[str, int] = {}
        for value in values:
            clean = str(value).strip()
            if clean:
                counts[clean] = counts.get(clean, 0) + 1
        return [item for item, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]]

    def _metric_names(self) -> tuple[str, ...]:
        return (
            "fomo_pressure",
            "ragebait",
            "clickbait_open_loop",
            "success_theater",
            "authority_flex",
            "vague_promise",
            "product_warmup",
            "audience_hook",
            "anti_building",
            "proof_of_building_reverse",
            "toxic_aggression",
            "shame_guilt_pressure",
            "monetization_intensity",
            "parasocial_tribe",
            "novelty_hype",
            "financial_lure",
            "certainty_overclaim",
            "evidence_specificity_reverse",
        )

    async def _complete_json(self, transport: OpenAITransport, messages: list[dict[str, str]]) -> str:
        try:
            async with self._llm_semaphore:
                return await transport.complete_json(messages)
        except Exception as exc:
            raise AirbotAnalysisError(f"llm request failed: {exc}") from exc

    def _compact_message(self, message: dict[str, Any]) -> dict[str, Any]:
        text = str(message.get("text") or "")
        return {
            "message_id": message.get("message_id"),
            "url": message.get("url"),
            "posted_at": message.get("posted_at"),
            "text": text[:900],
            "views": message.get("views"),
            "links": list(message.get("links") or [])[:10],
            "mentions": list(message.get("mentions") or [])[:20],
        }

    def _compact_chunk_result(self, value: dict[str, Any]) -> dict[str, Any]:
        metrics = value.get("metric_scores") if isinstance(value.get("metric_scores"), dict) else value.get("metrics")
        item = {
            "message_count": value.get("message_count"),
            "niche_tags": self._strings(value.get("niche_tags"), 4),
            "core_traits": self._strings(value.get("core_traits"), 5),
            "metric_scores": self._compact_metric_scores(metrics),
            "detected_tactics": self._compact_chunk_tactics(value.get("detected_tactics") or value.get("tactics")),
            "counter_signals": self._compact_chunk_counter_signals(value.get("counter_signals")),
            "notable_quotes": self._compact_chunk_quotes(value.get("notable_quotes") or value.get("evidence")),
            "meme_observations": self._compact_meme_observations(value.get("meme_observations")),
        }
        notes = self._strings(value.get("notes"), 2)
        if notes:
            item["notes"] = [note[:160] for note in notes]
        return item

    def _compact_metric_scores(self, value: Any) -> dict[str, int]:
        raw = value if isinstance(value, dict) else {}
        out: dict[str, int] = {}
        for key, metric in raw.items():
            if isinstance(metric, dict):
                score = metric.get("score")
            else:
                score = metric
            out[str(key)] = self._clamp_int(score, 0, 100)
        return out

    def _compact_chunk_tactics(self, value: Any) -> list[dict[str, Any]]:
        items = value if isinstance(value, list) else []
        out = []
        for raw in items[:6]:
            item = raw if isinstance(raw, dict) else {}
            out.append(
                {
                    "code": str(item.get("code") or "")[:80],
                    "strength": self._clamp_int(item.get("strength") or item.get("intensity"), 0, 100),
                    "hint": str(item.get("hint") or item.get("name") or "")[:180],
                    "evidence_urls": self._strings(item.get("evidence_urls"), 3),
                }
            )
        return out

    def _compact_chunk_counter_signals(self, value: Any) -> list[dict[str, Any]]:
        items = value if isinstance(value, list) else []
        out = []
        for raw in items[:4]:
            item = raw if isinstance(raw, dict) else {}
            out.append(
                {
                    "name": str(item.get("name") or "")[:120],
                    "evidence_urls": self._strings(item.get("evidence_urls"), 2),
                }
            )
        return out

    def _compact_chunk_quotes(self, value: Any) -> list[dict[str, str]]:
        items = value if isinstance(value, list) else []
        out = []
        for raw in items[:5]:
            item = raw if isinstance(raw, dict) else {}
            out.append(
                {
                    "url": str(item.get("url") or "")[:220],
                    "metric": str(item.get("metric") or "")[:80],
                    "snippet": str(item.get("snippet") or item.get("text") or "")[:180],
                    "why": str(item.get("why") or item.get("comment") or "")[:160],
                }
            )
        return out

    def _compact_meme_observations(self, value: Any) -> dict[str, list[str]]:
        item = value if isinstance(value, dict) else {}
        return {
            "promise_patterns": [x[:140] for x in self._strings(item.get("promise_patterns"), 3)],
            "missing_action_patterns": [x[:140] for x in self._strings(item.get("missing_action_patterns"), 3)],
            "weather_metaphors": [x[:120] for x in self._strings(item.get("weather_metaphors"), 3)],
            "safe_joke_angles": [x[:140] for x in self._strings(item.get("safe_joke_angles"), 3)],
        }

    def _json_object(self, content: str) -> dict[str, Any]:
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start < 0 or end <= start:
                raise AirbotAnalysisError("llm returned invalid json") from None
            data = json.loads(content[start : end + 1])
        if not isinstance(data, dict):
            raise AirbotAnalysisError("llm returned non-object json")
        return data

    def _normalize_result(self, result: dict[str, Any]) -> dict[str, Any]:
        message_count = self._clamp_int(result.get("message_count"), 0, 1000000)
        normalized = dict(result)
        normalized["schema_version"] = AIRBOT_RESULT_SCHEMA_VERSION
        normalized["channel"] = str(result.get("channel") or "")
        normalized["message_count"] = message_count
        metrics = self._metrics(result.get("metrics"))
        elo = air_elo(metrics)
        rank = elo_rank(elo)
        normalized["air_elo"] = elo
        normalized["air_elo_label"] = f"{elo} ELO"
        normalized["air_elo_bucket"] = rank.get("elo")
        normalized["air_elo_emoji"] = str(rank.get("emoji") or "")
        normalized["air_rank_unit_value"] = rank.get("unit_value")
        normalized["air_rank_unit"] = str(rank.get("unit") or "")
        normalized["air_rank_measure"] = self._rank_measure(rank)
        normalized["niche_tags"] = self._strings(result.get("niche_tags"), 8)
        normalized["core_traits"] = self._strings(result.get("core_traits"), 8)
        normalized["confidence"] = self._clamp_float(result.get("confidence"), 0.0, 1.0)
        normalized["humorous_label"] = str(result.get("humorous_label") or result.get("verdict") or "")
        normalized["summary"] = str(result.get("summary") or result.get("verdict") or "")
        normalized["metrics"] = metrics
        normalized["tactics"] = self._tactics(result.get("tactics"))
        normalized["evidence"] = self._rich_evidence(result.get("evidence"))
        normalized["counter_signals"] = self._counter_signals(result.get("counter_signals"))
        normalized["text_generation_hooks"] = self._text_generation_hooks(result.get("text_generation_hooks"))
        normalized["meme_profile"] = self._meme_profile(result.get("meme_profile"), elo, rank)
        normalized["fun_metrics"] = self._fun_metrics(elo)
        normalized["disclaimer"] = str(result.get("disclaimer") or "")
        normalized.pop("score", None)
        normalized.pop("score_label", None)
        return normalized

    def _metrics(self, value: Any) -> dict[str, dict[str, Any]]:
        names = (
            "fomo_pressure",
            "ragebait",
            "clickbait_open_loop",
            "success_theater",
            "authority_flex",
            "vague_promise",
            "product_warmup",
            "audience_hook",
            "anti_building",
            "proof_of_building_reverse",
            "toxic_aggression",
            "shame_guilt_pressure",
            "monetization_intensity",
            "parasocial_tribe",
            "novelty_hype",
            "financial_lure",
            "certainty_overclaim",
            "evidence_specificity_reverse",
        )
        raw = value if isinstance(value, dict) else {}
        out: dict[str, dict[str, Any]] = {}
        for name in names:
            item = raw.get(name) if isinstance(raw.get(name), dict) else {}
            out[name] = {
                "score": self._clamp_int(item.get("score"), 0, 100),
                "why": str(item.get("why") or ""),
            }
        return out

    def _tactics(self, value: Any) -> list[dict[str, Any]]:
        items = value if isinstance(value, list) else []
        return [self._tactic(item) for item in items[:12]]

    def _tactic(self, value: Any) -> dict[str, Any]:
        item = value if isinstance(value, dict) else {}
        return {
            "code": str(item.get("code") or ""),
            "name": str(item.get("name") or ""),
            "intensity": self._clamp_int(item.get("intensity"), 0, 100),
            "hint": str(item.get("hint") or ""),
            "why_it_keeps_attention": str(item.get("why_it_keeps_attention") or ""),
            "evidence_urls": self._strings(item.get("evidence_urls"), 5),
        }

    def _rich_evidence(self, value: Any) -> list[dict[str, str]]:
        items = value if isinstance(value, list) else []
        return [self._rich_evidence_item(item) for item in items[:12]]

    def _rich_evidence_item(self, value: Any) -> dict[str, str]:
        item = value if isinstance(value, dict) else {}
        return {
            "url": str(item.get("url") or ""),
            "metric": str(item.get("metric") or ""),
            "snippet": str(item.get("snippet") or item.get("text") or "")[:500],
            "comment": str(item.get("comment") or ""),
        }

    def _counter_signals(self, value: Any) -> list[dict[str, Any]]:
        items = value if isinstance(value, list) else []
        out = []
        for item in items[:8]:
            raw = item if isinstance(item, dict) else {}
            out.append(
                {
                    "name": str(raw.get("name") or ""),
                    "comment": str(raw.get("comment") or ""),
                    "evidence_urls": self._strings(raw.get("evidence_urls"), 5),
                }
            )
        return out

    def _text_generation_hooks(self, value: Any) -> dict[str, list[str]]:
        item = value if isinstance(value, dict) else {}
        return {
            "roast_angles": self._strings(item.get("roast_angles"), 8),
            "running_jokes": self._strings(item.get("running_jokes"), 8),
            "safe_phrases": self._strings(item.get("safe_phrases"), 8),
        }

    def _meme_profile(self, value: Any, elo: int, rank: dict[str, Any]) -> dict[str, Any]:
        raw = value if isinstance(value, dict) else {}
        return {
            "rank": self._clamp_int(raw.get("rank") if raw.get("rank") is not None else elo_visual_score(elo), 0, 200),
            "air_elo": elo,
            "air_elo_label": f"{elo} ELO",
            "air_elo_bucket": rank.get("elo"),
            "air_elo_emoji": str(rank.get("emoji") or ""),
            "rank_unit_value": rank.get("unit_value"),
            "rank_unit": str(rank.get("unit") or ""),
            "rank_measure": self._rank_measure(rank),
            "rank_name": str(rank.get("tag") or ""),
            "rank_summary": str(rank.get("summary") or ""),
            "air_title": self._string_or_default(raw.get("air_title"), rank.get("air_title")),
            "weather_metaphor": self._string_or_default(raw.get("weather_metaphor"), rank.get("weather_metaphor")),
            "one_liner": self._string_or_default(raw.get("one_liner"), rank.get("one_liner")),
            "custom_tag": str(raw.get("custom_tag") or raw.get("air_title") or ""),
            "promise_gap": str(raw.get("promise_gap") or ""),
            "safe_roast": str(raw.get("safe_roast") or ""),
        }

    def _rank_measure(self, rank: dict[str, Any]) -> str:
        value = rank.get("unit_value")
        unit = str(rank.get("unit") or "")
        if value is None or not unit:
            return ""
        return f"{value} {unit}"

    def _fun_metrics(self, elo: int) -> list[dict[str, Any]]:
        specs = bot_texts().get("fun_metrics")
        if not isinstance(specs, list):
            return []
        out: list[dict[str, Any]] = []
        for spec in specs:
            item = spec if isinstance(spec, dict) else {}
            key = str(item.get("key") or "")
            label = str(item.get("label") or "")
            unit = str(item.get("unit") or "")
            if not key or not label or not unit:
                continue
            value = (float(elo) ** self._float_default(item.get("power"), 1.0, 0.0, 4.0)) * self._float_default(item.get("multiplier"), 1.0, 0.0, 1000000.0)
            value += self._float_default(item.get("offset"), 0.0, -1000000.0, 1000000.0)
            value = self._clamp_float(
                value,
                self._float_default(item.get("min"), 0.0, 0.0, 1000000000.0),
                self._float_default(item.get("max"), 1000000000.0, 0.0, 1000000000.0),
            )
            precision = self._clamp_int(item.get("precision"), 0, 4)
            display = f"{round(value):,}".replace(",", " ") if precision == 0 else f"{value:.{precision}f}"
            out.append({"key": key, "label": label, "value": round(value, precision), "unit": unit, "display": f"{display} {unit}"})
        return out

    def _string_or_default(self, value: Any, default: Any) -> str:
        parsed = str(value or "")
        if parsed:
            return parsed
        return str(default or "")

    def _strings(self, value: Any, limit: int) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item) for item in value[:limit] if item is not None]

    def _clamp_int(self, value: Any, low: int, high: int) -> int:
        try:
            parsed = int(value)
        except Exception:
            parsed = low
        return min(max(parsed, low), high)

    def _clamp_float(self, value: Any, low: float, high: float) -> float:
        try:
            parsed = float(value)
        except Exception:
            parsed = low
        return min(max(parsed, low), high)

    def _float_default(self, value: Any, default: float, low: float, high: float) -> float:
        try:
            parsed = float(value)
        except Exception:
            parsed = default
        return min(max(parsed, low), high)
