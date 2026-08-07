from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any

from tg_radar.agent_runtime.openai_transport import ChatCompletionsTransport, OpenAITransport, OpenAITransportConfig
from tg_radar.agent_runtime.prompt_loader import PromptLoader
from tg_radar.config import Settings
from tg_radar.schemas import ParsedMessage


OPERATOR_QUALITY_SCHEMA_VERSION = 1
NICHE_CATEGORIES = ("novel", "saturated", "hobby", "mixed", "unclear")
INCOME_AUTHENTICITY_VALUES = ("likely_real_income", "public_activity_or_hobby", "uncertain")


class OperatorQualityError(RuntimeError):
    pass


class OperatorQualityAnalyzer:
    def __init__(
        self,
        settings: Settings,
        transport: OpenAITransport | None = None,
        prompt_loader: PromptLoader | None = None,
    ) -> None:
        self.settings = settings
        self.prompt_loader = prompt_loader or PromptLoader(override_dir=settings.agent_prompt_dir)
        self.transport = transport

    async def analyze(self, channel: str, messages: list[ParsedMessage]) -> dict[str, Any]:
        if not messages:
            raise OperatorQualityError("no messages")
        transport = self.transport or self._build_transport(self.settings)
        limit = max(1, self.settings.operator_quality_message_limit)
        recent = sorted(messages, key=self._sort_key, reverse=True)[:limit]
        content = await self._complete_json(
            transport,
            [
                {"role": "system", "content": self.prompt_loader.load("operator_quality_classification.md")},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "channel": channel,
                            "message_count": len(messages),
                            "messages": [self._compact_message(message) for message in recent],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        )
        parsed = self._json_object(content)
        parsed["channel"] = channel
        parsed["message_count"] = len(messages)
        return self._normalize_result(parsed)

    def _sort_key(self, message: ParsedMessage) -> datetime:
        posted_at = message.posted_at
        if posted_at and not posted_at.tzinfo:
            posted_at = posted_at.replace(tzinfo=timezone.utc)
        return posted_at or datetime.min.replace(tzinfo=timezone.utc)

    def _build_transport(self, settings: Settings) -> OpenAITransport:
        if not settings.llm_base_url or not settings.llm_api_key:
            raise OperatorQualityError("llm is not configured")
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

    def _compact_message(self, message: ParsedMessage) -> dict[str, Any]:
        return {
            "url": message.url,
            "posted_at": message.posted_at.isoformat() if message.posted_at else None,
            "text": (message.text or "")[:900],
            "links": list(message.links or [])[:10],
            "mentions": list(message.mentions or [])[:20],
        }

    async def _complete_json(self, transport: OpenAITransport, messages: list[dict[str, str]]) -> str:
        try:
            return await transport.complete_json(messages)
        except Exception as exc:
            raise OperatorQualityError(f"llm request failed: {exc}") from exc

    def _json_object(self, content: str) -> dict[str, Any]:
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start < 0 or end <= start:
                raise OperatorQualityError("llm returned invalid json") from None
            try:
                data = json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                raise OperatorQualityError("llm returned invalid json") from None
        if not isinstance(data, dict):
            raise OperatorQualityError("llm returned non-object json")
        return data

    def _normalize_result(self, result: dict[str, Any]) -> dict[str, Any]:
        niche_category = str(result.get("niche_category") or "")
        income_authenticity = str(result.get("income_authenticity") or "")
        return {
            "schema_version": OPERATOR_QUALITY_SCHEMA_VERSION,
            "channel": str(result.get("channel") or ""),
            "message_count": self._clamp_int(result.get("message_count"), 0, 1000000),
            "niche_tags": [tag[:64] for tag in self._strings(result.get("niche_tags"), 8)],
            "niche_category": niche_category if niche_category in NICHE_CATEGORIES else "unclear",
            "income_authenticity": income_authenticity if income_authenticity in INCOME_AUTHENTICITY_VALUES else "uncertain",
            "confidence": self._clamp_float(result.get("confidence"), 0.0, 1.0),
            "evidence": self._evidence(result.get("evidence")),
            "notes": [note[:200] for note in self._strings(result.get("notes"), 8)],
        }

    def _evidence(self, value: Any) -> list[dict[str, str]]:
        items = value if isinstance(value, list) else []
        out: list[dict[str, str]] = []
        for raw in items[:10]:
            item = raw if isinstance(raw, dict) else {}
            out.append(
                {
                    "url": str(item.get("url") or "")[:220],
                    "snippet": str(item.get("snippet") or "")[:180],
                    "why": str(item.get("why") or "")[:160],
                }
            )
        return out

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
