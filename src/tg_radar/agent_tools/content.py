from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from tg_radar.agent_core.models import AgentObservation, ToolSpec
from tg_radar.config import Settings
from tg_radar.content_factory import (
    DEFAULT_MIN_USABLE_EVIDENCE,
    build_topic_evidence_map,
    build_topic_outline,
    cluster_topic_signals,
    create_article,
    finalize_topic_article,
    generate_cards_from_topic,
    generate_draft,
    generate_outline,
    infer_topic_signal_taxonomy,
    propose_topic_angles,
    review_topic_evidence,
    verify_topic_claims,
    write_topic_draft,
)


class ContentToolRegistry:
    def __init__(
        self,
        settings: Settings,
        sessionmaker: async_sessionmaker[AsyncSession],
        enabled_tools: set[str] | None = None,
    ) -> None:
        self.settings = settings
        self.sessionmaker = sessionmaker
        self.enabled_tools = enabled_tools

    def specs(self) -> list[ToolSpec]:
        specs = [
            ToolSpec(
                name="review_evidence",
                risk_level="safe",
                description="Review collected Telegram messages for topic match and usable article evidence.",
                args_schema={
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "article_focus": {"type": "string"},
                        "focus_keywords": {"type": "array", "items": {"type": "string"}},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 300},
                        "min_required": {"type": "integer", "minimum": 1, "maximum": 50},
                    },
                    "required": ["topic"],
                },
            ),
            ToolSpec(
                name="write_article",
                risk_level="db_write",
                description="Generate content cards, article outline, and draft from collected Telegram evidence.",
                args_schema={
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "article_focus": {"type": "string"},
                        "focus_keywords": {"type": "array", "items": {"type": "string"}},
                        "title": {"type": "string"},
                        "angle": {"type": "string"},
                        "instruction": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 300},
                        "min_required": {"type": "integer", "minimum": 1, "maximum": 50},
                        "use_llm": {"type": "boolean"},
                    },
                    "required": ["topic"],
                },
            ),
            ToolSpec(
                name="infer_signal_taxonomy",
                risk_level="db_write",
                description="Infer dynamic signal categories from focused Telegram evidence before clustering.",
                args_schema=self._topic_args_schema(),
            ),
            ToolSpec(
                name="cluster_signals",
                risk_level="safe",
                description="Cluster usable Telegram evidence into repeated signals using inferred taxonomy.",
                args_schema=self._topic_args_schema(),
            ),
            ToolSpec(
                name="build_evidence_map",
                risk_level="db_write",
                description="Build thesis, claims, caveats, and message-level evidence map from clustered signals.",
                args_schema=self._topic_args_schema(),
            ),
            ToolSpec(
                name="propose_angles",
                risk_level="db_write",
                description="Propose article angles based on the evidence map.",
                args_schema=self._topic_args_schema(required_limit=False),
            ),
            ToolSpec(
                name="build_outline",
                risk_level="db_write",
                description="Build article outline from claims and evidence map.",
                args_schema=self._topic_args_schema(required_limit=False),
            ),
            ToolSpec(
                name="write_draft",
                risk_level="db_write",
                description="Write a draft from outline blocks and source-backed claims.",
                args_schema={
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "article_focus": {"type": "string"},
                        "focus_keywords": {"type": "array", "items": {"type": "string"}},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 300},
                        "use_llm": {"type": "boolean"},
                    },
                    "required": ["topic"],
                },
            ),
            ToolSpec(
                name="verify_claims",
                risk_level="safe",
                description="Verify draft claims against source-backed claim cards.",
                args_schema={
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "article_focus": {"type": "string"},
                        "focus_keywords": {"type": "array", "items": {"type": "string"}},
                        "article_id": {"type": "integer", "minimum": 1},
                    },
                    "required": ["topic"],
                },
            ),
            ToolSpec(
                name="final_article",
                risk_level="db_write",
                description="Finalize an article only after claim verification passes.",
                args_schema={
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "article_focus": {"type": "string"},
                        "focus_keywords": {"type": "array", "items": {"type": "string"}},
                        "article_id": {"type": "integer", "minimum": 1},
                    },
                    "required": ["topic"],
                },
            ),
        ]
        if self.enabled_tools is None:
            return specs
        return [tool for tool in specs if tool.name in self.enabled_tools]

    async def run(self, name: str, args: dict[str, Any]) -> AgentObservation:
        if self.enabled_tools is not None and name not in self.enabled_tools:
            return AgentObservation(tool_name=name, ok=False, summary="tool disabled")
        if name not in {
            "review_evidence",
            "write_article",
            "infer_signal_taxonomy",
            "cluster_signals",
            "build_evidence_map",
            "propose_angles",
            "build_outline",
            "write_draft",
            "verify_claims",
            "final_article",
        }:
            return AgentObservation(tool_name=name, ok=False, summary="unknown tool")
        if not self.settings.content_factory_enabled:
            return AgentObservation(tool_name=name, ok=False, summary="content factory disabled")

        topic = str(args.get("topic") or "").strip()
        if not topic:
            return AgentObservation(tool_name=name, ok=False, summary="topic is required")
        article_focus = str(args.get("article_focus") or "").strip() or None
        focus_keywords = [str(item).strip() for item in (args.get("focus_keywords") or []) if str(item).strip()]
        instruction = str(
            args.get("instruction")
            or "Ragebait but evidence-based: thesis, evidence map, headlines, outline, draft. Do not invent facts."
        )
        use_llm = bool(args.get("use_llm", False))

        try:
            async with self.sessionmaker() as session:
                async with session.begin():
                    limit = int(args.get("limit") or 80)
                    min_required = int(args.get("min_required") or DEFAULT_MIN_USABLE_EVIDENCE)
                    review = await review_topic_evidence(session, topic, limit, min_required=min_required, article_focus=article_focus, focus_keywords=focus_keywords)
                    if name == "review_evidence":
                        return AgentObservation(
                            tool_name=name,
                            ok=True,
                            summary=(
                                f"usable={review.get('usable', 0)} "
                                f"rejected={review.get('rejected', 0)} "
                                f"min_required={review.get('min_required', min_required)}"
                            ),
                            data=review,
                        )
                    if name == "infer_signal_taxonomy":
                        result = await infer_topic_signal_taxonomy(
                            session,
                            self.settings,
                            topic,
                            limit,
                            min_required=min_required,
                            article_focus=article_focus,
                            focus_keywords=focus_keywords,
                        )
                        return AgentObservation(
                            tool_name=name,
                            ok=bool(result.get("ok")),
                            summary=f"signals={len(result.get('taxonomy') or [])} usable={result.get('usable', 0)} rejected={result.get('rejected', 0)}",
                            data=result,
                        )
                    if name == "cluster_signals":
                        taxonomy = args.get("taxonomy") if isinstance(args.get("taxonomy"), list) else None
                        result = await cluster_topic_signals(session, topic, limit, min_required=min_required, article_focus=article_focus, focus_keywords=focus_keywords, settings=self.settings, taxonomy=taxonomy)
                        return AgentObservation(
                            tool_name=name,
                            ok=bool(result.get("ok")),
                            summary=f"signals={len(result.get('signals') or [])} usable={result.get('usable', 0)} rejected={result.get('rejected', 0)}",
                            data=result,
                        )
                    if name == "build_evidence_map":
                        taxonomy = args.get("taxonomy") if isinstance(args.get("taxonomy"), list) else None
                        result = await build_topic_evidence_map(session, topic, limit, min_required=min_required, article_focus=article_focus, focus_keywords=focus_keywords, settings=self.settings, taxonomy=taxonomy)
                        return AgentObservation(
                            tool_name=name,
                            ok=bool(result.get("ok")),
                            summary=f"claims={len(result.get('claims') or [])} signals={len(result.get('signals') or [])}",
                            data=result,
                        )
                    if name == "propose_angles":
                        result = await propose_topic_angles(session, topic, limit, article_focus=article_focus, focus_keywords=focus_keywords, settings=self.settings)
                        return AgentObservation(
                            tool_name=name,
                            ok=bool(result.get("ok")),
                            summary=f"angles={len(result.get('angles') or [])}",
                            data=result,
                        )
                    if name == "build_outline":
                        result = await build_topic_outline(session, topic, limit, article_focus=article_focus, focus_keywords=focus_keywords, settings=self.settings)
                        return AgentObservation(
                            tool_name=name,
                            ok=bool(result.get("ok")),
                            summary=f"outline_blocks={result.get('outline_blocks', 0)} article_id={result.get('article_id')}",
                            data=result,
                        )
                    if name == "write_draft":
                        result = await write_topic_draft(session, self.settings, topic, limit, use_llm=use_llm, article_focus=article_focus, focus_keywords=focus_keywords)
                        return AgentObservation(
                            tool_name=name,
                            ok=bool(result.get("ok")),
                            summary=f"draft_id={result.get('draft_id')} draft_chars={result.get('draft_chars', 0)}",
                            data=result,
                        )
                    if name == "verify_claims":
                        article_id = args.get("article_id")
                        result = await verify_topic_claims(session, topic, int(article_id) if article_id else None, article_focus=article_focus, focus_keywords=focus_keywords)
                        return AgentObservation(
                            tool_name=name,
                            ok=bool(result.get("ok")),
                            summary=(
                                f"checks={len(result.get('checks') or [])} "
                                f"unsupported={result.get('unsupported_claims', 0)} "
                                f"missing_refs={result.get('missing_draft_refs', 0)} "
                                f"weak={result.get('weak_generalizations', 0)}"
                            ),
                            data=result,
                        )
                    if name == "final_article":
                        article_id = args.get("article_id")
                        result = await finalize_topic_article(session, topic, int(article_id) if article_id else None, article_focus=article_focus, focus_keywords=focus_keywords)
                        return AgentObservation(
                            tool_name=name,
                            ok=bool(result.get("ok")),
                            summary=f"final={bool(result.get('final'))} article_id={result.get('article_id')} draft_id={result.get('draft_id')}",
                            data=result,
                        )
                    if not review.get("ok"):
                        return AgentObservation(
                            tool_name=name,
                            ok=False,
                            summary=(
                                f"insufficient usable evidence: "
                                f"{review.get('usable', 0)}/{review.get('min_required', min_required)}"
                            ),
                            data=review,
                        )
                    cards = await generate_cards_from_topic(
                        session,
                        topic,
                        limit,
                        usable_message_ids=set(int(item) for item in review.get("usable_message_ids", [])),
                        include_framework=False,
                    )
                    if not cards:
                        return AgentObservation(
                            tool_name=name,
                            ok=False,
                            summary="no content cards; collect topic evidence first",
                            data={**review, "cards": 0},
                        )
                    article = await create_article(
                        session,
                        topic,
                        str(args.get("title") or f"{topic}: ragebait draft"),
                        "developers, tech leads, founders in AI/LLM/vibe coding",
                        str(args.get("angle") or instruction),
                        [card.id for card in cards[:8]],
                    )
                    outlined = await generate_outline(session, self.settings, article.id, "provocative_engineering", instruction, use_llm)
                    draft = await generate_draft(session, self.settings, article.id, "provocative_engineering", instruction, use_llm)
                    return AgentObservation(
                        tool_name=name,
                        ok=True,
                        summary=(
                            f"article draft ready cards={len(cards)} "
                            f"blocks={len(outlined.blocks)} draft_chars={len(draft.text)} "
                            f"llm={str(use_llm).lower()}"
                        ),
                        data={
                            "topic": topic,
                            "article_focus": article_focus,
                            "focus_keywords": focus_keywords,
                            "cards": len(cards),
                            "usable_evidence": review.get("usable", 0),
                            "rejected_evidence": review.get("rejected", 0),
                            "article_id": article.id,
                            "blocks": len(outlined.blocks),
                            "draft_id": draft.id,
                            "draft_chars": len(draft.text),
                            "draft_preview": draft.text[:1200],
                        },
                    )
        except Exception as exc:
            return AgentObservation(tool_name=name, ok=False, summary=f"{type(exc).__name__}: {exc}")

    def _topic_args_schema(self, required_limit: bool = True) -> dict[str, Any]:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "article_focus": {"type": "string"},
                "focus_keywords": {"type": "array", "items": {"type": "string"}},
                "taxonomy": {"type": "array", "items": {"type": "object"}},
                "limit": {"type": "integer", "minimum": 1, "maximum": 300},
                "min_required": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["topic"],
        }
        if not required_limit:
            schema["properties"].pop("min_required")
        return schema
