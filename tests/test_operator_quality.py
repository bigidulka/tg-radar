from datetime import datetime, timedelta, timezone
import json
from unittest.mock import AsyncMock, Mock

import pytest

from tg_radar.config import Settings
from tg_radar.db import Channel, maybe_update_operator_quality
from tg_radar.operator_quality import OPERATOR_QUALITY_SCHEMA_VERSION, OperatorQualityAnalyzer, OperatorQualityError
from tg_radar.schemas import ParsedMessage


def _settings(**overrides) -> Settings:
    return Settings(llm_base_url="http://localhost:8317/v1", llm_api_key="sk-test", **overrides)


def _message(index: int, text: str = "hello", posted_at: datetime | None = None) -> ParsedMessage:
    return ParsedMessage(
        channel_username="demochannel",
        tg_msg_id=index,
        url=f"https://t.me/demochannel/{index}",
        posted_at=posted_at,
        text=text,
        content_hash=f"hash-{index}",
    )


class FakeTransport:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    async def complete_json(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_operator_quality_analyzer_normalizes_well_formed_response():
    payload = {
        "niche_category": "novel",
        "income_authenticity": "likely_real_income",
        "niche_tags": ["onchain-mm"],
        "confidence": 0.82,
        "evidence": [{"url": "https://t.me/demochannel/1", "snippet": "x", "why": "y"}],
        "notes": ["consistent infra mentions"],
    }
    transport = FakeTransport([json.dumps(payload)])
    analyzer = OperatorQualityAnalyzer(_settings(), transport=transport)

    result = await analyzer.analyze("demochannel", [_message(1)])

    assert len(transport.calls) == 1
    assert result["schema_version"] == OPERATOR_QUALITY_SCHEMA_VERSION
    assert result["channel"] == "demochannel"
    assert result["message_count"] == 1
    assert result["niche_category"] == "novel"
    assert result["income_authenticity"] == "likely_real_income"
    assert result["niche_tags"] == ["onchain-mm"]
    assert result["confidence"] == 0.82
    assert result["evidence"] == [{"url": "https://t.me/demochannel/1", "snippet": "x", "why": "y"}]
    assert result["notes"] == ["consistent infra mentions"]


@pytest.mark.asyncio
async def test_operator_quality_analyzer_tolerates_code_fenced_junk_json():
    payload = {"niche_category": "saturated", "income_authenticity": "uncertain"}
    wrapped = f"```json\nhere is the result:\n{json.dumps(payload)}\n```"
    transport = FakeTransport([wrapped])
    analyzer = OperatorQualityAnalyzer(_settings(), transport=transport)

    result = await analyzer.analyze("demochannel", [_message(1)])

    assert result["niche_category"] == "saturated"
    assert result["income_authenticity"] == "uncertain"


@pytest.mark.asyncio
async def test_operator_quality_analyzer_clamps_invalid_enum_values():
    payload = {"niche_category": "totally-not-a-category", "income_authenticity": "definitely-fake"}
    transport = FakeTransport([json.dumps(payload)])
    analyzer = OperatorQualityAnalyzer(_settings(), transport=transport)

    result = await analyzer.analyze("demochannel", [_message(1)])

    assert result["niche_category"] == "unclear"
    assert result["income_authenticity"] == "uncertain"


@pytest.mark.asyncio
async def test_operator_quality_analyzer_clamps_confidence_out_of_range():
    transport = FakeTransport([json.dumps({"confidence": 5.0})])
    analyzer = OperatorQualityAnalyzer(_settings(), transport=transport)
    result = await analyzer.analyze("demochannel", [_message(1)])
    assert result["confidence"] == 1.0

    transport = FakeTransport([json.dumps({"confidence": -3.0})])
    analyzer = OperatorQualityAnalyzer(_settings(), transport=transport)
    result = await analyzer.analyze("demochannel", [_message(1)])
    assert result["confidence"] == 0.0


@pytest.mark.asyncio
async def test_operator_quality_analyzer_caps_oversized_lists():
    payload = {
        "niche_tags": [f"tag-{i}" for i in range(20)],
        "evidence": [{"url": f"https://t.me/demochannel/{i}", "snippet": "x", "why": "y"} for i in range(20)],
        "notes": [f"note-{i}" for i in range(20)],
    }
    transport = FakeTransport([json.dumps(payload)])
    analyzer = OperatorQualityAnalyzer(_settings(), transport=transport)

    result = await analyzer.analyze("demochannel", [_message(1)])

    assert len(result["niche_tags"]) == 8
    assert len(result["evidence"]) == 10
    assert len(result["notes"]) == 8


@pytest.mark.asyncio
async def test_operator_quality_analyzer_raises_operator_quality_error_on_truncated_json():
    truncated = '{"niche_category": "novel", "evidence": [{"url": "https://t.me/x/1"}'
    transport = FakeTransport([truncated])
    analyzer = OperatorQualityAnalyzer(_settings(), transport=transport)

    with pytest.raises(OperatorQualityError):
        await analyzer.analyze("demochannel", [_message(1)])


@pytest.mark.asyncio
async def test_operator_quality_analyzer_requires_messages():
    analyzer = OperatorQualityAnalyzer(_settings(), transport=FakeTransport([]))

    with pytest.raises(OperatorQualityError):
        await analyzer.analyze("demochannel", [])


@pytest.mark.asyncio
async def test_operator_quality_analyzer_requires_llm_configuration():
    analyzer = OperatorQualityAnalyzer(Settings(llm_base_url=None, llm_api_key=None))

    with pytest.raises(OperatorQualityError):
        await analyzer.analyze("demochannel", [_message(1)])


@pytest.mark.asyncio
async def test_maybe_update_operator_quality_noop_when_disabled():
    session = AsyncMock()
    analyzer = AsyncMock()
    channel = Channel(username="demochannel")

    await maybe_update_operator_quality(session, channel, [_message(1)], _settings(operator_quality_enabled=False), analyzer=analyzer)

    analyzer.analyze.assert_not_called()
    session.execute.assert_not_called()


@pytest.mark.asyncio
async def test_maybe_update_operator_quality_noop_below_min_messages():
    session = AsyncMock()
    analyzer = AsyncMock()
    channel = Channel(username="demochannel")

    await maybe_update_operator_quality(
        session,
        channel,
        [_message(1)],
        _settings(operator_quality_enabled=True, operator_quality_min_messages=5),
        analyzer=analyzer,
    )

    analyzer.analyze.assert_not_called()
    session.execute.assert_not_called()


@pytest.mark.asyncio
async def test_maybe_update_operator_quality_noop_within_recheck_window():
    session = AsyncMock()
    analyzer = AsyncMock()
    channel = Channel(username="demochannel", operator_niche_updated_at=datetime.now(timezone.utc) - timedelta(days=1))

    await maybe_update_operator_quality(
        session,
        channel,
        [_message(1)],
        _settings(operator_quality_enabled=True, operator_quality_min_messages=1, operator_quality_recheck_interval_days=7),
        analyzer=analyzer,
    )

    analyzer.analyze.assert_not_called()
    session.execute.assert_not_called()


@pytest.mark.asyncio
async def test_maybe_update_operator_quality_writes_all_columns_when_enabled_and_fresh():
    session = AsyncMock()
    analyzer = AsyncMock()
    analyzer.analyze.return_value = {
        "niche_category": "novel",
        "income_authenticity": "likely_real_income",
        "niche_tags": ["mev"],
        "confidence": 0.9,
        "evidence": [{"url": "https://t.me/demochannel/1", "snippet": "x", "why": "y"}],
        "notes": ["note"],
    }
    channel = Channel(username="demochannel")
    messages = [_message(1)]

    await maybe_update_operator_quality(
        session,
        channel,
        messages,
        _settings(operator_quality_enabled=True, operator_quality_min_messages=1),
        analyzer=analyzer,
    )

    analyzer.analyze.assert_awaited_once_with("demochannel", messages)
    session.execute.assert_awaited_once()
    execute_params = session.execute.call_args.args[1]
    assert execute_params == {
        "username": "demochannel",
        "niche_category": "novel",
        "niche_tags": ["mev"],
        "niche_confidence": 0.9,
    }
    assert channel.operator_niche_category == "novel"
    assert channel.operator_income_authenticity == "likely_real_income"
    assert channel.operator_niche_tags == ["mev"]
    assert channel.operator_niche_confidence == 0.9
    assert channel.operator_niche_evidence == {"items": [{"url": "https://t.me/demochannel/1", "snippet": "x", "why": "y"}]}
    assert channel.operator_niche_notes == ["note"]
    assert channel.operator_niche_updated_at is not None


@pytest.mark.asyncio
async def test_maybe_update_operator_quality_swallows_analyzer_errors():
    session = AsyncMock()
    analyzer = AsyncMock()
    analyzer.analyze.side_effect = OperatorQualityError("llm request failed")
    channel = Channel(username="demochannel")

    await maybe_update_operator_quality(
        session,
        channel,
        [_message(1)],
        _settings(operator_quality_enabled=True, operator_quality_min_messages=1),
        analyzer=analyzer,
    )

    session.execute.assert_not_called()
    assert channel.operator_niche_category is None


@pytest.mark.asyncio
async def test_maybe_update_operator_quality_swallows_non_operator_quality_errors():
    session = AsyncMock()
    analyzer = AsyncMock()
    analyzer.analyze.side_effect = ValueError("truncated json")
    channel = Channel(username="demochannel")

    await maybe_update_operator_quality(
        session,
        channel,
        [_message(1)],
        _settings(operator_quality_enabled=True, operator_quality_min_messages=1),
        analyzer=analyzer,
    )

    session.execute.assert_not_called()
    assert channel.operator_niche_category is None


@pytest.mark.asyncio
async def test_maybe_update_operator_quality_disabled_never_constructs_analyzer(monkeypatch):
    session = AsyncMock()
    channel = Channel(username="demochannel")
    constructed = Mock()
    monkeypatch.setattr("tg_radar.db.OperatorQualityAnalyzer", constructed)

    await maybe_update_operator_quality(
        session,
        channel,
        [_message(1)],
        _settings(operator_quality_enabled=False),
    )

    constructed.assert_not_called()
    session.execute.assert_not_called()
