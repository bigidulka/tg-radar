import json

import pytest

from tg_radar.bot.analysis import AIRBOT_RESULT_SCHEMA_VERSION, AirbotAnalyzer, AirbotAnalysisError
from tg_radar.bot.ratings import air_elo, elo_rank
from tg_radar.config import Settings

METRIC_NAMES = (
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


class FakeTransport:
    def __init__(self) -> None:
        self.calls = []

    async def complete_json(self, messages):
        self.calls.append(messages)
        if len(self.calls) < 3:
            return json.dumps(
                {
                    "message_count": 3,
                    "niche_tags": ["demo"],
                    "core_traits": ["fomo"],
                    "metric_scores": {name: 60 for name in METRIC_NAMES},
                    "detected_tactics": [{"code": "fomo_pressure", "strength": 60, "hint": "fomo", "evidence_urls": ["https://t.me/demochannel/1"]}],
                    "notable_quotes": [{"url": "https://t.me/demochannel/1", "metric": "fomo_pressure", "snippet": "x", "why": "x"}],
                }
            )
        return json.dumps(
            {
                "schema_version": AIRBOT_RESULT_SCHEMA_VERSION,
                "channel": "demochannel",
                "message_count": 6,
                "niche_tags": ["demo"],
                "core_traits": ["fomo"],
                "confidence": 0.7,
                "humorous_label": "high",
                "summary": "high",
                "metrics": {name: {"score": 60, "why": "x"} for name in METRIC_NAMES},
                "tactics": [{"code": "fomo_pressure", "name": "fomo", "intensity": 60, "hint": "x", "why_it_keeps_attention": "x", "evidence_urls": ["https://t.me/demochannel/1"]}],
                "evidence": [{"url": "https://t.me/demochannel/1", "metric": "fomo_pressure", "snippet": "x", "comment": "x"}],
                "counter_signals": [],
                "text_generation_hooks": {"roast_angles": ["x"], "running_jokes": [], "safe_phrases": []},
                "meme_profile": {"rank": 6, "rank_name": "microburst", "one_liner": "x"},
                "disclaimer": "joke",
            }
        )


@pytest.mark.asyncio
async def test_airbot_analyzer_runs_chunked_map_reduce():
    transport = FakeTransport()
    analyzer = AirbotAnalyzer(Settings(bot_llm_chunk_size=5), transport=transport)
    progress = []

    result = await analyzer.analyze(
        "demochannel",
        [
            {"message_id": "1", "text": "a", "url": "https://t.me/demochannel/1"},
            {"message_id": "2", "text": "b", "url": "https://t.me/demochannel/2"},
            {"message_id": "3", "text": "c", "url": "https://t.me/demochannel/3"},
            {"message_id": "4", "text": "d", "url": "https://t.me/demochannel/4"},
            {"message_id": "5", "text": "e", "url": "https://t.me/demochannel/5"},
            {"message_id": "6", "text": "f", "url": "https://t.me/demochannel/6"},
        ],
        progress=lambda done, total: progress_append(progress, done, total),
    )

    assert len(transport.calls) == 2
    assert progress == [(1, 2), (2, 2)]
    assert "score" not in result
    assert "score_label" not in result
    assert result["message_count"] == 6
    assert result["schema_version"] == AIRBOT_RESULT_SCHEMA_VERSION
    assert result["air_elo"] == 3281
    assert result["air_elo_label"] == "3281 ELO"
    assert result["air_rank_measure"]
    assert result["fun_metrics"]
    assert result["fun_metrics"][0]["key"] == "air_kwh"
    assert result["niche_tags"] == ["demo"]
    assert result["core_traits"] == ["fomo"]
    assert result["metrics"]["fomo_pressure"]["score"] == 60
    assert "toxic_aggression" in result["metrics"]
    assert "shame_guilt_pressure" in result["metrics"]
    assert "monetization_intensity" in result["metrics"]
    assert "evidence_specificity_reverse" in result["metrics"]
    assert result["meme_profile"]["air_elo"] == 3281
    assert result["meme_profile"]["rank_measure"] == result["air_rank_measure"]
    assert result["tactics"][0]["code"] == "fomo_pressure"
    assert result["evidence"][0]["url"] == "https://t.me/demochannel/1"


def test_airbot_merge_payload_is_compacted():
    analyzer = AirbotAnalyzer(Settings(), transport=FakeTransport())
    compacted = analyzer._compact_chunk_result(
        {
            "message_count": 40,
            "niche_tags": ["crypto", "trading", "x", "y", "z"],
            "core_traits": ["fomo", "rage", "click", "money", "tribe", "extra"],
            "metric_scores": {"fomo_pressure": 80, "ragebait": {"score": 50, "why": "long"}},
            "detected_tactics": [
                {"code": "fomo_pressure", "strength": 90, "hint": "x" * 500, "evidence_urls": ["u1", "u2", "u3", "u4"]},
            ],
            "notable_quotes": [{"url": "u", "snippet": "s" * 500, "why": "w" * 500}],
            "notes": ["n" * 500],
            "unused_blob": "z" * 10000,
        }
    )

    dumped = json.dumps(compacted, ensure_ascii=False)
    assert "unused_blob" not in dumped
    assert len(dumped) < 2000
    assert compacted["metric_scores"] == {"fomo_pressure": 80, "ragebait": 50}


def test_air_elo_is_trait_score_rating():
    high = {name: {"score": 78, "why": "x"} for name in METRIC_NAMES}
    low = {name: {"score": 25, "why": "x"} for name in METRIC_NAMES}

    assert air_elo(high) >= 3500
    assert air_elo(high) > air_elo(low)


def test_air_elo_has_hard_cap_and_no_mid_high_autocap():
    maxed = {name: {"score": 100, "why": "x"} for name in METRIC_NAMES}
    high = {name: {"score": 70, "why": "x"} for name in METRIC_NAMES}

    assert air_elo(maxed) == 4000
    assert air_elo(high) < 4000


def test_air_elo_uses_composite_trait_bonuses():
    baseline = {name: {"score": 30, "why": "x"} for name in METRIC_NAMES}
    composite = {name: {"score": 30, "why": "x"} for name in METRIC_NAMES}
    for name in ("fomo_pressure", "product_warmup", "monetization_intensity"):
        composite[name] = {"score": 90, "why": "x"}

    assert air_elo(composite) > air_elo(baseline)


def test_elo_rank_has_weather_measure():
    rank = elo_rank(3936)

    assert rank["unit_value"] == 1000
    assert rank["unit"]
    assert rank["emoji"] in {"🫧", "🌬️", "💨", "🌫️", "🌪️", "🌀"}


@pytest.mark.asyncio
async def test_airbot_analyzer_requires_messages():
    analyzer = AirbotAnalyzer(Settings(), transport=FakeTransport())

    with pytest.raises(AirbotAnalysisError):
        await analyzer.analyze("demochannel", [])


async def progress_append(values, done, total):
    values.append((done, total))
