import asyncio
import json

import pytest

from tg_radar.bot.analysis import AIRBOT_RESULT_SCHEMA_VERSION
from tg_radar.bot.fight import AIRBOT_FIGHT_SCHEMA_VERSION, AirbotFightArena, fight_result_html, fight_result_plain
from tg_radar.bot.fight_runner import AirbotFightEloGapTooHigh, AirbotFightRunner
from tg_radar.bot.store import BotRunStore
from tg_radar.config import Settings


class FakeFightTransport:
    def __init__(self) -> None:
        self.messages = []

    async def complete_json(self, messages):
        self.messages = messages
        return json.dumps(
            {
                "rounds": [
                    {
                        "round": 1,
                        "title": "FOMO gust",
                        "winner": "left",
                        "decisive": False,
                        "snapshot": "left opens a loop",
                        "logic": "fomo_pressure wins",
                    },
                    {
                        "round": 2,
                        "title": "Promise storm",
                        "winner": "right",
                        "decisive": False,
                        "snapshot": "right counters",
                        "logic": "product_warmup wins",
                    },
                    {
                        "round": 3,
                        "title": "Final wind",
                        "winner": "left",
                        "decisive": True,
                        "snapshot": "left spins faster",
                        "logic": "detail score plus roll",
                    },
                ],
                "final": {
                    "winner": "left",
                    "decisive_round": 3,
                    "method": "wind ko",
                    "verdict": "left wins",
                    "why": "better topic match",
                },
                "share_line": "share this wind",
            }
        )


class CountingFightArena:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.calls = []

    async def fight(self, left, right, topic):
        self.calls.append((left.channel, right.channel, topic))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return {
            "schema_version": AIRBOT_FIGHT_SCHEMA_VERSION,
            "fight_id": "",
            "topic": topic,
            "fighters": [
                {"side": "left", "channel": left.channel, "air_elo": 1200, "rank": "rank", "detail_score": 70, "random_rolls": {}},
                {"side": "right", "channel": right.channel, "air_elo": 1300, "rank": "rank", "detail_score": 60, "random_rolls": {}},
            ],
            "rounds": [{"number": 1, "title": "r", "winner": "left", "winner_channel": left.channel, "decisive": True, "snapshot": "s", "logic": "l"}],
            "final": {"winner": "left", "channel": left.channel, "decisive_round": 1, "method": "m", "verdict": "v", "why": "w"},
            "share_line": "share",
        }


@pytest.mark.asyncio
async def test_airbot_fight_uses_leaderboard_results_and_renders(tmp_path):
    store = BotRunStore(tmp_path / "bot.sqlite3")
    left = store.create_run(user_id=1, chat_id=2, channel="left")
    right = store.create_run(user_id=2, chat_id=2, channel="right")
    store.complete_run(left.run_id, sample_result("left", 1200, 80))
    store.complete_run(right.run_id, sample_result("right", 2600, 70))
    left = store.get_run(left.run_id)
    right = store.get_run(right.run_id)
    transport = FakeFightTransport()
    arena = AirbotFightArena(Settings(), transport=transport)

    result = await arena.fight(left, right, "who keeps the audience hotter")

    payload = json.loads(transport.messages[1]["content"])
    assert payload["topic"] == "who keeps the audience hotter"
    assert payload["fighters"][0]["channel"] == "left"
    assert payload["fighters"][0]["detail_score"] > 0
    assert payload["fighters"][0]["metrics"][0]["code"] == "fomo_pressure"
    assert 1 <= payload["fighters"][0]["random_rolls"]["chaos"] <= 100
    assert result["final"]["channel"] == "left"
    assert result["rounds"][2]["decisive"] is True

    html = fight_result_html(result)
    plain = fight_result_plain(result)
    assert '<a href="https://t.me/left">@left</a>' in html
    assert "<details open>" in html
    assert "<h2>Финал</h2>" in html
    assert "left wins" in plain
    store.close()


@pytest.mark.asyncio
async def test_airbot_fight_runner_limits_concurrency_and_reuses_cache(tmp_path):
    store = BotRunStore(tmp_path / "bot.sqlite3")
    left = store.create_run(user_id=1, chat_id=2, channel="left")
    right = store.create_run(user_id=2, chat_id=2, channel="right")
    store.complete_run(left.run_id, sample_result("left", 1200, 80))
    store.complete_run(right.run_id, sample_result("right", 2600, 70))
    left = store.get_run(left.run_id)
    right = store.get_run(right.run_id)
    arena = CountingFightArena()
    runner = AirbotFightRunner(Settings(bot_fight_global_concurrency=5), store, arena)

    prepared = [await runner.prepare(index + 10, index + 20, left, right, f"topic {index}") for index in range(10)]
    completed = await asyncio.gather(*(runner.execute(item.fight.fight_id) for item in prepared))

    assert all(fight.status == "completed" for fight in completed)
    assert arena.max_active == 5

    cached = await runner.prepare(99, 100, left, right, "topic 0")
    assert cached.cached is True
    assert cached.fight.fight_id == completed[0].fight_id
    store.close()


@pytest.mark.asyncio
async def test_airbot_fight_runner_blocks_too_large_elo_gap(tmp_path):
    store = BotRunStore(tmp_path / "bot.sqlite3")
    left = store.create_run(user_id=1, chat_id=2, channel="left")
    right = store.create_run(user_id=2, chat_id=2, channel="right")
    store.complete_run(left.run_id, sample_result("left", 100, 30))
    store.complete_run(right.run_id, sample_result("right", 3900, 90))
    runner = AirbotFightRunner(Settings(bot_fight_max_elo_gap=1000), store, CountingFightArena())

    with pytest.raises(AirbotFightEloGapTooHigh) as error:
        await runner.prepare(9, 10, store.get_run(left.run_id), store.get_run(right.run_id), "topic")

    assert error.value.gap == 3800
    assert error.value.max_gap == 1000
    assert store._conn.execute("SELECT count(*) FROM bot_fight_runs").fetchone()[0] == 0
    store.close()


def sample_result(channel, elo, fomo):
    return {
        "schema_version": AIRBOT_RESULT_SCHEMA_VERSION,
        "channel": channel,
        "air_elo": elo,
        "summary": "summary",
        "niche_tags": ["crypto"],
        "core_traits": ["fomo"],
        "metrics": {
            "fomo_pressure": {"score": fomo, "why": "pushes urgency"},
            "product_warmup": {"score": 60, "why": "warms product"},
        },
        "tactics": [{"code": "fomo", "name": "urgency", "intensity": fomo, "hint": "keeps attention"}],
        "evidence": [{"url": f"https://t.me/{channel}/1", "snippet": "soon", "comment": "open loop"}],
        "fun_metrics": [{"label": "fans", "display": "10 pcs"}],
        "meme_profile": {"rank_name": "rank"},
    }
