import asyncio

import pytest

from tg_radar.bot.analysis import AIRBOT_RESULT_SCHEMA_VERSION
from tg_radar.bot.runner import AirbotBusyUser, AirbotCooldown, AirbotRunner, render_result
from tg_radar.bot.store import BotRunStore
from tg_radar.config import Settings


class FakeApi:
    def __init__(self, messages=None) -> None:
        self.messages = messages or [{"message_id": "demochannel:1", "url": "https://t.me/demochannel/1", "text": "hello"}]
        self.calls = []

    async def fetch_messages(self, channel, limit):
        self.calls.append((channel, limit))
        return self.messages


class FakeAnalyzer:
    async def analyze(self, channel, messages, progress=None):
        if progress:
            await progress(1, 1)
        return {
            "schema_version": AIRBOT_RESULT_SCHEMA_VERSION,
            "channel": channel,
            "message_count": len(messages),
            "air_elo": 1550,
            "air_elo_label": "1550 ELO",
            "confidence": 0.5,
            "humorous_label": "ok",
            "summary": "ok",
            "metrics": {},
            "tactics": [],
            "evidence": [{"url": messages[0]["url"], "metric": "fomo_pressure", "snippet": "x", "comment": "x"}],
            "counter_signals": [],
            "text_generation_hooks": {},
            "disclaimer": "joke",
        }


class FailingAnalyzer:
    async def analyze(self, channel, messages, progress=None):
        raise RuntimeError("llm down")


class CountingAnalyzer:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    async def analyze(self, channel, messages, progress=None):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return {
            "schema_version": AIRBOT_RESULT_SCHEMA_VERSION,
            "channel": channel,
            "message_count": len(messages),
            "air_elo": 1550,
            "air_elo_label": "1550 ELO",
            "metrics": {},
            "tactics": [],
            "evidence": [],
            "counter_signals": [],
            "text_generation_hooks": {},
        }


@pytest.mark.asyncio
async def test_runner_executes_run_and_reuses_cache(tmp_path):
    store = BotRunStore(tmp_path / "bot.sqlite3")
    api = FakeApi()
    runner = AirbotRunner(Settings(bot_state_path=str(tmp_path / "bot.sqlite3")), store, api, FakeAnalyzer())
    stages = []

    prepared = await runner.prepare(1, 2, "@demochannel")
    completed = await runner.execute(prepared.run.run_id, stage=lambda value: append_stage(stages, value))

    assert completed.status == "completed"
    assert completed.result["air_elo"] == 1550
    assert store.raw_messages(completed.run_id)[0]["text"] == "hello"
    assert api.calls == [("demochannel", 100)]
    assert any("1/1" in stage for stage in stages)

    cached = await runner.prepare(2, 3, "https://t.me/demochannel")
    assert cached.cached is True
    assert cached.run.run_id == completed.run_id

    store.close()


@pytest.mark.asyncio
async def test_runner_blocks_user_but_queues_global_concurrency(tmp_path):
    store = BotRunStore(tmp_path / "bot.sqlite3")
    runner = AirbotRunner(Settings(bot_global_concurrency=1), store, FakeApi(), FakeAnalyzer())

    first = await runner.prepare(1, 2, "demochannel")
    with pytest.raises(AirbotBusyUser):
        await runner.prepare(1, 2, "other")
    second = await runner.prepare(2, 3, "other")

    assert store.queue_position(first.run.run_id) == 1
    assert store.queue_position(second.run.run_id) == 2

    store.close()


@pytest.mark.asyncio
async def test_runner_global_queue_limits_execute_concurrency(tmp_path):
    store = BotRunStore(tmp_path / "bot.sqlite3")
    analyzer = CountingAnalyzer()
    runner = AirbotRunner(Settings(bot_global_concurrency=5), store, FakeApi(), analyzer)
    prepared = [await runner.prepare(index, index + 100, f"demo{index}") for index in range(10)]

    completed = await asyncio.gather(*(runner.execute(item.run.run_id) for item in prepared))

    assert all(run.status == "completed" for run in completed)
    assert analyzer.max_active == 5
    store.close()


@pytest.mark.asyncio
async def test_runner_applies_cooldown_after_success(tmp_path):
    store = BotRunStore(tmp_path / "bot.sqlite3")
    runner = AirbotRunner(Settings(bot_cooldown_seconds=300), store, FakeApi(), FakeAnalyzer())

    prepared = await runner.prepare(1, 2, "demochannel")
    await runner.execute(prepared.run.run_id)

    with pytest.raises(AirbotCooldown):
        await runner.prepare(1, 2, "other")

    store.close()


@pytest.mark.asyncio
async def test_runner_failure_records_failed_run_and_retry_can_create_new(tmp_path):
    store = BotRunStore(tmp_path / "bot.sqlite3")
    runner = AirbotRunner(Settings(), store, FakeApi(), FailingAnalyzer())
    stages = []

    prepared = await runner.prepare(1, 2, "demochannel")
    failed = await runner.execute(prepared.run.run_id, stage=lambda value: append_stage(stages, value))

    assert failed.status == "failed"
    assert failed.result is None
    assert "RuntimeError" in failed.error
    assert any("llm down" in stage for stage in stages)

    retry = await runner.prepare(1, 2, "demochannel", retry=True)
    assert retry.cached is False
    assert retry.run.run_id != failed.run_id

    store.close()


def test_render_result_contains_elo_and_evidence():
    value = render_result(
        {
            "channel": "demochannel",
            "air_elo": 1200,
            "air_elo_label": "1200 ELO",
            "humorous_label": "low",
            "evidence": [{"url": "https://t.me/demochannel/1", "snippet": "x"}],
            "disclaimer": "joke",
        }
    )

    assert '"channel": "demochannel"' in value
    assert '"air_elo_label": "1200 ELO"' in value
    assert "https://t.me/demochannel/1" in value


async def append_stage(values, value):
    values.append(value)
