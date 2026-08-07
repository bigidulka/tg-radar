from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from tg_radar.bot.fight import AIRBOT_FIGHT_SCHEMA_VERSION, AirbotFightArena, AirbotFightError, fight_failed_public_error
from tg_radar.bot.store import BotFightRun, BotRun, BotRunStore
from tg_radar.bot.texts import text
from tg_radar.config import Settings


FightStageCallback = Callable[[str], Awaitable[None]]


class AirbotFightRunnerError(RuntimeError):
    pass


class AirbotFightBusyUser(AirbotFightRunnerError):
    pass


class AirbotFightEloGapTooHigh(AirbotFightRunnerError):
    def __init__(self, left_elo: int, right_elo: int, max_gap: int) -> None:
        super().__init__("elo gap is too high")
        self.left_elo = left_elo
        self.right_elo = right_elo
        self.gap = abs(left_elo - right_elo)
        self.max_gap = max_gap


@dataclass(frozen=True)
class PreparedFight:
    fight: BotFightRun
    cached: bool = False


class AirbotFightRunner:
    def __init__(self, settings: Settings, store: BotRunStore, arena: AirbotFightArena) -> None:
        self.settings = settings
        self.store = store
        self.arena = arena
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(settings.bot_fight_global_concurrency)

    async def prepare(self, user_id: int, chat_id: int, left: BotRun, right: BotRun, topic: str) -> PreparedFight:
        async with self._lock:
            self.store.cleanup(self.settings.bot_history_ttl_seconds)
            left = self.store.run_with_fight_rating(left)
            right = self.store.run_with_fight_rating(right)
            self._check_elo_gap(left, right)
            cached = self.store.cached_fight(
                left.channel,
                right.channel,
                topic,
                self.settings.bot_fight_cache_ttl_seconds,
                schema_version=AIRBOT_FIGHT_SCHEMA_VERSION,
            )
            if cached:
                return PreparedFight(cached, cached=True)
            if self.store.active_fight_count_for_user(user_id) >= self.settings.bot_fight_user_concurrency:
                raise AirbotFightBusyUser("user is busy")
            return PreparedFight(self.store.create_fight_run(user_id, chat_id, left, right, topic))

    def _check_elo_gap(self, left: BotRun, right: BotRun) -> None:
        left_elo = self._run_elo(left)
        right_elo = self._run_elo(right)
        max_gap = self.settings.bot_fight_max_elo_gap
        if max_gap > 0 and abs(left_elo - right_elo) > max_gap:
            raise AirbotFightEloGapTooHigh(left_elo, right_elo, max_gap)

    def _run_elo(self, run: BotRun) -> int:
        result = run.result or {}
        for key in ("battle_elo", "air_elo", "score"):
            try:
                return int(result.get(key))
            except Exception:
                continue
        return 0

    async def execute(self, fight_id: str, stage: FightStageCallback | None = None) -> BotFightRun:
        async with self._semaphore:
            fight = self.store.get_fight_run(fight_id)
            if fight is None:
                raise AirbotFightRunnerError("fight run not found")
            try:
                fight = self.store.set_fight_status(fight.fight_id, "running")
                if stage:
                    await stage(text("fight.running", left=fight.left_channel, right=fight.right_channel, topic=fight.topic))
                left = self.store.get_run(fight.left_run_id)
                right = self.store.get_run(fight.right_run_id)
                if not left or not right or not left.result or not right.result:
                    raise AirbotFightRunnerError("fighter result is missing")
                left = self.store.run_with_fight_rating(left)
                right = self.store.run_with_fight_rating(right)
                result = await self.arena.fight(left, right, fight.topic)
                result["fight_id"] = fight.fight_id
                result["left_channel"] = fight.left_channel
                result["right_channel"] = fight.right_channel
                return self.store.complete_fight_run(fight.fight_id, result)
            except Exception as exc:
                failed = self.store.fail_fight_run(fight.fight_id, f"{type(exc).__name__}: {exc}")
                if stage:
                    await stage(text("fight.failed", error=fight_failed_public_error(exc)))
                return failed
