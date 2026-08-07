from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import json

from tg_radar.bot.analysis import AIRBOT_RESULT_SCHEMA_VERSION, AirbotAnalysisError, AirbotAnalyzer
from tg_radar.bot.api_client import AirbotApiError, TgRadarApiClient
from tg_radar.bot.store import BotRun, BotRunStore
from tg_radar.bot.texts import text
from tg_radar.config import Settings
from tg_radar.text import clean_username


StageCallback = Callable[[str], Awaitable[None]]


class AirbotRunnerError(RuntimeError):
    pass


class AirbotBusyUser(AirbotRunnerError):
    pass


class AirbotInvalidChannel(AirbotRunnerError):
    pass


class AirbotCooldown(AirbotRunnerError):
    def __init__(self, seconds: int) -> None:
        super().__init__("cooldown")
        self.seconds = seconds


@dataclass(frozen=True)
class PreparedRun:
    run: BotRun
    cached: bool = False


class AirbotRunner:
    def __init__(
        self,
        settings: Settings,
        store: BotRunStore,
        api_client: TgRadarApiClient,
        analyzer: AirbotAnalyzer,
    ) -> None:
        self.settings = settings
        self.store = store
        self.api_client = api_client
        self.analyzer = analyzer
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(settings.bot_global_concurrency)

    async def prepare(self, user_id: int, chat_id: int, raw_channel: str, retry: bool = False) -> PreparedRun:
        channel = clean_username(raw_channel)
        if not channel:
            raise AirbotInvalidChannel("invalid channel")
        async with self._lock:
            self.store.cleanup(self.settings.bot_history_ttl_seconds)
            if not retry:
                cached = self.store.cached_completed_for_channel(
                    channel,
                    self.settings.bot_cache_ttl_seconds,
                    schema_version=AIRBOT_RESULT_SCHEMA_VERSION,
                )
                if cached:
                    return PreparedRun(cached, cached=True)
                cooldown = self.store.user_success_in_cooldown(user_id, self.settings.bot_cooldown_seconds)
                if cooldown:
                    raise AirbotCooldown(self._remaining_cooldown(cooldown))
            if self.store.active_count_for_user(user_id) >= self.settings.bot_user_concurrency:
                raise AirbotBusyUser("user is busy")
            return PreparedRun(self.store.create_run(user_id, chat_id, channel))

    async def execute(self, run_id: str, stage: StageCallback | None = None) -> BotRun:
        async with self._semaphore:
            run = self.store.get_run(run_id)
            if run is None:
                raise AirbotRunnerError("run not found")
            try:
                await self._stage(run, "fetching", stage)
                limit = min(self.settings.bot_default_limit, self.settings.bot_max_limit)
                messages = await self.api_client.fetch_messages(run.channel, limit)
                if not messages:
                    raise AirbotRunnerError("no messages")
                self.store.set_raw_messages(run.run_id, messages)
                await self._stage(run, "analyzing", stage)

                async def progress(done: int, total: int) -> None:
                    if stage:
                        await stage(text("analyzing", done=done, total=total))

                result = await self.analyzer.analyze(run.channel, messages, progress=progress)
                completed = self.store.complete_run(run.run_id, result)
                return completed
            except Exception as exc:
                failed = self.store.fail_run(run.run_id, f"{type(exc).__name__}: {exc}")
                if stage:
                    await stage(text("failed", error=self._public_error(exc)))
                return failed

    async def _stage(self, run: BotRun, status: str, stage: StageCallback | None) -> None:
        self.store.set_status(run.run_id, status)
        if not stage:
            return
        if status == "fetching":
            await stage(text("fetching", channel=run.channel))
        elif status == "analyzing":
            await stage(text("analyzing", done=0, total=1))

    def _remaining_cooldown(self, run: BotRun) -> int:
        elapsed = datetime.now(timezone.utc) - run.created_at
        return max(1, int(self.settings.bot_cooldown_seconds - elapsed.total_seconds()))

    def _public_error(self, exc: Exception) -> str:
        if isinstance(exc, AirbotApiError):
            value = str(exc)
            if "ReadTimeout" in value or "Timeout" in value:
                return text("fetch_timeout")
            return value
        if isinstance(exc, AirbotAnalysisError):
            value = str(exc)
            if value == "llm is not configured":
                return text("llm_missing")
            if value.startswith("llm request failed"):
                return text("llm_unavailable")
            if value == "no messages":
                return text("no_messages")
            if value.startswith("llm returned"):
                return text("llm_bad_json")
            return value
        if isinstance(exc, AirbotRunnerError) and str(exc) == "no messages":
            return text("no_messages")
        return str(exc) or type(exc).__name__


def render_result(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2)
