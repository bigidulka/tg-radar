from __future__ import annotations

import asyncio
import logging

from tg_radar.bot.analysis import AirbotAnalyzer
from tg_radar.bot.api_client import TgRadarApiClient
from tg_radar.bot.avatar import ChannelAvatarLoader
from tg_radar.bot.runner import AirbotRunner
from tg_radar.bot.store import BotRunStore
from tg_radar.bot.texts import text
from tg_radar.config import get_settings
from tg_radar.crawler import TelegramPublicCrawler


LOG = logging.getLogger(__name__)


async def run_bot() -> None:
    settings = get_settings()
    logging.basicConfig(level=logging.INFO)
    if not settings.bot_token:
        LOG.warning(text("disabled"))
        while True:
            await asyncio.sleep(3600)

    from aiogram import Bot, Dispatcher
    from tg_radar.bot.handlers import build_router
    from tg_radar.bot.middleware import ReferralMiddleware

    store = BotRunStore(settings.bot_state_path)
    store.configure_fight_rating(
        settings.bot_fight_max_elo_gap,
        settings.bot_fight_elo_gap_boost_start,
        settings.bot_fight_elo_gap_boost_max,
    )
    store.fail_active("process restarted")
    store.fail_active_fights("process restarted")
    api_client = TgRadarApiClient(settings.bot_api_base_url, settings.api_token, settings.bot_http_timeout_seconds)
    analyzer = AirbotAnalyzer(settings)
    runner = AirbotRunner(settings, store, api_client, analyzer)
    avatar_loader = ChannelAvatarLoader(TelegramPublicCrawler(settings))
    bot = Bot(settings.bot_token)
    dispatcher = Dispatcher()
    dispatcher.message.middleware(ReferralMiddleware(store))
    dispatcher.include_router(build_router(runner, avatar_loader=avatar_loader))
    try:
        await bot.delete_webhook(drop_pending_updates=True)
        await dispatcher.start_polling(bot)
    finally:
        await api_client.close()
        store.close()
        await bot.session.close()


def main() -> None:
    asyncio.run(run_bot())


if __name__ == "__main__":
    main()
