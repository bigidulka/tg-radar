from __future__ import annotations

import asyncio
from html import escape
from pathlib import Path

from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, FSInputFile, InlineKeyboardButton, InlineKeyboardMarkup, Message

from tg_radar.bot.analysis import AIRBOT_RESULT_SCHEMA_VERSION
from tg_radar.bot.avatar import ChannelAvatarLoader
from tg_radar.bot.cards import ask_channel_html, leaderboard_html, leaderboard_text, result_screen_plain, start_html, start_plain
from tg_radar.bot.fight import (
    AirbotFightArena,
    fight_match_button_label,
    fight_matches_html,
    fight_matches_plain,
    fight_result_html,
    fight_result_plain,
)
from tg_radar.bot.fight_runner import AirbotFightBusyUser, AirbotFightEloGapTooHigh, AirbotFightRunner
from tg_radar.bot.fight_images import AirbotFightImageGenerator
from tg_radar.bot.image_generator import AirbotImageGenerator
from tg_radar.bot.renderer import AirbotTelegramRenderer
from tg_radar.bot.runner import AirbotBusyUser, AirbotCooldown, AirbotInvalidChannel, AirbotRunner
from tg_radar.bot.store import BotFightRun, BotRun
from tg_radar.bot.texts import text
from tg_radar.text import clean_username


PAGE_SIZE = 5
HOME_CALLBACK = "v1:home"
MENU_CALLBACK = "v1:menu"
ANALYZE_CALLBACK = "v1:analyze"
LAST_CALLBACK = "v1:last"
LEADERBOARD_CALLBACK = "v1:lb"
LEADERBOARD_SEARCH_CALLBACK = "v1:lbs"
IMAGE_CALLBACK = "v1:img"
FIGHT_CALLBACK = "v1:fight"
FIGHT_PICK_CALLBACK = "v1:ftp"
FIGHT_PICK_RUN_CALLBACK = "v1:ftr"
FIGHT_PAGE_CALLBACK = "v1:ftpg"
FIGHT_SEARCH_CALLBACK = "v1:fts"
FIGHT_VIEW_CALLBACK = "v1:fv"
FIGHT_MATCHES_CALLBACK = "v1:fms"


def build_router(runner: AirbotRunner, avatar_loader: ChannelAvatarLoader | None = None) -> Router:
    router = Router()
    renderer = AirbotTelegramRenderer(avatar_loader=avatar_loader)
    image_generator = AirbotImageGenerator(runner.settings, avatar_loader=avatar_loader)
    fight_image_generator = AirbotFightImageGenerator(runner.settings, image_generator)
    fight_arena = AirbotFightArena(runner.settings)
    fight_runner = AirbotFightRunner(runner.settings, runner.store, fight_arena)
    search_modes: dict[tuple[int, int], str] = {}

    @router.message(CommandStart())
    async def start(message: Message) -> None:
        clear_search_mode_message(message, search_modes)
        await show_home_from_message(message, runner, renderer)

    @router.message(Command("help"))
    async def help_message(message: Message) -> None:
        clear_search_mode_message(message, search_modes)
        await show_home_from_message(message, runner, renderer)

    @router.message(Command("analyze"))
    async def analyze_command(message: Message) -> None:
        channel = channel_from_text(message.text or "")
        if not channel:
            await show_ask_from_message(message, runner, renderer)
            return
        await enqueue(message, runner, renderer, channel)

    @router.message(Command("fight"))
    async def fight_command(message: Message) -> None:
        clear_search_mode_message(message, search_modes)
        await show_fight_from_message(message, runner, renderer)

    @router.message(F.text)
    async def channel_message(message: Message) -> None:
        user_id = message.from_user.id if message.from_user else 0
        search_mode = search_modes.pop((user_id, message.chat.id), None)
        if search_mode:
            await handle_search_text(message, runner, renderer, search_mode)
            return
        if runner.store.awaiting_fight_topic(user_id, message.chat.id):
            await start_fight_from_topic(message, runner, renderer, fight_runner, fight_image_generator)
            return
        channel = channel_from_text(message.text or "")
        if not channel:
            if runner.store.awaiting_channel(user_id, message.chat.id):
                await render_saved_or_new(
                    message,
                    runner,
                    renderer,
                    text("screens.invalid_html"),
                    text("invalid_channel"),
                    back_menu(),
                    awaiting_channel=True,
                )
            return
        await enqueue(message, runner, renderer, channel)

    @router.callback_query(lambda callback: callback.data in {"home", HOME_CALLBACK})
    async def home_callback(callback: CallbackQuery) -> None:
        await callback.answer()
        await show_home_from_callback(callback, runner, renderer)

    @router.callback_query(lambda callback: callback.data in {"menu", MENU_CALLBACK})
    async def menu_callback(callback: CallbackQuery) -> None:
        await callback.answer()
        await show_home_from_callback(callback, runner, renderer)

    @router.callback_query(lambda callback: callback.data in {"analyze", ANALYZE_CALLBACK})
    async def analyze_callback(callback: CallbackQuery) -> None:
        await callback.answer()
        if user_has_active_run(callback, runner):
            await edit_callback_screen(
                callback,
                runner,
                renderer,
                text("screens.busy_html"),
                text("busy_user"),
                back_menu(),
                track_screen=False,
            )
            return
        await show_ask_from_callback(callback, runner, renderer)

    @router.callback_query(lambda callback: callback.data in {"last", LAST_CALLBACK})
    async def last_callback(callback: CallbackQuery) -> None:
        await callback.answer()
        if not callback.message or not callback.from_user:
            return
        run = runner.store.latest_completed_for_user(callback.from_user.id, schema_version=AIRBOT_RESULT_SCHEMA_VERSION)
        if not run or not run.result:
            await edit_callback_screen(
                callback,
                runner,
                renderer,
                text("screens.last_missing_html"),
                text("last_missing"),
                back_menu(),
                track_screen=not user_has_active_run(callback, runner),
            )
            return
        await edit_result(
            callback.message,
            runner,
            renderer,
            run,
            callback.from_user.id,
            result_menu(HOME_CALLBACK, run.run_id),
            track_screen=not user_has_active_run(callback, runner),
        )

    @router.callback_query(lambda callback: callback.data in {"lb", LEADERBOARD_CALLBACK})
    async def leaderboard_callback(callback: CallbackQuery) -> None:
        await callback.answer()
        clear_search_mode(callback, search_modes)
        await show_leaderboard(callback, runner, renderer, 0)

    @router.callback_query(lambda callback: callback.data == LEADERBOARD_SEARCH_CALLBACK)
    async def leaderboard_search_callback(callback: CallbackQuery) -> None:
        await callback.answer()
        set_search_mode(callback, search_modes, "leaderboard")
        await ask_search_from_callback(callback, runner, renderer, "search.leaderboard_html", "search.leaderboard_plain")

    @router.callback_query(lambda callback: callback.data == FIGHT_CALLBACK)
    async def fight_callback(callback: CallbackQuery) -> None:
        await callback.answer()
        clear_search_mode(callback, search_modes)
        await show_fight_from_callback(callback, runner, renderer)

    @router.callback_query(lambda callback: bool(callback.data and callback.data.startswith(f"{FIGHT_PAGE_CALLBACK}:")))
    async def fight_page_callback(callback: CallbackQuery) -> None:
        await callback.answer()
        step, page = fight_page_payload(callback.data or "")
        await show_fight_picker(callback, runner, renderer, step, page)

    @router.callback_query(lambda callback: bool(callback.data and callback.data.startswith(f"{FIGHT_PICK_CALLBACK}:")))
    async def fight_pick_callback(callback: CallbackQuery) -> None:
        await callback.answer()
        await handle_fight_pick(callback, runner, renderer)

    @router.callback_query(lambda callback: bool(callback.data and callback.data.startswith(f"{FIGHT_PICK_RUN_CALLBACK}:")))
    async def fight_pick_run_callback(callback: CallbackQuery) -> None:
        await callback.answer()
        await handle_fight_pick_run(callback, runner, renderer)

    @router.callback_query(lambda callback: bool(callback.data and callback.data.startswith(f"{FIGHT_SEARCH_CALLBACK}:")))
    async def fight_search_callback(callback: CallbackQuery) -> None:
        await callback.answer()
        if not callback.data:
            return
        step = fight_search_payload(callback.data)
        set_search_mode(callback, search_modes, f"fight:{step}")
        await ask_search_from_callback(callback, runner, renderer, "search.fight_html", "search.fight_plain")

    @router.callback_query(lambda callback: bool(callback.data and callback.data.startswith(f"{FIGHT_MATCHES_CALLBACK}:")))
    async def fight_matches_callback(callback: CallbackQuery) -> None:
        await callback.answer()
        await show_fight_matches(callback, runner, renderer)

    @router.callback_query(lambda callback: bool(callback.data and callback.data.startswith(f"{FIGHT_VIEW_CALLBACK}:")))
    async def fight_view_callback(callback: CallbackQuery) -> None:
        await callback.answer()
        await show_fight_view(callback, runner, renderer, fight_image_generator)

    @router.callback_query(lambda callback: bool(callback.data and (callback.data.startswith("lb:") or callback.data.startswith(f"{LEADERBOARD_CALLBACK}:"))))
    async def leaderboard_page_callback(callback: CallbackQuery) -> None:
        await callback.answer()
        if not callback.data:
            return
        await show_leaderboard(callback, runner, renderer, page_from_callback(callback.data))

    @router.callback_query(lambda callback: bool(callback.data and (callback.data.startswith("card:") or callback.data.startswith("v1:card:"))))
    async def card_callback(callback: CallbackQuery) -> None:
        await callback.answer()
        if not callback.message or not callback.from_user or not callback.data:
            return
        run_id, page = card_payload(callback.data)
        run = runner.store.get_run(run_id)
        if not run or not run.result:
            await edit_callback_screen(
                callback,
                runner,
                renderer,
                text("screens.last_missing_html"),
                text("last_missing"),
                back_menu(),
                track_screen=not user_has_active_run(callback, runner),
            )
            return
        await edit_result(
            callback.message,
            runner,
            renderer,
            run,
            callback.from_user.id,
            result_menu(f"{LEADERBOARD_CALLBACK}:{page}", run.run_id),
            track_screen=not user_has_active_run(callback, runner),
        )

    @router.callback_query(lambda callback: bool(callback.data and (callback.data.startswith("img:") or callback.data.startswith(f"{IMAGE_CALLBACK}:"))))
    async def image_callback(callback: CallbackQuery) -> None:
        await callback.answer(text("image.wait")[:200])
        if not callback.message or not callback.data:
            return
        run = runner.store.get_run(image_payload(callback.data))
        if not run or not run.result:
            await callback.message.answer(text("image.missing"))
            return
        result = result_with_leaderboard_place(runner, run)
        try:
            await callback.message.bot.send_chat_action(chat_id=callback.message.chat.id, action="upload_photo")
            path = await image_generator.get_or_create(result)
            await callback.message.answer_photo(
                FSInputFile(path),
                caption=image_caption(result),
                parse_mode="HTML",
            )
        except Exception as exc:
            await callback.message.answer(text("image.failed", error=str(exc)[:500]))

    return router


async def show_home_from_message(message: Message, runner: AirbotRunner, renderer: AirbotTelegramRenderer) -> None:
    user_id = message.from_user.id if message.from_user else 0
    if user_id:
        runner.store.clear_fight_session(user_id, message.chat.id)
    if user_id and runner.store.active_count_for_user(user_id) > 0:
        await send_detached_screen(
            message=message,
            renderer=renderer,
            html=start_html(),
            fallback=start_plain(),
            menu=main_menu(),
        )
        return
    await send_fresh_screen(
        message=message,
        runner=runner,
        renderer=renderer,
        html=start_html(),
        fallback=start_plain(),
        menu=main_menu(),
        awaiting_channel=False,
    )


async def show_home_from_callback(callback: CallbackQuery, runner: AirbotRunner, renderer: AirbotTelegramRenderer) -> None:
    if callback.from_user and callback.message:
        runner.store.clear_fight_session(callback.from_user.id, callback.message.chat.id)
    if active_progress_callback(callback, runner):
        await send_detached_screen(
            message=callback.message,
            renderer=renderer,
            html=start_html(),
            fallback=start_plain(),
            menu=main_menu(),
        )
        return
    await edit_callback_screen(
        callback,
        runner,
        renderer,
        start_html(),
        start_plain(),
        main_menu(),
        awaiting_channel=False,
        track_screen=not user_has_active_run(callback, runner),
    )


async def show_ask_from_message(message: Message, runner: AirbotRunner, renderer: AirbotTelegramRenderer) -> None:
    await render_saved_or_new(message, runner, renderer, ask_channel_html(), text("ask_channel"), back_menu(), awaiting_channel=True)
    await delete_incoming(message)


async def show_ask_from_callback(callback: CallbackQuery, runner: AirbotRunner, renderer: AirbotTelegramRenderer) -> None:
    await edit_callback_screen(callback, runner, renderer, ask_channel_html(), text("ask_channel"), back_menu(), awaiting_channel=True)


async def ask_search_from_callback(
    callback: CallbackQuery,
    runner: AirbotRunner,
    renderer: AirbotTelegramRenderer,
    html_key: str,
    plain_key: str,
) -> None:
    await edit_callback_screen(
        callback,
        runner,
        renderer,
        text(html_key),
        text(plain_key),
        back_menu(),
        track_screen=not user_has_active_run(callback, runner),
    )


async def handle_search_text(message: Message, runner: AirbotRunner, renderer: AirbotTelegramRenderer, mode: str) -> None:
    query = search_query_from_text(message.text or "")
    if not query:
        await render_saved_or_new(message, runner, renderer, text("search.empty_query_html"), text("search.empty_query_plain"), back_menu())
        return
    if mode == "leaderboard":
        await show_leaderboard_search_from_message(message, runner, renderer, query)
        return
    if mode.startswith("fight:"):
        await show_fight_search_from_message(message, runner, renderer, query, fight_search_mode_step(mode))


async def show_leaderboard_search_from_message(
    message: Message,
    runner: AirbotRunner,
    renderer: AirbotTelegramRenderer,
    query: str,
) -> None:
    runs = runner.store.find_leaderboard(query, limit=PAGE_SIZE)
    fight_stats = {run.channel: runner.store.fight_stats(run.channel) for run in runs}
    if runs:
        body_html = leaderboard_html(
            runs,
            0,
            False,
            False,
            image_urls=renderer.generated_image_urls([run.channel for run in runs]),
            fight_stats=fight_stats,
        )
        body_plain = leaderboard_text(runs, fight_stats=fight_stats)
    else:
        body_html = text("search.no_results_html")
        body_plain = text("search.no_results_plain")
    await render_saved_or_new(
        message,
        runner,
        renderer,
        text("search.results_html", query=escape(query), body=body_html),
        text("search.results_plain", query=query, body=body_plain),
        leaderboard_menu(runs, 0, False),
    )
    await delete_incoming(message)


async def show_fight_search_from_message(
    message: Message,
    runner: AirbotRunner,
    renderer: AirbotTelegramRenderer,
    query: str,
    step: int,
) -> None:
    if not message.from_user:
        return
    step = 2 if step == 2 else 1
    session = runner.store.fight_session(message.from_user.id, message.chat.id)
    first = runner.store.get_run(session.first_run_id) if session and session.first_run_id else None
    if step == 2 and not first:
        runner.store.start_fight_session(message.from_user.id, message.chat.id)
        step = 1
    runs = runner.store.find_leaderboard(query, limit=PAGE_SIZE)
    await render_saved_or_new(
        message,
        runner,
        renderer,
        fight_search_html(step, first, query, bool(runs)),
        fight_search_plain(step, first, query, bool(runs)),
        fight_pick_search_menu(runs, step),
        awaiting_channel=False,
    )
    await delete_incoming(message)


async def show_leaderboard(callback: CallbackQuery, runner: AirbotRunner, renderer: AirbotTelegramRenderer, page: int) -> None:
    if not callback.message:
        return
    page = max(0, page)
    runs = runner.store.leaderboard(limit=PAGE_SIZE + 1, offset=page * PAGE_SIZE)
    visible = runs[:PAGE_SIZE]
    fight_stats = {run.channel: runner.store.fight_stats(run.channel) for run in visible}
    html = leaderboard_html(
        visible,
        page,
        page > 0,
        len(runs) > PAGE_SIZE,
        image_urls=renderer.generated_image_urls([run.channel for run in visible]),
        fight_stats=fight_stats,
    )
    await edit_callback_screen(
        callback,
        runner,
        renderer,
        html,
        leaderboard_text(visible, fight_stats=fight_stats),
        leaderboard_menu(visible, page, len(runs) > PAGE_SIZE),
        track_screen=not user_has_active_run(callback, runner),
    )


async def show_fight_from_message(message: Message, runner: AirbotRunner, renderer: AirbotTelegramRenderer) -> None:
    if not message.from_user:
        return
    runner.store.start_fight_session(message.from_user.id, message.chat.id)
    runs = runner.store.leaderboard(limit=PAGE_SIZE + 1)
    visible = runs[:PAGE_SIZE]
    await render_saved_or_new(
        message,
        runner,
        renderer,
        fight_pick_html(1, None, bool(visible)),
        fight_pick_plain(1, None, bool(visible)),
        fight_pick_menu(visible, 0, len(runs) > PAGE_SIZE, 1),
        awaiting_channel=False,
    )


async def show_fight_from_callback(callback: CallbackQuery, runner: AirbotRunner, renderer: AirbotTelegramRenderer) -> None:
    if not callback.message or not callback.from_user:
        return
    runner.store.start_fight_session(callback.from_user.id, callback.message.chat.id)
    await show_fight_picker(callback, runner, renderer, 1, 0)


async def show_fight_picker(
    callback: CallbackQuery,
    runner: AirbotRunner,
    renderer: AirbotTelegramRenderer,
    step: int,
    page: int,
) -> None:
    if not callback.message or not callback.from_user:
        return
    page = max(0, page)
    step = 2 if step == 2 else 1
    session = runner.store.fight_session(callback.from_user.id, callback.message.chat.id)
    first = runner.store.get_run(session.first_run_id) if session and session.first_run_id else None
    if step == 2 and not first:
        runner.store.start_fight_session(callback.from_user.id, callback.message.chat.id)
        step = 1
    runs = runner.store.leaderboard(limit=PAGE_SIZE + 1, offset=page * PAGE_SIZE)
    visible = runs[:PAGE_SIZE]
    await edit_callback_screen(
        callback,
        runner,
        renderer,
        fight_pick_html(step, first, bool(visible)),
        fight_pick_plain(step, first, bool(visible)),
        fight_pick_menu(visible, page, len(runs) > PAGE_SIZE, step),
        track_screen=not user_has_active_run(callback, runner),
    )


async def handle_fight_pick(callback: CallbackQuery, runner: AirbotRunner, renderer: AirbotTelegramRenderer) -> None:
    if not callback.message or not callback.from_user or not callback.data:
        return
    step, page, index = fight_pick_payload(callback.data)
    run = fight_run_from_index(runner, page, index)
    await handle_fight_selected(callback, runner, renderer, run, step, page)


async def handle_fight_pick_run(callback: CallbackQuery, runner: AirbotRunner, renderer: AirbotTelegramRenderer) -> None:
    if not callback.message or not callback.from_user or not callback.data:
        return
    step, run_id = fight_pick_run_payload(callback.data)
    await handle_fight_selected(callback, runner, renderer, runner.store.get_run(run_id), step, 0)


async def handle_fight_selected(
    callback: CallbackQuery,
    runner: AirbotRunner,
    renderer: AirbotTelegramRenderer,
    run: BotRun | None,
    step: int,
    page: int,
) -> None:
    if not callback.message or not callback.from_user:
        return
    if not run or not run.result:
        await edit_callback_screen(callback, runner, renderer, text("fight.missing_html"), text("fight.missing"), back_menu())
        return
    if step == 1:
        runner.store.set_fight_first(callback.from_user.id, callback.message.chat.id, run.run_id)
        await show_fight_picker(callback, runner, renderer, 2, 0)
        return
    session = runner.store.fight_session(callback.from_user.id, callback.message.chat.id)
    first = runner.store.get_run(session.first_run_id) if session and session.first_run_id else None
    if not first:
        runner.store.start_fight_session(callback.from_user.id, callback.message.chat.id)
        await show_fight_picker(callback, runner, renderer, 1, 0)
        return
    if first.run_id == run.run_id:
        runs = runner.store.leaderboard(limit=PAGE_SIZE + 1, offset=page * PAGE_SIZE)
        await edit_callback_screen(
            callback,
            runner,
            renderer,
            text("fight.same_html", channel=escape(run.channel)),
            text("fight.same", channel=run.channel),
            fight_pick_menu(runs[:PAGE_SIZE], page, len(runs) > PAGE_SIZE, 2),
            track_screen=not user_has_active_run(callback, runner),
        )
        return
    gap_error = fight_elo_gap_error(runner, first, run)
    if gap_error:
        await edit_callback_screen(
            callback,
            runner,
            renderer,
            text(
                "fight.elo_gap_blocked_html",
                gap=gap_error.gap,
                max_gap=gap_error.max_gap,
                left=gap_error.left_elo,
                right=gap_error.right_elo,
            ),
            text(
                "fight.elo_gap_blocked",
                gap=gap_error.gap,
                max_gap=gap_error.max_gap,
                left=gap_error.left_elo,
                right=gap_error.right_elo,
            ),
            back_menu(),
            track_screen=not user_has_active_run(callback, runner),
        )
        return
    runner.store.set_fight_second(callback.from_user.id, callback.message.chat.id, run.run_id)
    await edit_callback_screen(
        callback,
        runner,
        renderer,
        text("fight.topic_html", left=channel_link_html(first.channel), right=channel_link_html(run.channel)),
        text("fight.topic_plain", left=f"@{first.channel}", right=f"@{run.channel}"),
        back_menu(),
        track_screen=not user_has_active_run(callback, runner),
    )


async def start_fight_from_topic(
    message: Message,
    runner: AirbotRunner,
    renderer: AirbotTelegramRenderer,
    fight_runner: AirbotFightRunner,
    fight_image_generator: AirbotFightImageGenerator,
) -> None:
    if not message.from_user:
        return
    user_id = message.from_user.id
    topic = fight_topic_from_text(message.text or "")
    if not topic:
        await render_saved_or_new(message, runner, renderer, text("fight.topic_invalid_html"), text("fight.topic_invalid"), back_menu())
        return
    session = runner.store.fight_session(user_id, message.chat.id)
    if not session:
        await render_saved_or_new(message, runner, renderer, text("fight.missing_html"), text("fight.missing"), back_menu())
        return
    left = runner.store.get_run(session.first_run_id or "")
    right = runner.store.get_run(session.second_run_id or "")
    if not left or not right or not left.result or not right.result:
        runner.store.clear_fight_session(user_id, message.chat.id)
        await render_saved_or_new(message, runner, renderer, text("fight.missing_html"), text("fight.missing"), back_menu())
        return
    left = runner.store.run_with_fight_rating(left)
    right = runner.store.run_with_fight_rating(right)
    try:
        prepared = await fight_runner.prepare(user_id, message.chat.id, left, right, topic)
    except AirbotFightBusyUser:
        await render_saved_or_new(message, runner, renderer, text("fight.busy_html"), text("fight.busy"), back_menu())
        return
    except AirbotFightEloGapTooHigh as exc:
        await render_saved_or_new(
            message,
            runner,
            renderer,
            text("fight.elo_gap_blocked_html", gap=exc.gap, max_gap=exc.max_gap, left=exc.left_elo, right=exc.right_elo),
            text("fight.elo_gap_blocked", gap=exc.gap, max_gap=exc.max_gap, left=exc.left_elo, right=exc.right_elo),
            back_menu(),
        )
        return
    runner.store.clear_fight_session(user_id, message.chat.id)
    if prepared.cached:
        if prepared.fight.result:
            await send_fight_summary(
                message,
                runner,
                renderer,
                fight_image_generator,
                prepared.fight,
            )
        return
    queued = text(
        "fight.queued",
        left=left.channel,
        right=right.channel,
        topic=topic,
        position=runner.store.fight_queue_position(prepared.fight.fight_id),
        concurrency=runner.settings.bot_fight_global_concurrency,
    )
    await render_saved_or_new(
        message,
        runner,
        renderer,
        status_html(queued),
        queued,
        progress_menu(),
    )

    async def stage(value: str) -> None:
        await render_saved_or_new(message, runner, renderer, status_html(value), value, progress_menu())

    async def run_and_send() -> None:
        completed = await fight_runner.execute(prepared.fight.fight_id, stage=stage)
        if completed.status == "completed" and completed.result:
            await send_live_fight(
                message,
                runner,
                renderer,
                fight_image_generator,
                completed,
            )
        elif completed.status == "failed":
            await render_saved_or_new(
                message,
                runner,
                renderer,
                text("fight.failed_html", error=escape(completed.error or "")[:700]),
                text("failed", error=completed.error or ""),
                fight_result_menu(),
            )

    asyncio.create_task(run_and_send())


async def show_fight_matches(callback: CallbackQuery, runner: AirbotRunner, renderer: AirbotTelegramRenderer) -> None:
    if not callback.message or not callback.from_user or not callback.data:
        return
    run_id, page = fight_matches_payload(callback.data)
    run = runner.store.get_run(run_id)
    if not run:
        await edit_callback_screen(callback, runner, renderer, text("fight.missing_html"), text("fight.missing"), back_menu())
        return
    page = max(0, page)
    fights = runner.store.channel_fights(run.channel, limit=PAGE_SIZE + 1, offset=page * PAGE_SIZE)
    visible = fights[:PAGE_SIZE]
    await edit_callback_screen(
        callback,
        runner,
        renderer,
        fight_matches_html(run.channel, visible, page),
        fight_matches_plain(run.channel, visible, page),
        fight_matches_menu(visible, run.channel, run.run_id, page, len(fights) > PAGE_SIZE),
        track_screen=not user_has_active_run(callback, runner),
    )
async def show_fight_view(
    callback: CallbackQuery,
    runner: AirbotRunner,
    renderer: AirbotTelegramRenderer,
    fight_image_generator: AirbotFightImageGenerator,
) -> None:
    if not callback.message or not callback.data:
        return
    fight = runner.store.get_fight_run(fight_view_payload(callback.data))
    if not fight or not fight.result:
        await edit_callback_screen(callback, runner, renderer, text("fight.missing_html"), text("fight.missing"), back_menu())
        return
    await edit_callback_screen(
        callback,
        runner,
        renderer,
        fight_result_html(fight.result),
        fight_result_plain(fight.result),
        fight_result_menu(),
        track_screen=not user_has_active_run(callback, runner),
    )
    await send_existing_fight_images(callback.message, fight_image_generator, fight)


async def send_fight_summary(
    message: Message,
    runner: AirbotRunner,
    renderer: AirbotTelegramRenderer,
    fight_image_generator: AirbotFightImageGenerator,
    fight: BotFightRun,
) -> None:
    if not fight.result:
        return
    await render_saved_or_new(
        message,
        runner,
        renderer,
        fight_result_html(fight.result),
        fight_result_plain(fight.result),
        fight_result_menu(),
    )
    await send_existing_fight_images(message, fight_image_generator, fight)


async def send_existing_fight_images(
    message: Message,
    fight_image_generator: AirbotFightImageGenerator,
    fight: BotFightRun,
) -> None:
    if not fight.result:
        return
    for stage, path in fight_existing_media_items(fight_image_generator, fight.result):
        await message.answer_photo(
            FSInputFile(path),
            caption=fight_image_generator.stage_caption(fight.result, stage),
            parse_mode="HTML",
        )


def fight_existing_media_items(fight_image_generator: AirbotFightImageGenerator, result: dict) -> list[tuple[str, Path]]:
    images = result.get("images") if isinstance(result.get("images"), dict) else {}
    items = []
    for stage in fight_image_generator.stages(result):
        value = images.get(stage)
        if not value:
            continue
        path = Path(str(value))
        if path.exists():
            items.append((stage, path))
    return items


async def send_live_fight(
    message: Message,
    runner: AirbotRunner,
    renderer: AirbotTelegramRenderer,
    fight_image_generator: AirbotFightImageGenerator,
    fight: BotFightRun,
) -> None:
    if not fight.result:
        return
    result = dict(fight.result)
    left = runner.store.get_run(fight.left_run_id)
    right = runner.store.get_run(fight.right_run_id)
    if not left or not right or not left.result or not right.result:
        await send_fight_summary(message, runner, renderer, fight_image_generator, fight)
        return
    for stage in fight_image_generator.stages(result):
        await render_saved_or_new(
            message,
            runner,
            renderer,
            status_html(text("fight.image_generating", stage=stage)),
            text("fight.image_generating", stage=stage),
            progress_menu(),
        )
        try:
            path = await fight_image_generator.generate_stage(fight, left, right, result, stage)
        except Exception as exc:
            await render_saved_or_new(
                message,
                runner,
                renderer,
                status_html(text("fight.image_failed", stage=stage, error=str(exc)[:300])),
                text("fight.image_failed", stage=stage, error=str(exc)[:300]),
                progress_menu(),
            )
            continue
        result = fight_image_generator.attach_image(result, stage, path)
        fight = runner.store.complete_fight_run(fight.fight_id, result)
        await message.answer_photo(
            FSInputFile(path),
            caption=fight_image_generator.stage_caption(result, stage),
            parse_mode="HTML",
        )
    await render_saved_or_new(
        message,
        runner,
        renderer,
        fight_result_html(result),
        fight_result_plain(result),
        fight_result_menu(),
    )


async def enqueue(message: Message, runner: AirbotRunner, renderer: AirbotTelegramRenderer, channel: str) -> None:
    user_id = message.from_user.id if message.from_user else 0
    await enqueue_message(message, runner, renderer, user_id, channel)


async def enqueue_message(
    message: Message,
    runner: AirbotRunner,
    renderer: AirbotTelegramRenderer,
    user_id: int,
    channel: str,
) -> None:
    try:
        prepared = await runner.prepare(user_id, message.chat.id, channel)
    except AirbotInvalidChannel:
        await render_saved_or_new(message, runner, renderer, text("screens.invalid_html"), text("invalid_channel"), back_menu(), awaiting_channel=True)
        return
    except AirbotBusyUser:
        await send_detached_screen(message, renderer, text("screens.busy_html"), text("busy_user"), back_menu())
        return
    except AirbotCooldown as exc:
        await render_saved_or_new(
            message,
            runner,
            renderer,
            text("screens.cooldown_html", seconds=exc.seconds),
            text("cooldown", seconds=exc.seconds),
            back_menu(),
        )
        return

    runner.store.set_awaiting_channel(user_id, message.chat.id, False)
    if prepared.cached:
        if prepared.run.result:
            await edit_result_by_message(message, runner, renderer, prepared.run, user_id, result_menu(HOME_CALLBACK, prepared.run.run_id))
        return

    queued = text(
        "queued",
        channel=prepared.run.channel,
        position=runner.store.queue_position(prepared.run.run_id),
        concurrency=runner.settings.bot_global_concurrency,
    )
    await render_saved_or_new(message, runner, renderer, status_html(queued), queued, progress_menu())

    async def stage(value: str) -> None:
        await render_saved_or_new(message, runner, renderer, status_html(value), value, progress_menu())

    async def run_and_send() -> None:
        completed = await runner.execute(prepared.run.run_id, stage=stage)
        if completed.status == "completed" and completed.result:
            await edit_result_by_message(message, runner, renderer, completed, user_id, result_menu(HOME_CALLBACK, completed.run_id))

    asyncio.create_task(run_and_send())


async def edit_result(
    message: Message,
    runner: AirbotRunner,
    renderer: AirbotTelegramRenderer,
    run: BotRun,
    referrer_user_id: int,
    reply_markup: InlineKeyboardMarkup,
    track_screen: bool = True,
) -> None:
    bot_username = await renderer.bot_username(message.bot)
    result = result_with_leaderboard_place(runner, run)
    new_id = await renderer.send_rich_screen(
        message,
        await renderer.result_html_variants(result, bot_username=bot_username, referrer_user_id=referrer_user_id),
        result_screen_plain(result, bot_username=bot_username, referrer_user_id=referrer_user_id),
        reply_markup,
    )
    await delete_message(message)
    if track_screen:
        runner.store.set_screen(referrer_user_id, message.chat.id, new_id, awaiting_channel=False)


async def edit_result_by_message(
    message: Message,
    runner: AirbotRunner,
    renderer: AirbotTelegramRenderer,
    run: BotRun,
    referrer_user_id: int,
    reply_markup: InlineKeyboardMarkup,
) -> None:
    bot_username = await renderer.bot_username(message.bot)
    result = result_with_leaderboard_place(runner, run)
    html = await renderer.result_html_variants(result, bot_username=bot_username, referrer_user_id=referrer_user_id)
    fallback = result_screen_plain(result, bot_username=bot_username, referrer_user_id=referrer_user_id)
    screen_id = runner.store.screen_message_id(referrer_user_id, message.chat.id)
    new_id = await renderer.send_rich_screen(message, html, fallback, reply_markup)
    if screen_id:
        await delete_message_by_id(message, screen_id)
    runner.store.set_screen(referrer_user_id, message.chat.id, new_id, awaiting_channel=False)


async def edit_callback_screen(
    callback: CallbackQuery,
    runner: AirbotRunner,
    renderer: AirbotTelegramRenderer,
    html: str | list[str],
    fallback: str,
    reply_markup: InlineKeyboardMarkup,
    awaiting_channel: bool = False,
    track_screen: bool = True,
) -> None:
    if not callback.message or not callback.from_user:
        return
    await renderer.edit_rich_screen(callback.message.bot, callback.message.chat.id, callback.message.message_id, html, fallback, reply_markup)
    if track_screen:
        runner.store.set_screen(callback.from_user.id, callback.message.chat.id, callback.message.message_id, awaiting_channel=awaiting_channel)


async def render_saved_or_new(
    message: Message,
    runner: AirbotRunner,
    renderer: AirbotTelegramRenderer,
    html: str | list[str],
    fallback: str,
    reply_markup: InlineKeyboardMarkup,
    awaiting_channel: bool = False,
) -> int:
    user_id = message.from_user.id if message.from_user else 0
    screen_id = runner.store.screen_message_id(user_id, message.chat.id)
    if screen_id:
        edited = await renderer.edit_rich_screen(message.bot, message.chat.id, screen_id, html, fallback, reply_markup)
        if edited:
            runner.store.set_screen(user_id, message.chat.id, screen_id, awaiting_channel=awaiting_channel)
            return screen_id
        try:
            await message.bot.delete_message(chat_id=message.chat.id, message_id=screen_id)
        except Exception:
            pass
    new_id = await renderer.send_rich_screen(message, html, fallback, reply_markup)
    runner.store.set_screen(user_id, message.chat.id, new_id, awaiting_channel=awaiting_channel)
    return new_id


async def send_fresh_screen(
    message: Message,
    runner: AirbotRunner,
    renderer: AirbotTelegramRenderer,
    html: str | list[str],
    fallback: str,
    menu: InlineKeyboardMarkup,
    awaiting_channel: bool = False,
) -> int:
    user_id = message.from_user.id if message.from_user else 0
    new_id = await renderer.send_rich_screen(message, html, fallback, menu)
    runner.store.set_screen(user_id, message.chat.id, new_id, awaiting_channel=awaiting_channel)
    return new_id


async def send_detached_screen(
    message: Message,
    renderer: AirbotTelegramRenderer,
    html: str | list[str],
    fallback: str,
    menu: InlineKeyboardMarkup,
) -> int:
    return await renderer.send_rich_screen(message, html, fallback, menu)


async def delete_incoming(message: Message) -> None:
    try:
        await message.delete()
    except Exception:
        pass


async def delete_message(message: Message) -> None:
    try:
        await message.delete()
    except Exception:
        pass


async def delete_message_by_id(message: Message, message_id: int) -> None:
    try:
        await message.bot.delete_message(chat_id=message.chat.id, message_id=message_id)
    except Exception:
        pass


def status_html(value: str) -> str:
    return text("screens.status_html", body=escape(value))


def result_with_leaderboard_place(runner: AirbotRunner, run: BotRun) -> dict:
    ranked = runner.store.run_with_fight_rating(run)
    result = dict(ranked.result or {})
    place = runner.store.leaderboard_place(run.channel)
    if place:
        result["leaderboard_place"] = place
    result["fight_stats"] = runner.store.fight_stats(run.channel)
    return result


def image_caption(result: dict) -> str:
    channel = str(result.get("channel") or "")
    elo = escape(str(result.get("air_elo") or result.get("score") or ""))
    profile = result.get("meme_profile") if isinstance(result.get("meme_profile"), dict) else {}
    rank = escape(str(profile.get("rank_name") or result.get("humorous_label") or ""))
    channel_link = f'<a href="https://t.me/{escape(channel)}">@{escape(channel)}</a>' if channel else ""
    return text("image.caption", channel=channel_link, elo=elo, rank=rank)[:1024]


def main_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=text("buttons.analyze"), callback_data=ANALYZE_CALLBACK)],
            [InlineKeyboardButton(text=text("buttons.fight"), callback_data=FIGHT_CALLBACK)],
            [
                InlineKeyboardButton(text=text("buttons.leaderboard"), callback_data=LEADERBOARD_CALLBACK),
                InlineKeyboardButton(text=text("buttons.leaderboard_search"), callback_data=LEADERBOARD_SEARCH_CALLBACK),
            ],
            [
                InlineKeyboardButton(text=text("buttons.last_result"), callback_data=LAST_CALLBACK),
            ],
        ]
    )


def back_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text=text("buttons.back"), callback_data=HOME_CALLBACK)]])


def progress_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text=text("buttons.menu"), callback_data=MENU_CALLBACK)]])


def result_menu(back_callback: str, run_id: str | None = None) -> InlineKeyboardMarkup:
    rows = []
    if run_id:
        rows.append([InlineKeyboardButton(text=text("buttons.image"), callback_data=f"{IMAGE_CALLBACK}:{run_id}")])
        rows.append([InlineKeyboardButton(text=text("buttons.fight_matches"), callback_data=f"{FIGHT_MATCHES_CALLBACK}:{run_id}:0")])
    rows.extend(
        [
            [InlineKeyboardButton(text=text("buttons.leaderboard"), callback_data=LEADERBOARD_CALLBACK)],
            [InlineKeyboardButton(text=text("buttons.back"), callback_data=back_callback)],
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def fight_matches_menu(fights: list[BotFightRun], channel: str, run_id: str, page: int, has_next: bool) -> InlineKeyboardMarkup:
    rows = []
    for index, fight in enumerate(fights, start=1 + page * PAGE_SIZE):
        rows.append(
            [
                InlineKeyboardButton(
                    text=fight_match_button_label(fight, channel, index)[:52],
                    callback_data=f"{FIGHT_VIEW_CALLBACK}:{fight.fight_id}",
                )
            ]
        )
    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton(text=text("buttons.prev"), callback_data=f"{FIGHT_MATCHES_CALLBACK}:{run_id}:{page - 1}"))
    if has_next:
        nav.append(InlineKeyboardButton(text=text("buttons.next"), callback_data=f"{FIGHT_MATCHES_CALLBACK}:{run_id}:{page + 1}"))
    if nav:
        rows.append(nav)
    rows.append([InlineKeyboardButton(text=text("buttons.back"), callback_data=HOME_CALLBACK)])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def fight_result_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=text("buttons.fight"), callback_data=FIGHT_CALLBACK)],
            [InlineKeyboardButton(text=text("buttons.back"), callback_data=HOME_CALLBACK)],
        ]
    )


def fight_pick_menu(runs: list[BotRun], page: int, has_next: bool, step: int) -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton(text=text("buttons.search"), callback_data=f"{FIGHT_SEARCH_CALLBACK}:{step}")]]
    for index, run in enumerate(runs):
        result = run.result or {}
        global_index = 1 + page * PAGE_SIZE + index
        rows.append(
            [
                InlineKeyboardButton(
                    text=text(
                        "buttons.fight_player",
                        index=global_index,
                        channel=run.channel,
                        elo=str(result.get("air_elo") or result.get("score") or ""),
                    )[:52],
                    callback_data=f"{FIGHT_PICK_CALLBACK}:{step}:{page}:{index}",
                )
            ]
        )
    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton(text=text("buttons.prev"), callback_data=f"{FIGHT_PAGE_CALLBACK}:{step}:{page - 1}"))
    if has_next:
        nav.append(InlineKeyboardButton(text=text("buttons.next"), callback_data=f"{FIGHT_PAGE_CALLBACK}:{step}:{page + 1}"))
    if nav:
        rows.append(nav)
    rows.append([InlineKeyboardButton(text=text("buttons.back"), callback_data=HOME_CALLBACK)])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def fight_pick_search_menu(runs: list[BotRun], step: int) -> InlineKeyboardMarkup:
    rows = []
    for run in runs:
        result = run.result or {}
        rows.append(
            [
                InlineKeyboardButton(
                    text=text(
                        "buttons.fight_player_search",
                        channel=run.channel,
                        elo=str(result.get("air_elo") or result.get("score") or ""),
                    )[:52],
                    callback_data=f"{FIGHT_PICK_RUN_CALLBACK}:{step}:{run.run_id}",
                )
            ]
        )
    rows.append([InlineKeyboardButton(text=text("buttons.search"), callback_data=f"{FIGHT_SEARCH_CALLBACK}:{step}")])
    rows.append([InlineKeyboardButton(text=text("buttons.back"), callback_data=FIGHT_CALLBACK)])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def leaderboard_menu(runs: list[BotRun], page: int, has_next: bool) -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton(text=text("buttons.search"), callback_data=LEADERBOARD_SEARCH_CALLBACK)]]
    for index, run in enumerate(runs, start=1 + page * PAGE_SIZE):
        result = run.result or {}
        label = text("buttons.leaderboard_item", index=index, channel=run.channel, elo=str(result.get("air_elo") or result.get("score") or ""))
        rows.append(
            [
                InlineKeyboardButton(text=label[:52], callback_data=f"v1:card:{run.run_id}:{page}"),
                InlineKeyboardButton(text=text("buttons.image"), callback_data=f"{IMAGE_CALLBACK}:{run.run_id}"),
            ]
        )
    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton(text=text("buttons.prev"), callback_data=f"{LEADERBOARD_CALLBACK}:{page - 1}"))
    if has_next:
        nav.append(InlineKeyboardButton(text=text("buttons.next"), callback_data=f"{LEADERBOARD_CALLBACK}:{page + 1}"))
    if nav:
        rows.append(nav)
    rows.append([InlineKeyboardButton(text=text("buttons.back"), callback_data=HOME_CALLBACK)])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def user_has_active_run(callback: CallbackQuery, runner: AirbotRunner) -> bool:
    return bool(callback.from_user and runner.store.active_count_for_user(callback.from_user.id) > 0)


def active_progress_callback(callback: CallbackQuery, runner: AirbotRunner) -> bool:
    if not callback.message or not callback.from_user or not user_has_active_run(callback, runner):
        return False
    if callback.data in {"menu", MENU_CALLBACK}:
        return True
    screen_id = runner.store.screen_message_id(callback.from_user.id, callback.message.chat.id)
    return bool(screen_id and callback.message.message_id == screen_id)


def fight_elo_gap_error(runner: AirbotRunner, left: BotRun, right: BotRun) -> AirbotFightEloGapTooHigh | None:
    left_elo = run_elo(runner.store.run_with_fight_rating(left))
    right_elo = run_elo(runner.store.run_with_fight_rating(right))
    max_gap = runner.settings.bot_fight_max_elo_gap
    if max_gap > 0 and abs(left_elo - right_elo) > max_gap:
        return AirbotFightEloGapTooHigh(left_elo, right_elo, max_gap)
    return None


def run_elo(run: BotRun) -> int:
    result = run.result or {}
    for key in ("battle_elo", "air_elo", "score"):
        try:
            return int(result.get(key))
        except Exception:
            continue
    return 0


def fight_pick_html(step: int, first: BotRun | None, has_rows: bool) -> str:
    if not has_rows:
        return text("fight.empty_html")
    if step == 2 and first:
        return text("fight.pick_second_html", first=channel_link_html(first.channel))
    return text("fight.pick_first_html")


def fight_pick_plain(step: int, first: BotRun | None, has_rows: bool) -> str:
    if not has_rows:
        return text("fight.empty")
    if step == 2 and first:
        return text("fight.pick_second", first=f"@{first.channel}")
    return text("fight.pick_first")


def fight_search_html(step: int, first: BotRun | None, query: str, has_rows: bool) -> str:
    if not has_rows:
        return text("search.no_results_html")
    if step == 2 and first:
        return text("search.fight_results_second_html", query=escape(query), first=channel_link_html(first.channel))
    return text("search.fight_results_first_html", query=escape(query))


def fight_search_plain(step: int, first: BotRun | None, query: str, has_rows: bool) -> str:
    if not has_rows:
        return text("search.no_results_plain")
    if step == 2 and first:
        return text("search.fight_results_second_plain", query=query, first=f"@{first.channel}")
    return text("search.fight_results_first_plain", query=query)


def fight_run_from_index(runner: AirbotRunner, page: int, index: int) -> BotRun | None:
    if index < 0 or index >= PAGE_SIZE:
        return None
    runs = runner.store.leaderboard(limit=PAGE_SIZE, offset=max(0, page) * PAGE_SIZE)
    return runs[index] if index < len(runs) else None


def fight_topic_from_text(value: str) -> str:
    return " ".join(value.split())[:180]


def search_query_from_text(value: str) -> str:
    channel = channel_from_text(value)
    if channel:
        return channel
    return " ".join(str(value or "").split())[:120]


def channel_link_html(channel: str) -> str:
    value = escape(channel)
    return f"<a href=\"https://t.me/{value}\">@{value}</a>"


def channel_from_text(value: str) -> str | None:
    for token in value.split():
        clean = clean_username(token.strip(".,;:()[]<>"))
        if clean:
            return clean
    return None


def page_from_callback(value: str) -> int:
    try:
        return int(value.split(":")[-1])
    except Exception:
        return 0


def card_payload(value: str) -> tuple[str, int]:
    parts = value.split(":")
    if len(parts) > 3 and parts[0] == "v1":
        run_id = parts[2]
        page_index = 3
    else:
        run_id = parts[1] if len(parts) > 1 else ""
        page_index = 2
    try:
        page = int(parts[page_index]) if len(parts) > page_index else 0
    except ValueError:
        page = 0
    return run_id, page


def image_payload(value: str) -> str:
    parts = value.split(":")
    if len(parts) > 2 and parts[0] == "v1":
        return parts[2]
    return parts[1] if len(parts) > 1 else ""


def fight_page_payload(value: str) -> tuple[int, int]:
    parts = value.split(":")
    try:
        step = int(parts[2])
        page = int(parts[3])
    except Exception:
        return 1, 0
    return (2 if step == 2 else 1), max(0, page)


def fight_pick_payload(value: str) -> tuple[int, int, int]:
    parts = value.split(":")
    try:
        step = int(parts[2])
        page = int(parts[3])
        index = int(parts[4])
    except Exception:
        return 1, 0, -1
    return (2 if step == 2 else 1), max(0, page), index


def fight_pick_run_payload(value: str) -> tuple[int, str]:
    parts = value.split(":")
    try:
        step = int(parts[2])
    except Exception:
        step = 1
    return (2 if step == 2 else 1), parts[3] if len(parts) > 3 else ""


def fight_search_payload(value: str) -> int:
    parts = value.split(":")
    try:
        step = int(parts[2])
    except Exception:
        step = 1
    return 2 if step == 2 else 1


def fight_search_mode_step(value: str) -> int:
    try:
        step = int(value.split(":", 1)[1])
    except Exception:
        step = 1
    return 2 if step == 2 else 1


def fight_matches_payload(value: str) -> tuple[str, int]:
    parts = value.split(":")
    run_id = parts[2] if len(parts) > 2 else ""
    try:
        page = int(parts[3])
    except Exception:
        page = 0
    return run_id, max(0, page)


def fight_view_payload(value: str) -> str:
    parts = value.split(":")
    return parts[2] if len(parts) > 2 else ""


def set_search_mode(callback: CallbackQuery, search_modes: dict[tuple[int, int], str], mode: str) -> None:
    if callback.from_user and callback.message:
        search_modes[(callback.from_user.id, callback.message.chat.id)] = mode


def clear_search_mode(callback: CallbackQuery, search_modes: dict[tuple[int, int], str]) -> None:
    if callback.from_user and callback.message:
        search_modes.pop((callback.from_user.id, callback.message.chat.id), None)


def clear_search_mode_message(message: Message, search_modes: dict[tuple[int, int], str]) -> None:
    if message.from_user:
        search_modes.pop((message.from_user.id, message.chat.id), None)
