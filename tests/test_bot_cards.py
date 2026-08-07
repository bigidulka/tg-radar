from io import BytesIO

from PIL import Image

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from tg_radar.bot.cards import AirbotCardRenderer, leaderboard_html, result_card_caption, result_screen_html
from tg_radar.bot.handlers import (
    FIGHT_CALLBACK,
    FIGHT_MATCHES_CALLBACK,
    FIGHT_SEARCH_CALLBACK,
    IMAGE_CALLBACK,
    LEADERBOARD_CALLBACK,
    LEADERBOARD_SEARCH_CALLBACK,
    back_menu,
    fight_pick_menu,
    leaderboard_menu,
    main_menu,
    result_menu,
)
from tg_radar.bot.renderer import AirbotTelegramRenderer


def test_airbot_card_renderer_outputs_png():
    result = {
        "channel": "demochannel",
        "message_count": 42,
        "air_elo": 1816,
        "air_elo_label": "1816 ELO",
        "air_elo_emoji": "x",
        "confidence": 0.75,
        "humorous_label": "demo",
        "summary": "short summary",
        "metrics": {
            "fomo_pressure": {"score": 80, "why": "x"},
            "ragebait": {"score": 70, "why": "x"},
            "product_warmup": {"score": 60, "why": "x"},
        },
        "fun_metrics": [{"label": "air", "display": "10 kWh"}],
        "tactics": [{"hint": "keeps attention"}],
        "evidence": [{"snippet": "evidence"}],
        "meme_profile": {
            "rank_name": "test rank",
            "air_title": "test title",
            "weather_metaphor": "test metaphor",
            "one_liner": "test one liner",
        },
        "disclaimer": "joke",
    }

    png = AirbotCardRenderer().render(result)

    assert png.startswith(b"\x89PNG")
    image = Image.open(BytesIO(png))
    assert image.size == (1200, 1600)
    assert result_card_caption(result).startswith('<a href="https://t.me/demochannel">@demochannel</a>')


def test_result_screen_html_contains_all_metrics_table():
    result = {
        "channel": "demochannel",
        "air_elo": 1816,
        "air_elo_label": "1816 ELO",
        "metrics": {f"metric_{index}": {"score": index, "why": "x"} for index in range(12)},
        "fun_metrics": [{"label": "air", "display": "10 kWh"}],
        "meme_profile": {"rank_name": "rank"},
    }

    html = result_screen_html(result)

    assert "<table bordered striped>" in html
    assert html.count("<tr><td>metric_") == 12
    assert "10 kWh" in html
    assert '<a href="https://t.me/demochannel">@demochannel</a>' in html


def test_result_screen_html_uses_rich_channel_icon_and_avatar_media_block():
    html = result_screen_html(
        {
            "channel": "demochannel",
            "air_elo": 1816,
            "metrics": {},
            "meme_profile": {},
        },
        avatar_url="https://example.com/avatar.jpg",
    )

    assert '<figure><img src="https://example.com/avatar.jpg"/>' in html
    assert '<tg-emoji emoji-id="5206208353751024833">' in html
    assert '<a href="https://t.me/demochannel">@demochannel</a>' in html


def test_result_screen_html_skips_non_http_avatar_url():
    html = result_screen_html(
        {
            "channel": "demochannel",
            "air_elo": 1816,
            "metrics": {},
            "meme_profile": {},
        },
        avatar_url="tg://emoji?id=1",
    )

    assert "<img" not in html


def test_result_screen_html_omits_disclaimer():
    html = result_screen_html(
        {
            "channel": "demochannel",
            "air_elo": 1816,
            "metrics": {},
            "meme_profile": {},
            "disclaimer": "joke disclaimer",
        }
    )

    assert "joke disclaimer" not in html


def test_rich_keyboard_payload_adds_button_style_and_icon():
    markup = InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="Run", callback_data="v1:analyze")]]
    )

    payload = AirbotTelegramRenderer()._reply_markup_payload(markup)
    button = payload["inline_keyboard"][0][0]

    assert button["style"] == "success"
    assert button["icon_custom_emoji_id"]


def test_back_button_uses_back_icon_even_for_leaderboard_callback():
    markup = back_menu()
    markup.inline_keyboard[0][0].callback_data = "v1:lb:0"

    payload = AirbotTelegramRenderer()._reply_markup_payload(markup)
    button = payload["inline_keyboard"][0][0]

    assert button["icon_custom_emoji_id"] == "5877536313623711363"


def test_result_menu_has_image_button_when_run_id_exists():
    markup = result_menu("v1:home", "bot_run")

    assert markup.inline_keyboard[0][0].callback_data == f"{IMAGE_CALLBACK}:bot_run"
    assert markup.inline_keyboard[1][0].callback_data == f"{FIGHT_MATCHES_CALLBACK}:bot_run:0"


def test_main_menu_has_fight_button():
    markup = main_menu()

    assert any(button.callback_data == FIGHT_CALLBACK for row in markup.inline_keyboard for button in row)
    assert any(button.callback_data == LEADERBOARD_CALLBACK for row in markup.inline_keyboard for button in row)
    assert any(button.callback_data == LEADERBOARD_SEARCH_CALLBACK for row in markup.inline_keyboard for button in row)


def test_leaderboard_links_channel_mentions():
    class Run:
        channel = "demochannel"
        result = {"air_elo": 1200}

    html = leaderboard_html([Run()], page=0, has_prev=False, has_next=False, fight_stats={"demochannel": {"wins": 2, "losses": 1}})

    assert '<a href="https://t.me/demochannel"><b>@demochannel</b></a>' in html
    assert "<td align=\"right\">2/1</td>" in html


def test_leaderboard_and_fight_picker_have_search_buttons():
    class Run:
        run_id = "bot_run"
        channel = "demochannel"
        result = {"air_elo": 1200}

    lb = leaderboard_menu([Run()], page=0, has_next=False)
    fight = fight_pick_menu([Run()], page=0, has_next=False, step=1)

    assert lb.inline_keyboard[0][0].callback_data == LEADERBOARD_SEARCH_CALLBACK
    assert fight.inline_keyboard[0][0].callback_data == f"{FIGHT_SEARCH_CALLBACK}:1"
