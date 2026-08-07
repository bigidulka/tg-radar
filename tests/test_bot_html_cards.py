from tg_radar.bot.avatar import parse_avatar_url
from tg_radar.bot.html_cards import CARD_HEIGHT, CARD_WIDTH, render_airbot_card_html


def test_render_airbot_card_html_contains_full_json():
    result = {
        "channel": "demochannel",
        "air_elo": 3936,
        "air_elo_label": "3936 ELO",
        "air_rank_measure": "1000 units",
        "niche_tags": ["crypto"],
        "core_traits": ["fomo"],
        "summary": "summary",
        "metrics": {"fomo_pressure": {"score": 90, "why": "why"}},
        "fun_metrics": [{"label": "air", "display": "6724 kwh"}],
        "tactics": [{"name": "hook", "hint": "hint"}],
        "evidence": [{"url": "https://t.me/demochannel/1", "snippet": "snippet"}],
        "meme_profile": {"one_liner": "one liner"},
    }

    html = render_airbot_card_html(result, avatar_url="https://example.com/avatar.jpg")

    assert str(CARD_WIDTH) in html
    assert str(CARD_HEIGHT) in html
    assert "data:image/png;base64," in html
    assert "https://example.com/avatar.jpg" in html
    assert "raw_json" in html
    assert "fomo_pressure" in html
    assert "6724 kwh" in html


def test_parse_avatar_url_from_tme_html():
    html = """
    <html><head><meta property="og:image" content="//cdn.example/avatar.jpg"></head>
    <body><div class="tgme_page_photo_image"><img src="/avatar/local.jpg"></div></body></html>
    """

    assert parse_avatar_url(html) == "https://t.me/avatar/local.jpg"
