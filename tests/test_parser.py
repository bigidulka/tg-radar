from pathlib import Path

from tg_radar.parser import parse_tme_page, parse_views


FIXTURE = Path(__file__).parent / "fixtures" / "tme_aostrikov.html"


def test_parse_views():
    assert parse_views("4.2K") == 4200
    assert parse_views("981") == 981
    assert parse_views(None) is None


def test_parse_tme_page_extracts_messages():
    page = parse_tme_page(FIXTURE.read_text(), "aostrikov_ai_agents")

    assert page.channel_title == "Остриков пилит агентов"
    assert page.next_before == 71
    assert len(page.messages) == 2

    first = page.messages[0]
    assert first.channel_username == "aostrikov_ai_agents"
    assert first.tg_msg_id == 117
    assert first.url == "https://t.me/aostrikov_ai_agents/117"
    assert first.views == 4200
    assert first.forward_from == "Яндекс нанимает"
    assert first.mentions == ["ai_grably", "aostrikov"]
    assert "Яндексе" in first.text
    assert first.content_hash


def test_parse_tme_page_ignores_video_duration_time():
    html = """
    <div class="tgme_widget_message" data-post="seeallochnaya/3839">
      <div class="tgme_widget_message_text">Видео с разбором</div>
      <time class="message_video_duration js-message_video_duration">1:23</time>
      <div class="tgme_widget_message_meta">
        <time datetime="2026-07-25T11:03:07+00:00">11:03</time>
      </div>
    </div>
    """
    page = parse_tme_page(html, "seeallochnaya")

    assert page.messages[0].posted_at is not None
    assert page.messages[0].posted_at.isoformat() == "2026-07-25T11:03:07+00:00"
