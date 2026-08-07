from tg_radar.text import clean_username, content_hash, extract_mentions, normalize_link, normalize_text


def test_normalize_text_collapses_spaces():
    assert normalize_text("  Яндекс\n\nнанимает\xa0AI  ") == "Яндекс нанимает AI"


def test_extract_mentions_from_mentions_and_links():
    text = "Пишите @aostrikov и смотрите https://t.me/ai_grably/481"
    assert extract_mentions(text) == ["ai_grably", "aostrikov"]


def test_clean_username_rejects_invalid_values():
    assert clean_username("@aostrikov_ai_agents") == "aostrikov_ai_agents"
    assert clean_username("https://t.me/s/ai_grably/481") == "ai_grably"
    assert clean_username("../bad") is None


def test_content_hash_is_stable():
    assert content_hash("A", 1, "x  y") == content_hash("a", 1, "x y")


def test_normalize_link_handles_relative_tme_links():
    assert normalize_link("/aostrikov_ai_agents/117") == "https://t.me/aostrikov_ai_agents/117"
