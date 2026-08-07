from tg_radar.search_providers import candidates_from_search_items, usernames_from_url, walk_values


def test_usernames_from_tme_urls():
    assert usernames_from_url("https://t.me/s/aostrikov_ai_agents/117") == {"aostrikov_ai_agents"}
    assert usernames_from_url("https://t.me/ai_grably") == {"ai_grably"}


def test_usernames_from_duckduckgo_redirect():
    url = "https://duckduckgo.com/l/?uddg=https%3A%2F%2Ft.me%2Fs%2Fyandexforml%2F1335"
    assert usernames_from_url(url) == {"yandexforml"}


def test_candidates_from_search_items_uses_username_field():
    items = [{"username": "@web3hiring", "title": "Web3 jobs"}]
    candidates = candidates_from_search_items(items, "x", "web3", 5)
    assert candidates[0].username == "web3hiring"


def test_walk_values_for_open_websearch_payload():
    payload = {"results": [{"url": "https://t.me/web3hiring", "description": "jobs @laborx"}]}
    text = " ".join(walk_values(payload))
    assert "https://t.me/web3hiring" in text
    assert "@laborx" in text
