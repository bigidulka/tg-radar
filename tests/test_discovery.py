from tg_radar.discovery import Discoverer, expand_keyword_queries
from tg_radar.parser import parse_tme_page
from tg_radar.schemas import ChannelCandidate


class FakeCrawler:
    def __init__(self, html: str) -> None:
        self.html = html

    async def crawl_channel(self, channel: str, pages: int = 1):
        return [parse_tme_page(self.html, channel)]


class FakeSearchProvider:
    async def search(self, keyword: str, limit: int):
        return [ChannelCandidate(username="search_chan", source="fake", reason=keyword)]


async def test_discoverer_expands_seed_links():
    html = """
    <div class="tgme_widget_message" data-post="seedchan/10">
      <div class="tgme_widget_message_text">Смотрите @target_chan https://t.me/other_chan/1</div>
      <time datetime="2026-01-01T00:00:00+00:00"></time>
    </div>
    """
    discoverer = Discoverer(FakeCrawler(html), FakeSearchProvider(), 10)

    candidates = await discoverer.expand_from_seeds(["seedchan"], depth=1, limit=10)
    usernames = {c.username for c in candidates}

    assert {"seedchan", "other_chan"} <= usernames
    assert "target_chan" not in usernames


async def test_discoverer_adds_external_keyword_results():
    discoverer = Discoverer(FakeCrawler(""), FakeSearchProvider(), 10)
    candidates = await discoverer.discover(["Яндекс нанимает"], [], depth=0, limit=5)
    assert candidates[0].username == "search_chan"


def test_expand_keyword_queries_for_job_terms():
    queries = expand_keyword_queries("web3 job")
    assert "web3 job" in queries
    assert "web3 jobs" in queries
    assert "web3 вакансии" in queries
    assert "web3 работа" in queries
    assert "web3 remote" in queries
