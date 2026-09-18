from datetime import datetime, timezone

from tg_radar.db import Channel, Message, ResearchTopic, TopicChannel, add_topic_seed_channels
from tg_radar.schemas import CandidateState, ChannelCandidate, CrawlResult, ParsedMessage
from tg_radar.service import IngestService
from tg_radar.topical import TopicGate, topic_admission


class FakeTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeResult:
    def __init__(self, rows: list) -> None:
        self.rows = rows

    def scalars(self):
        return self

    def all(self):
        return list(self.rows)

    def __iter__(self):
        return iter(self.rows)


class FakeSession:
    """Enough of AsyncSession to run IngestService against real db helpers."""

    CRAWL_POLICY_COLUMNS = ["username", "last_crawled_at", "crawl_interval_seconds"]

    def __init__(self, entities: dict | None = None, rows: list | None = None, policies: list | None = None) -> None:
        self.entities = entities or {}
        self.rows = rows or []
        self.policies = policies or []
        self.executed: list[tuple[str, dict]] = []
        self.added: list = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def begin(self):
        return FakeTransaction()

    async def scalar(self, statement):
        entity = statement.column_descriptions[0]["entity"]
        return self.entities.get(entity.__name__)

    async def execute(self, statement, params=None):
        self.executed.append((str(statement), params or {}))
        columns = [column["name"] for column in getattr(statement, "column_descriptions", [])]
        if columns == self.CRAWL_POLICY_COLUMNS:
            return FakeResult(self.policies)
        return FakeResult(self.rows)

    def add(self, item):
        self.added.append(item)

    async def flush(self):
        return None

    def states(self) -> list[str]:
        return [params["state"] for _, params in self.executed if "state" in params]

    def added_of(self, model) -> list:
        return [item for item in self.added if isinstance(item, model)]


class FakeDiscoverer:
    def __init__(self, candidates: list[ChannelCandidate]) -> None:
        self.candidates = candidates

    async def discover(self, keywords, seed_channels, depth, limit):
        return list(self.candidates)


class FakeIndexing:
    def __init__(self, outcomes: dict) -> None:
        self.outcomes = outcomes
        self.calls: list[str] = []

    async def crawl_and_index(self, session, channel_username, pages=1, crawl_mode="backfill"):
        self.calls.append(channel_username)
        outcome = self.outcomes[channel_username]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def parsed(username: str, tg_msg_id: int, text: str = "post") -> ParsedMessage:
    return ParsedMessage(
        channel_username=username,
        tg_msg_id=tg_msg_id,
        url=f"https://t.me/{username}/{tg_msg_id}",
        posted_at=datetime(2026, 8, 6, 12, 0, tzinfo=timezone.utc),
        text=text,
        content_hash=f"hash-{username}-{tg_msg_id}",
    )


def build_service(outcomes: dict, session: FakeSession, candidates: list[ChannelCandidate]) -> IngestService:
    return IngestService(
        FakeDiscoverer(candidates),
        FakeIndexing(outcomes),
        lambda: session,
        concurrency=1,
        crawl_cooldown_seconds=0,
    )


def seed_candidate(username: str) -> ChannelCandidate:
    return ChannelCandidate(username=username, source="seed", reason="seed/link graph", score=1.0)


async def test_repeat_crawl_without_new_messages_keeps_channel_indexed():
    session = FakeSession()
    candidate = seed_candidate("boris_again")
    service = build_service(
        {"boris_again": CrawlResult(saved=[], seen=[parsed("boris_again", 4043)])},
        session,
        [candidate],
    )

    response = await service.ingest([], ["boris_again"], 0, 5, 1, True, task_name="ai-dev-blog")

    assert CandidateState.empty_public.value not in session.states()
    assert candidate.state is CandidateState.indexed_channel
    assert response.junk_candidates == 0
    assert response.valid_candidates == 1
    assert response.channels_crawled == 1
    assert response.messages_saved == 0


async def test_crawl_marks_empty_public_when_page_has_no_messages():
    session = FakeSession()
    candidate = seed_candidate("deleted_channel")
    service = build_service({"deleted_channel": CrawlResult()}, session, [candidate])

    response = await service.ingest([], ["deleted_channel"], 0, 5, 1, True, task_name="ai-dev-blog")

    assert CandidateState.empty_public.value in session.states()
    assert response.junk_candidates == 1
    assert response.channels_crawled == 0
    assert response.valid_candidates == 0


async def test_unreachable_channel_does_not_stop_the_pass():
    session = FakeSession()
    candidates = [seed_candidate("gone_channel"), seed_candidate("boris_again")]
    service = build_service(
        {
            "gone_channel": FileNotFoundError("gone_channel"),
            "boris_again": CrawlResult(saved=[parsed("boris_again", 4044)], seen=[parsed("boris_again", 4044)]),
        },
        session,
        candidates,
    )

    response = await service.ingest([], ["gone_channel", "boris_again"], 0, 5, 1, True, task_name="ai-dev-blog")

    assert response.channels_crawled == 1
    assert response.messages_saved == 1
    assert any("gone_channel" in error for error in response.errors)
    assert CandidateState.empty_public.value not in session.states()


async def test_expansion_candidate_below_gate_is_rejected_with_a_reason():
    channel = Channel(id=7, username="offtopic_chan", quality_score=0.9)
    stored = Message(id=1, channel_id=7, tg_msg_id=1, pain_score=0.1)
    session = FakeSession(entities={"Channel": channel}, rows=[stored])
    candidate = ChannelCandidate(username="offtopic_chan", source="linked:boris_again", reason="auto graph expansion")
    seen = [parsed("offtopic_chan", 1, "llm agents")] + [parsed("offtopic_chan", index) for index in range(2, 11)]
    service = build_service({"offtopic_chan": CrawlResult(saved=seen, seen=seen)}, session, [candidate])

    response = await service.ingest(["llm"], [], 0, 5, 1, True, task_name="ai-dev-blog")

    assert response.admitted_channels == []
    assert response.rejected_channels == ["@offtopic_chan: topic_gate_match_rate:0.10<0.55"]
    assert candidate.state is CandidateState.rejected_low_quality
    assert session.added_of(TopicChannel) == []
    assert response.valid_candidates == 0
    assert response.messages_saved == len(seen)


async def test_expansion_budget_defers_extra_candidates():
    channel = Channel(id=7, username="ontopic_chan", quality_score=0.9)
    stored = [Message(id=index, channel_id=7, tg_msg_id=index, pain_score=0.9) for index in range(1, 11)]
    session = FakeSession(entities={"Channel": channel}, rows=stored)
    candidates = [
        ChannelCandidate(username=f"ontopic_chan{index}", source="linked:boris_again", reason="auto graph expansion")
        for index in range(2)
    ]
    seen = [parsed("ontopic_chan", index, "llm agents") for index in range(1, 11)]
    service = IngestService(
        FakeDiscoverer(candidates),
        FakeIndexing({candidate.username: CrawlResult(saved=seen, seen=seen) for candidate in candidates}),
        lambda: session,
        concurrency=1,
        crawl_cooldown_seconds=0,
        max_new_channels_per_run=1,
    )

    response = await service.ingest(["llm"], [], 0, 5, 1, True, task_name="ai-dev-blog")

    assert len(response.admitted_channels) == 1
    assert response.rejected_channels == [f"@{response.candidates[1].username}: topic_gate_deferred:expansion budget"]
    assert candidates[1].state is CandidateState.validated_channel


async def test_cooldown_cached_path_applies_the_same_gate():
    channel = Channel(id=7, username="offtopic_chan", quality_score=0.9)
    stored = [Message(id=index, channel_id=7, tg_msg_id=index, text="post", pain_score=0.1) for index in range(1, 11)]
    stored[0].text = "llm agents"
    session = FakeSession(
        entities={"Channel": channel},
        rows=stored,
        policies=[("offtopic_chan", datetime.now(timezone.utc), 1800)],
    )
    candidate = ChannelCandidate(username="offtopic_chan", source="linked:boris_again", reason="auto graph expansion")
    service = IngestService(
        FakeDiscoverer([candidate]),
        FakeIndexing({}),
        lambda: session,
        concurrency=1,
        crawl_cooldown_seconds=1800,
    )

    response = await service.ingest(["llm"], [], 0, 5, 1, True, task_name="ai-dev-blog")

    assert response.rejected_channels == ["@offtopic_chan: topic_gate_match_rate:0.10<0.55"]
    assert session.added_of(TopicChannel) == []
    assert response.cached_messages_used == 0


def test_topic_admission_accepts_a_relevant_channel():
    admission = topic_admission(TopicGate(), seen=20, matched_pain_scores=[0.6] * 15, quality_score=0.8)

    assert admission.accepted
    assert admission.match_rate == 0.75
    assert "topic_gate_pass" in admission.reason


def test_topic_admission_rejects_a_thin_sample_before_scoring():
    admission = topic_admission(TopicGate(), seen=3, matched_pain_scores=[0.9] * 3, quality_score=0.99)

    assert not admission.accepted
    assert admission.reason == "topic_gate_sample:3<10"


def test_topic_admission_rejects_low_pain_channels():
    admission = topic_admission(TopicGate(), seen=20, matched_pain_scores=[0.2] * 18, quality_score=0.9)

    assert not admission.accepted
    assert admission.reason.startswith("topic_gate_pain:")


async def test_admitted_channels_join_the_topic_seed_pool():
    topic = ResearchTopic(slug="ai-dev-blog", seed_channels=["boris_again"])

    added = await add_topic_seed_channels(FakeSession(), topic, ["@new_chan", "BORIS_again", "new_chan"])

    assert added == ["new_chan"]
    assert topic.seed_channels == ["boris_again", "new_chan"]
