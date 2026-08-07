from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class ChannelStatus(StrEnum):
    pending = "pending"
    active = "active"
    blocked = "blocked"
    not_found = "not_found"
    error = "error"


class CandidateState(StrEnum):
    raw_candidate = "raw_candidate"
    validated_channel = "validated_channel"
    indexed_channel = "indexed_channel"
    rejected_user_or_bot = "rejected_user_or_bot"
    empty_public = "empty_public"


class EdgeType(StrEnum):
    mention = "mention"
    link = "link"
    forward = "forward"
    seed = "seed"
    search = "search"


class CrawlMode(StrEnum):
    refresh = "refresh"
    backfill = "backfill"


class ParsedMessage(BaseModel):
    channel_username: str
    channel_title: str | None = None
    tg_msg_id: int
    url: str
    posted_at: datetime | None = None
    text: str
    views: int | None = None
    links: list[str] = Field(default_factory=list)
    mentions: list[str] = Field(default_factory=list)
    forward_from: str | None = None
    sender_id: int | None = None
    sender_name: str | None = None
    thread_id: int | None = None
    reply_to_msg_id: int | None = None
    content_hash: str


class ParsedPage(BaseModel):
    channel_username: str
    channel_title: str | None = None
    messages: list[ParsedMessage]
    next_before: int | None = None


class ChannelCandidate(BaseModel):
    username: str
    source: str
    reason: str
    score: float = 1.0
    depth: int = 0
    task_name: str | None = None
    state: CandidateState = CandidateState.raw_candidate
    quality_score: float = 0.0
    quality_reasons: list[str] = Field(default_factory=list)
    operator_niche_category: str | None = None
    operator_niche_tags: list[str] = Field(default_factory=list)
    operator_niche_confidence: float = 0.0


class SearchFilters(BaseModel):
    channel_usernames: list[str] | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None


class SearchRequest(BaseModel):
    query: str
    filters: SearchFilters = Field(default_factory=SearchFilters)
    limit: int = Field(default=10, ge=1, le=100)
    semantic: bool = True
    rerank: bool = True


class SearchHit(BaseModel):
    message_id: str
    channel_username: str
    channel_title: str | None = None
    url: str
    posted_at: datetime | None = None
    text: str
    score: float
    highlights: list[str] = Field(default_factory=list)
    score_reasons: list[str] = Field(default_factory=list)
    why_matched: list[str] = Field(default_factory=list)
    cluster_key: str | None = None
    cluster_size: int = 1
    pain_score: float = 0.0
    pain_type: str | None = None
    intent: str | None = None
    discussion_key: str | None = None


class ChannelFetchRequest(BaseModel):
    limit: int = Field(default=100, ge=1, le=300)


class ChannelFetchedMessage(BaseModel):
    message_id: str
    channel_username: str
    channel_title: str | None = None
    url: str
    posted_at: datetime | None = None
    text: str
    views: int | None = None
    links: list[str] = Field(default_factory=list)
    mentions: list[str] = Field(default_factory=list)
    forward_from: str | None = None


class ChannelFetchResponse(BaseModel):
    channel: str
    source: str = "tme"
    messages: list[ChannelFetchedMessage]


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]


class DiscoverRequest(BaseModel):
    keywords: list[str] = Field(default_factory=list)
    seed_channels: list[str] = Field(default_factory=list)
    depth: int = Field(default=1, ge=0, le=5)
    limit: int = Field(default=50, ge=1, le=1000)


class DiscoverResponse(BaseModel):
    candidates: list[ChannelCandidate]


class IngestRequest(BaseModel):
    topic: str | None = None
    task_name: str | None = None
    keywords: list[str] = Field(default_factory=list)
    seed_channels: list[str] = Field(default_factory=list)
    depth: int = Field(default=1, ge=0, le=5)
    limit: int = Field(default=50, ge=1, le=1000)
    pages_per_channel: int = Field(default=1, ge=1, le=20)
    crawl: bool = True
    crawl_mode: CrawlMode = CrawlMode.backfill
    freshness_days: int | None = Field(default=None, ge=1, le=3650)
    since: datetime | None = None
    max_live_crawl: int | None = Field(default=None, ge=0, le=1000)


class IngestResponse(BaseModel):
    candidates_found: int
    valid_candidates: int = 0
    junk_candidates: int = 0
    channels_crawled: int
    messages_saved: int
    cached_messages_used: int = 0
    edges_saved: int
    errors: list[str] = Field(default_factory=list)
    candidates: list[ChannelCandidate] = Field(default_factory=list)
    freshness_days: int | None = None
    newest_message_at: datetime | None = None
    oldest_included_message_at: datetime | None = None
    source_count: int = 0
    message_count: int = 0
    deleted_or_missing_marked: int = 0


class CollectionAgentMode(StrEnum):
    auto = "auto"
    status = "status"
    discover = "discover"
    ingest = "ingest"
    create_task = "create_task"
    run_task = "run_task"


class CollectionAgentRequest(BaseModel):
    run_id: str | None = None
    goal: str
    mode: CollectionAgentMode = CollectionAgentMode.auto
    phase: str | None = None
    topic: str | None = None
    article_focus: str | None = None
    focus_keywords: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    seed_channels: list[str] = Field(default_factory=list)
    task_name: str | None = None
    depth: int = Field(default=1, ge=0, le=5)
    limit: int = Field(default=50, ge=1, le=1000)
    pages_per_channel: int = Field(default=1, ge=1, le=20)
    crawl: bool = True
    crawl_mode: CrawlMode = CrawlMode.backfill
    freshness_days: int | None = Field(default=None, ge=1, le=3650)
    since: datetime | None = None
    interval_seconds: int = Field(default=300, ge=30, le=86400)
    enabled: bool = True
    tool_name: str | None = None
    tool_args: dict = Field(default_factory=dict)
    approved_tools: list[str] = Field(default_factory=list)
    max_steps: int = Field(default=6, ge=1, le=20)


class AgentActionType(StrEnum):
    tool_call = "tool_call"
    plan_update = "plan_update"
    final = "final"
    ask_user = "ask_user"
    delegate = "delegate"
    stop = "stop"


class CollectionAgentAction(BaseModel):
    type: AgentActionType
    tool_name: str | None = None
    args: dict = Field(default_factory=dict)
    reason: str
    expected_result: str | None = None


class CollectionAgentObservation(BaseModel):
    tool_name: str
    ok: bool
    summary: str
    data: dict = Field(default_factory=dict)


class CollectionAgentStep(BaseModel):
    step: int
    action: CollectionAgentAction
    observation: CollectionAgentObservation | None = None


class CollectionAgentResponse(BaseModel):
    run_id: str | None = None
    final: str
    steps: list[CollectionAgentStep]
    context: dict = Field(default_factory=dict)


class AgentRunStartResponse(BaseModel):
    run_id: str
    status: str


class AgentRunStatusResponse(BaseModel):
    run_id: str
    status: str
    final: str | None = None
    error: str | None = None
    started_at: datetime
    finished_at: datetime | None = None
    steps: list[CollectionAgentStep] = Field(default_factory=list)


class SearchTaskRequest(BaseModel):
    name: str = "default"
    keywords: list[str] = Field(default_factory=list)
    seed_channels: list[str] = Field(default_factory=list)
    auto_tune: bool = True
    depth: int = Field(default=1, ge=0, le=5)
    limit: int = Field(default=50, ge=1, le=1000)
    pages_per_channel: int = Field(default=1, ge=1, le=20)
    crawl: bool = True
    crawl_mode: CrawlMode = CrawlMode.backfill
    freshness_days: int | None = Field(default=None, ge=1, le=3650)
    interval_seconds: int = Field(default=300, ge=30, le=86400)
    enabled: bool = True


class SearchTaskStatus(BaseModel):
    name: str
    keywords: list[str]
    seed_channels: list[str]
    auto_tune: bool = True
    depth: int
    limit: int
    pages_per_channel: int
    crawl: bool
    crawl_mode: CrawlMode = CrawlMode.backfill
    freshness_days: int | None = None
    interval_seconds: int
    enabled: bool
    running: bool
    last_run_at: datetime | None = None
    next_run_at: datetime | None = None
    running_started_at: datetime | None = None
    last_error: str | None = None
    total_runs: int = 0
    total_candidates: int = 0
    total_valid_candidates: int = 0
    total_junk_candidates: int = 0
    precision: float = 0.0
    total_channels_crawled: int = 0
    total_messages_saved: int = 0
    last_candidates_found: int = 0
    last_valid_candidates: int = 0
    last_junk_candidates: int = 0
    last_channels_crawled: int = 0
    last_messages_saved: int = 0
    last_edges_saved: int = 0


class SearchTaskRunResponse(BaseModel):
    task: SearchTaskStatus
    ingest: IngestResponse | None = None


class AgentEvalRunRequest(BaseModel):
    cases: list[dict] = Field(default_factory=list)
    harness_config: dict = Field(default_factory=dict)


class AgentEvalRunStatus(BaseModel):
    id: int
    model: str | None = None
    prompt_hash: str | None = None
    harness_config: dict = Field(default_factory=dict)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    score: float = 0.0
    cases: int = 0
    failures: int = 0


class AgentEvalCaseStatus(BaseModel):
    id: int
    eval_run_id: int
    case_name: str
    request: dict = Field(default_factory=dict)
    expected_tools: list[str] = Field(default_factory=list)
    actual_tools: list[str] = Field(default_factory=list)
    final: str | None = None
    failures: list[str] = Field(default_factory=list)


class DeepHealthStatus(BaseModel):
    ok: bool
    db: bool
    worker: bool
    llm: bool
    vespa: bool
    crawler: bool
    details: dict = Field(default_factory=dict)


class ResearchTopicRequest(BaseModel):
    slug: str
    title: str | None = None
    keywords: list[str] = Field(default_factory=list)
    negative_keywords: list[str] = Field(default_factory=list)
    seed_channels: list[str] = Field(default_factory=list)
    enabled: bool = True


class ResearchTopicStatus(BaseModel):
    slug: str
    title: str | None = None
    keywords: list[str]
    negative_keywords: list[str] = Field(default_factory=list)
    seed_channels: list[str] = Field(default_factory=list)
    enabled: bool = True
    channels: int = 0
    messages: int = 0
    pain_items: int = 0
    avg_pain_score: float = 0.0
    updated_at: datetime | None = None


class PainInsight(BaseModel):
    message_id: str
    channel_username: str
    channel_title: str | None = None
    url: str
    posted_at: datetime | None = None
    text: str
    pain_score: float
    pain_type: str
    intent: str
    reasons: list[str] = Field(default_factory=list)
    highlights: list[str] = Field(default_factory=list)
    discussion_key: str
    thread_id: int | None = None
    reply_to_msg_id: int | None = None


class PainGroup(BaseModel):
    name: str
    count: int


class PainReport(BaseModel):
    topic: str
    total: int
    groups: list[PainGroup]
    items: list[PainInsight]


class ContentCardOut(BaseModel):
    id: int
    topic_slug: str
    card_type: str
    title: str
    summary: str
    body: str
    source_channel: str | None = None
    source_url: str | None = None
    score: float = 0.0
    status: str = "new"
    tags: list[str] = Field(default_factory=list)
    payload: dict = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ContentBoard(BaseModel):
    topic_slug: str
    cards: list[ContentCardOut]
    articles: list["ArticleSessionOut"] = Field(default_factory=list)


class GenerateContentCardsRequest(BaseModel):
    topic_slug: str
    limit: int = Field(default=60, ge=1, le=300)
    include_llm: bool = False


class ContentCardActionRequest(BaseModel):
    action: str
    note: str | None = None

    @field_validator("action")
    @classmethod
    def validate_action(cls, value: str) -> str:
        if value not in {"approve", "reject", "save", "used", "new"}:
            raise ValueError("invalid content card action")
        return value


class CreateArticleRequest(BaseModel):
    topic_slug: str
    title: str | None = None
    audience: str = "developers, tech leads, founders in AI/LLM/vibe coding"
    angle: str | None = None
    card_ids: list[int] = Field(default_factory=list)


class UpdateArticleRequest(BaseModel):
    title: str | None = None
    audience: str | None = None
    angle: str | None = None
    status: str | None = None
    source_card_ids: list[int] | None = None


class ArticleBlockOut(BaseModel):
    id: int
    position: int
    block_type: str
    title: str | None = None
    text: str
    status: str = "draft"
    source_card_ids: list[int] = Field(default_factory=list)


class ArticleDraftOut(BaseModel):
    id: int
    session_id: int
    version: int
    status: str
    format: str
    text: str
    rich_json: dict = Field(default_factory=dict)
    score: float = 0.0
    created_at: datetime | None = None


class ArticleSessionOut(BaseModel):
    id: int
    topic_slug: str
    title: str
    audience: str | None = None
    angle: str | None = None
    status: str
    source_card_ids: list[int] = Field(default_factory=list)
    blocks: list[ArticleBlockOut] = Field(default_factory=list)
    drafts: list[ArticleDraftOut] = Field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class GenerateArticleRequest(BaseModel):
    style: str = "provocative_engineering"
    instruction: str | None = None
    use_llm: bool = True


class UpdateArticleBlockRequest(BaseModel):
    title: str | None = None
    text: str | None = None
    block_type: str | None = None
    status: str | None = None
    position: int | None = None


class UpdateArticleDraftRequest(BaseModel):
    text: str | None = None
    status: str | None = None
    format: str | None = None


class TelegramPreview(BaseModel):
    article_id: int
    draft_id: int
    text: str
    rich_json: dict


class ExpandRequest(BaseModel):
    channel: str
    depth: int = Field(default=1, ge=1, le=5)
    limit: int = Field(default=100, ge=1, le=1000)


class GraphEdge(BaseModel):
    source: str
    target: str
    edge_type: EdgeType
    message_url: str | None = None


class ExpandResponse(BaseModel):
    channel: str
    edges: list[GraphEdge]


class MonitorRequest(BaseModel):
    query: str
    since: datetime | None = None
    limit: int = Field(default=25, ge=1, le=100)


class TelethonSearchRequest(BaseModel):
    query: str
    limit: int = Field(default=20, ge=1, le=100)
    broadcasts_only: bool = False
    groups_only: bool = False
    users_only: bool = False


class TelethonEntityRequest(BaseModel):
    entity: str
    query: str | None = None
    limit: int = Field(default=20, ge=1, le=100)


class TelethonCommonChatsRequest(BaseModel):
    user: str
    limit: int = Field(default=20, ge=1, le=100)


class TelethonSimilarRequest(BaseModel):
    channel: str
    limit: int = Field(default=20, ge=1, le=100)


class TelethonCandidateRequest(BaseModel):
    entity: str
    query: str | None = None
    limit: int = Field(default=50, ge=1, le=200)


class TelethonThreadRequest(BaseModel):
    entity: str
    thread_id: int
    limit: int = Field(default=50, ge=1, le=200)


class ChannelProfile(BaseModel):
    username: str
    title: str | None = None
    status: ChannelStatus
    first_seen_at: datetime
    last_seen_at: datetime | None = None
    message_count: int = 0
    neighbors: list[str] = Field(default_factory=list)
    operator_niche_category: str | None = None
    operator_income_authenticity: str | None = None
    operator_niche_tags: list[str] = Field(default_factory=list)
    operator_niche_confidence: float = 0.0
    operator_niche_evidence: list[dict] = Field(default_factory=list)
    operator_niche_notes: list[str] = Field(default_factory=list)
    operator_niche_updated_at: datetime | None = None


class StoryCaseOut(BaseModel):
    number: int
    title: str
    format: str
    section: str
    takeaway: str | None = None


class StoryMatchSignalOut(BaseModel):
    message_id: str
    channel: str
    url: str
    posted_at: datetime | None = None
    pain_type: str
    gist: str


class StoryMatchPair(BaseModel):
    case: StoryCaseOut
    signals: list[StoryMatchSignalOut]
    you_add: str
    confidence: str
    channels: list[str] = Field(default_factory=list)
    multi_channel: bool = False


class StoryMatchReport(BaseModel):
    topic: str
    window_days: int
    generated_at: datetime
    signals_considered: int
    signals_matched: int
    cases_total: int
    min_confidence: str
    pairs: list[StoryMatchPair] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
