import time
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from tg_radar.api import create_app
from tg_radar.api_admin import router as legacy_admin_router
from tg_radar.agent_core.models import ToolSpec
from tg_radar.agent_runs import AgentRunRegistry
from tg_radar.api_routes.admin import router as admin_router
from tg_radar.api_routes.channels import fetch_channel_messages
from tg_radar.api_routes.deps import get_state
from tg_radar.schemas import AgentActionType, ChannelFetchRequest, CollectionAgentAction, CollectionAgentResponse, CollectionAgentStep, ParsedMessage, ParsedPage


def test_system_router_paths_are_registered():
    app = create_app()
    paths = {route.path for route in app.routes if hasattr(route, "path")}

    for path in {
        "/content/board",
        "/content/cards/generate",
        "/content/cards/{card_id}/action",
        "/content/articles",
        "/content/articles/{article_id}",
        "/content/articles/{article_id}/outline",
        "/content/articles/{article_id}/draft",
        "/content/articles/{article_id}/format-telegram",
        "/content/blocks/{block_id}",
        "/content/drafts/{draft_id}",
        "/search",
        "/core/search",
        "/discover",
        "/ingest",
        "/core/ingest",
        "/candidates",
        "/tasks/{name}/candidates",
        "/core/tasks/{name}/candidates",
        "/expand",
        "/monitor",
        "/topics",
        "/core/topics",
        "/topics/{slug}/backfill",
        "/core/topics/{slug}/backfill",
        "/topics/{slug}/channels",
        "/core/topics/{slug}/channels",
        "/topics/{slug}/pain",
        "/core/topics/{slug}/pain",
        "/topics/{slug}/story-matches",
        "/core/topics/{slug}/story-matches",
        "/vespa/status",
        "/admin/vespa/reindex",
        "/admin/backfill/quality",
        "/stats",
        "/core/stats",
        "/channels/{username}",
        "/channels/{username}/messages",
        "/channels/{username}/fetch-messages",
        "/crawl/{username}",
        "/core/channels/{username}",
        "/core/channels/{username}/messages",
        "/core/channels/{username}/fetch-messages",
        "/core/crawl/{username}",
        "/telethon/status",
        "/telethon/search/posts",
        "/telethon/search/global",
        "/telethon/search/contacts",
        "/telethon/dialogs",
        "/telethon/entity",
        "/telethon/full-info",
        "/telethon/messages",
        "/telethon/forum-topics",
        "/telethon/thread-messages",
        "/telethon/forum-ingest",
        "/telethon/participants",
        "/telethon/participant-profiles",
        "/telethon/common-chats",
        "/telethon/similar",
        "/telethon/inspect",
        "/telethon/candidates",
        "/tasks",
        "/tasks/{name}",
        "/tasks/{name}/run",
        "/engine/start",
        "/engine/stop",
        "/engine/status",
        "/agent/run",
        "/agent/runs",
        "/agent/runs/{run_id}",
        "/agent/tools",
        "/core/status",
        "/core/tasks",
        "/core/tasks/{name}",
        "/core/tasks/{name}/run",
        "/core/engine/start",
        "/core/engine/stop",
        "/core/engine/status",
        "/core/agent/run",
        "/core/agent/runs",
        "/core/agent/runs/{run_id}",
        "/core/agent/tools",
    }:
        assert path in paths


def test_api_route_modules_are_importable_from_new_and_legacy_paths():
    assert admin_router is legacy_admin_router
    assert callable(get_state)


class EmptyAgentStore:
    async def events(self, run_id: str):
        return []


class FakeOrchestrator:
    state_store = EmptyAgentStore()


class FakeToolRegistry:
    def specs(self):
        return [ToolSpec(name="core_status", description="Core status")]


class FakeAgent:
    tool_registry = FakeToolRegistry()
    orchestrator = FakeOrchestrator()

    async def run(self, request):
        return CollectionAgentResponse(
            run_id=request.run_id,
            final="ok",
            steps=[
                CollectionAgentStep(
                    step=1,
                    action=CollectionAgentAction(type=AgentActionType.final, reason="done"),
                )
            ],
        )


class FakeState:
    def __init__(self) -> None:
        self.agent = FakeAgent()
        self.agent_runs = AgentRunRegistry()


def test_async_agent_run_api_returns_status_and_steps():
    app = create_app()
    fake_state = FakeState()
    app.dependency_overrides[get_state] = lambda: fake_state
    client = TestClient(app)

    started = client.post("/core/agent/runs", json={"goal": "status", "mode": "status"}).json()

    assert started["run_id"].startswith("run_")
    assert started["status"] == "running"

    status = {}
    for _ in range(20):
        status = client.get(f"/core/agent/runs/{started['run_id']}").json()
        if status["status"] == "completed":
            break
        time.sleep(0.02)

    assert status["status"] == "completed"
    assert status["final"] == "ok"
    assert status["steps"][0]["action"]["type"] == "final"
    assert client.get("/core/agent/tools").json()[0]["name"] == "core_status"


class FakeCrawler:
    def __init__(self, pages=None, error=None) -> None:
        self.pages = pages or []
        self.error = error
        self.calls = []

    async def crawl_channel(self, channel, pages=1):
        self.calls.append((channel, pages))
        if self.error:
            raise self.error
        return self.pages


class FakeFetchState:
    def __init__(self, crawler) -> None:
        self.crawler = crawler


@pytest.mark.asyncio
async def test_fetch_channel_messages_returns_transient_crawler_messages():
    posted_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    crawler = FakeCrawler(
        [
            ParsedPage(
                channel_username="demochannel",
                channel_title="Demo",
                next_before=None,
                messages=[
                    ParsedMessage(
                        channel_username="demochannel",
                        channel_title="Demo",
                        tg_msg_id=10,
                        url="https://t.me/demochannel/10",
                        posted_at=posted_at,
                        text="hello",
                        views=42,
                        links=["https://example.com"],
                        mentions=["other"],
                        forward_from="source",
                        content_hash="hash",
                    )
                ],
            )
        ]
    )

    response = await fetch_channel_messages("demochannel", ChannelFetchRequest(limit=1), FakeFetchState(crawler))

    assert response.channel == "demochannel"
    assert response.source == "tme"
    assert response.messages[0].url == "https://t.me/demochannel/10"
    assert crawler.calls == [("demochannel", 1)]


@pytest.mark.asyncio
async def test_fetch_channel_messages_rejects_invalid_channel():
    with pytest.raises(Exception) as exc:
        await fetch_channel_messages("bad handle", ChannelFetchRequest(), FakeFetchState(FakeCrawler()))

    assert getattr(exc.value, "status_code", None) == 400


@pytest.mark.asyncio
async def test_fetch_channel_messages_maps_not_found_to_400():
    with pytest.raises(Exception) as exc:
        await fetch_channel_messages("missing", ChannelFetchRequest(), FakeFetchState(FakeCrawler(error=FileNotFoundError("missing"))))

    assert getattr(exc.value, "status_code", None) == 400
