import respx
from httpx import Response

from tg_radar.agent import OpenAICompatibleCollectionRuntime, RuleBasedCollectionRuntime, _to_core_request
from tg_radar.agent_core.models import AgentAction, AgentActionType as CoreActionType, AgentContextPack, AgentObservation, AgentRequest, AgentStep, ToolSpec
from tg_radar.agent_runtime.rule_based import RuleBasedCollectionRuntime as CoreRuleBasedCollectionRuntime
from tg_radar.agent_runtime.openai_compatible import OpenAICompatibleRuntime
from tg_radar.agent_tools.content import ContentToolRegistry
from tg_radar.config import Settings
from tg_radar.content_factory import _assign_taxonomy_signal, _cluster_signal_name, _focus_key, _focus_match_score
from tg_radar.schemas import AgentActionType, CollectionAgentMode, CollectionAgentRequest


class FakeTransport:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.calls = 0

    async def complete_json(self, messages):
        self.calls += 1
        return self.responses.pop(0)


async def test_runtime_routes_collect_goal_to_ingest():
    runtime = RuleBasedCollectionRuntime()
    request = CollectionAgentRequest(goal="собери каналы про AI hiring", keywords=["AI hiring"])

    action = await runtime.next_action(request, {}, [])

    assert action.type == AgentActionType.tool_call
    assert action.tool_name == "ingest"
    assert action.args["keywords"] == ["AI hiring"]


async def test_runtime_treats_task_name_as_topic_context_in_auto_mode():
    runtime = RuleBasedCollectionRuntime()
    request = CollectionAgentRequest(
        goal="собери сигналы для статьи",
        keywords=["web3 jobs"],
        task_name="web3-jobs",
        mode=CollectionAgentMode.auto,
    )

    action = await runtime.next_action(request, {}, [])

    assert action.tool_name == "ingest"
    assert action.args["topic"] == "web3-jobs"


async def test_runtime_routes_explicit_create_task_mode_to_create_task():
    runtime = RuleBasedCollectionRuntime()
    request = CollectionAgentRequest(
        goal="мониторь раз в пять минут",
        keywords=["web3 jobs"],
        task_name="web3-jobs",
        mode=CollectionAgentMode.create_task,
    )

    action = await runtime.next_action(request, {}, [])

    assert action.tool_name == "create_task"
    assert action.args["name"] == "web3-jobs"


async def test_runtime_asks_for_input_without_keywords_or_seeds():
    runtime = RuleBasedCollectionRuntime()
    request = CollectionAgentRequest(goal="собери данные")

    action = await runtime.next_action(request, {}, [])

    assert action.type == AgentActionType.ask_user
    assert action.expected_result == "keywords or seed_channels"


async def test_runtime_routes_explicit_tool_request():
    runtime = RuleBasedCollectionRuntime()
    request = CollectionAgentRequest(
        goal="read config",
        tool_name="read_file",
        tool_args={"path": "pyproject.toml"},
    )

    action = await runtime.next_action(request, {}, [])

    assert action.type == AgentActionType.tool_call
    assert action.tool_name == "read_file"
    assert action.args == {"path": "pyproject.toml"}


def test_collection_request_maps_approved_tools_to_core_request():
    request = CollectionAgentRequest(goal="write", topic="vibe-coding-ai", phase="edit", approved_tools=["write_file"])

    core = _to_core_request(request)

    assert core.inputs["topic"] == "vibe-coding-ai"
    assert core.inputs["phase"] == "edit"
    assert core.inputs["approved_tools"] == ["write_file"]


@respx.mock
async def test_openai_runtime_returns_structured_action():
    respx.post("http://localhost:8317/v1/chat/completions").mock(
        return_value=Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"type":"tool_call","tool_name":"ingest",'
                                '"args":{"keywords":["AI hiring"],"seed_channels":[],"depth":1,'
                                '"limit":20,"pages_per_channel":1,"crawl":true},'
                                '"reason":"one-shot collection","expected_result":"messages saved"}'
                            )
                        }
                    }
                ]
            },
        )
    )
    runtime = OpenAICompatibleCollectionRuntime(
        "http://localhost:8317/v1",
        "sk-local-dev-key",
        "gpt-5.5",
        5,
    )

    action = await runtime.next_action(
        CollectionAgentRequest(goal="собери AI hiring", keywords=["AI hiring"], limit=20),
        {"available_tools": [{"name": "ingest"}]},
        [],
    )

    assert action.type == AgentActionType.tool_call
    assert action.tool_name == "ingest"
    assert action.args["limit"] == 20


@respx.mock
async def test_openai_runtime_stops_recurring_tool_for_one_shot_goal_without_rewrite():
    respx.post("http://localhost:8317/v1/chat/completions").mock(
        return_value=Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"type":"tool_call","tool_name":"create_task",'
                                '"args":{"keywords":["AI hiring"]},'
                                '"reason":"model picked recurring","expected_result":"task"}'
                            )
                        }
                    }
                ]
            },
        )
    )
    runtime = OpenAICompatibleCollectionRuntime("http://localhost:8317/v1", "sk-local-dev-key", "gpt-5.5", 5)

    action = await runtime.next_action(
        CollectionAgentRequest(goal="напиши article про AI hiring", keywords=["AI hiring"], limit=20),
        {"available_tools": [{"name": "create_task"}, {"name": "ingest"}]},
        [],
    )

    assert action.type == AgentActionType.stop
    assert action.reason == "unsafe recurring tool for one-shot article goal: create_task"


async def test_openai_runtime_auto_does_not_silent_fallback_on_planner_error():
    runtime = OpenAICompatibleRuntime(
        "http://localhost:8317/v1",
        "sk-local-dev-key",
        "gpt-5.5",
        5,
        "collection_agent_system.md",
        transport=FakeTransport("not json"),
    )

    action = await runtime.next_action(
        AgentRequest(goal="write article", inputs={"topic": "vibe-coding-ai"}),
        AgentContextPack(user_goal="write article", tool_manifest=[ToolSpec(name="ingest", description="Collect")]),
        [],
    )

    assert action.type == CoreActionType.stop
    assert action.reason.startswith("planner failed:")


async def test_openai_runtime_keeps_llm_in_control_after_tool_observation():
    transport = FakeTransport(
        '{"type":"tool_call","tool_name":"review_evidence","args":{"topic":"vibe-coding-ai"},'
        '"reason":"review before writing","expected_result":"quality gate"}'
    )
    runtime = OpenAICompatibleRuntime(
        "http://localhost:8317/v1",
        "sk-local-dev-key",
        "gpt-5.5",
        5,
        "collection_agent_system.md",
        transport=transport,
    )
    context = AgentContextPack(
        user_goal="write article",
        tool_manifest=[
            ToolSpec(name="ingest", description="Collect"),
            ToolSpec(name="review_evidence", description="Review"),
            ToolSpec(name="write_article", description="Write"),
        ],
    )
    steps = [
        AgentStep(
            step=1,
            action=AgentAction(type=CoreActionType.tool_call, tool_name="ingest", reason="collect"),
            observation=AgentObservation(tool_name="ingest", ok=True, summary="messages saved"),
        )
    ]

    action = await runtime.next_action(
        AgentRequest(goal="write article", inputs={"topic": "vibe-coding-ai", "keywords": ["vibe coding"]}),
        context,
        steps,
    )

    assert transport.calls == 1
    assert action.tool_name == "review_evidence"


async def test_openai_runtime_guards_write_article_until_evidence_review_passes():
    transport = FakeTransport(
        '{"type":"tool_call","tool_name":"write_article","args":{"topic":"vibe-coding-ai"},'
        '"reason":"write now","expected_result":"draft"}'
    )
    runtime = OpenAICompatibleRuntime(
        "http://localhost:8317/v1",
        "sk-local-dev-key",
        "gpt-5.5",
        5,
        "collection_agent_system.md",
        transport=transport,
    )
    context = AgentContextPack(
        user_goal="write article",
        tool_manifest=[
            ToolSpec(name="review_evidence", description="Review"),
            ToolSpec(name="write_article", description="Write"),
        ],
    )

    action = await runtime.next_action(
        AgentRequest(goal="write article", inputs={"topic": "vibe-coding-ai"}),
        context,
        [],
    )

    assert action.tool_name == "review_evidence"
    assert action.reason.startswith("guardrail")


def test_content_tool_registry_exposes_granular_article_pipeline():
    registry = ContentToolRegistry(Settings(content_factory_enabled=True), None)  # type: ignore[arg-type]

    names = {tool.name for tool in registry.specs()}

    assert {
        "review_evidence",
        "infer_signal_taxonomy",
        "cluster_signals",
        "build_evidence_map",
        "propose_angles",
        "build_outline",
        "write_draft",
        "verify_claims",
        "final_article",
    }.issubset(names)
    review_schema = next(tool.args_schema for tool in registry.specs() if tool.name == "review_evidence")
    assert "article_focus" in review_schema["properties"]
    assert "focus_keywords" in review_schema["properties"]
    taxonomy_schema = next(tool.args_schema for tool in registry.specs() if tool.name == "cluster_signals")
    assert "taxonomy" in taxonomy_schema["properties"]


def test_focus_scoring_separates_article_intent_from_generic_topic():
    focus = "экономия при покупке подписок"
    generic = "Cursor and Claude Code are useful AI coding agents for vibe coding workflows."
    specific = "ChatGPT Plus подписка дешевле через региональную цену, но оплата и лимиты создают риск доступа."

    generic_score, _ = _focus_match_score(generic, focus)
    specific_score, hits = _focus_match_score(specific, focus)

    assert generic_score < 0.22
    assert specific_score >= 0.22
    assert {"подписк", "дешев", "цен"}.intersection(hits)


def test_taxonomy_assignment_does_not_fall_back_to_legacy_signal_labels():
    item = {
        "text": "Cursor and Claude Code workflow notes without pricing terms.",
        "focus_hits": [],
        "signal_name": "pricing_pain",
    }
    taxonomy = [{"name": "gray_account_market", "inclusion_terms": ["account rental"]}]

    assert _assign_taxonomy_signal(item, taxonomy) is None
    assert _cluster_signal_name(item, taxonomy) == "uncategorized_evidence"
    assert _cluster_signal_name(item, []) == "pricing_pain"


async def test_openai_runtime_passes_article_focus_to_writing_tools():
    transport = FakeTransport(
        '{"type":"tool_call","tool_name":"review_evidence",'
        '"args":{"topic":"vibe-coding-ai","limit":40},'
        '"reason":"review","expected_result":"focused evidence"}'
    )
    runtime = OpenAICompatibleRuntime(
        "http://localhost:8317/v1",
        "sk-local-dev-key",
        "gpt-5.5",
        5,
        "collection_agent_system.md",
        transport=transport,
    )

    action = await runtime.next_action(
        AgentRequest(
            goal="напиши статью",
            inputs={
                "topic": "vibe-coding-ai",
                "article_focus": "экономия при покупке AI подписок",
                "focus_keywords": ["подписки", "цены", "лимиты"],
            },
        ),
        AgentContextPack(user_goal="напиши статью", tool_manifest=[ToolSpec(name="review_evidence", description="Review")]),
        [],
    )

    assert action.tool_name == "review_evidence"
    assert action.args["article_focus"] == "экономия при покупке AI подписок"
    assert action.args["focus_keywords"] == ["подписки", "цены", "лимиты"]
    assert _focus_key(action.args["article_focus"], action.args["focus_keywords"])


async def test_rule_based_fallback_continues_granular_article_workflow():
    runtime = CoreRuleBasedCollectionRuntime()
    context = AgentContextPack(
        user_goal="write article",
        tool_manifest=[
            ToolSpec(name="infer_signal_taxonomy", description="Infer taxonomy"),
            ToolSpec(name="cluster_signals", description="Cluster"),
            ToolSpec(name="build_evidence_map", description="Map"),
        ],
    )
    steps = [
        AgentStep(
            step=1,
            action=AgentAction(type=CoreActionType.tool_call, tool_name="review_evidence", reason="review"),
            observation=AgentObservation(
                tool_name="review_evidence",
                ok=True,
                summary="usable=10 rejected=2",
                data={"ok": True, "usable": 10},
            ),
        )
    ]

    action = await runtime.next_action(
        AgentRequest(goal="write article", inputs={"topic": "vibe-coding-ai", "limit": 40}),
        context,
        steps,
    )

    assert action.tool_name == "infer_signal_taxonomy"
    assert action.args["topic"] == "vibe-coding-ai"


async def test_openai_runtime_guards_final_article_until_claims_verified():
    transport = FakeTransport(
        '{"type":"tool_call","tool_name":"final_article","args":{"topic":"vibe-coding-ai"},'
        '"reason":"publish","expected_result":"final"}'
    )
    runtime = OpenAICompatibleRuntime(
        "http://localhost:8317/v1",
        "sk-local-dev-key",
        "gpt-5.5",
        5,
        "collection_agent_system.md",
        transport=transport,
    )
    context = AgentContextPack(
        user_goal="finalize article",
        tool_manifest=[
            ToolSpec(name="verify_claims", description="Verify"),
            ToolSpec(name="final_article", description="Finalize"),
        ],
    )

    action = await runtime.next_action(
        AgentRequest(goal="finalize article", inputs={"topic": "vibe-coding-ai"}),
        context,
        [],
    )

    assert action.tool_name == "verify_claims"
    assert action.reason.startswith("guardrail")


async def test_openai_runtime_does_not_overwrite_model_tool_args_with_request_defaults():
    transport = FakeTransport(
        '{"type":"tool_call","tool_name":"build_evidence_map",'
        '"args":{"topic":"model-topic","limit":17,"min_required":3},'
        '"reason":"map evidence","expected_result":"claims"}'
    )
    runtime = OpenAICompatibleRuntime(
        "http://localhost:8317/v1",
        "sk-local-dev-key",
        "gpt-5.5",
        5,
        "collection_agent_system.md",
        transport=transport,
    )

    action = await runtime.next_action(
        AgentRequest(goal="write article", inputs={"topic": "request-topic", "limit": 99}),
        AgentContextPack(
            user_goal="write article",
            tool_manifest=[ToolSpec(name="build_evidence_map", description="Map")],
        ),
        [],
    )

    assert action.tool_name == "build_evidence_map"
    assert action.args["topic"] == "model-topic"
    assert action.args["limit"] == 17
    assert action.args["min_required"] == 3


async def test_openai_runtime_uses_latest_article_id_for_verify_tools():
    transport = FakeTransport(
        '{"type":"tool_call","tool_name":"verify_claims",'
        '"args":{"topic":"vibe-coding-ai","article_id":18},'
        '"reason":"verify","expected_result":"checks"}'
    )
    runtime = OpenAICompatibleRuntime(
        "http://localhost:8317/v1",
        "sk-local-dev-key",
        "gpt-5.5",
        5,
        "collection_agent_system.md",
        transport=transport,
    )
    steps = [
        AgentStep(
            step=1,
            action=AgentAction(type=CoreActionType.tool_call, tool_name="write_draft", reason="draft"),
            observation=AgentObservation(
                tool_name="write_draft",
                ok=True,
                summary="draft_id=26",
                data={"ok": True, "article_id": 26, "draft_id": 26},
            ),
        )
    ]

    action = await runtime.next_action(
        AgentRequest(goal="finalize article", inputs={"topic": "vibe-coding-ai"}),
        AgentContextPack(user_goal="finalize article", tool_manifest=[ToolSpec(name="verify_claims", description="Verify")]),
        steps,
    )

    assert action.tool_name == "verify_claims"
    assert action.args["article_id"] == 26


async def test_openai_runtime_treats_final_article_as_terminal_tool_result():
    transport = FakeTransport(
        '{"type":"tool_call","tool_name":"ingest","args":{},'
        '"reason":"should not be called","expected_result":"never"}'
    )
    runtime = OpenAICompatibleRuntime(
        "http://localhost:8317/v1",
        "sk-local-dev-key",
        "gpt-5.5",
        5,
        "collection_agent_system.md",
        transport=transport,
    )
    steps = [
        AgentStep(
            step=1,
            action=AgentAction(type=CoreActionType.tool_call, tool_name="final_article", reason="finalize"),
            observation=AgentObservation(
                tool_name="final_article",
                ok=True,
                summary="final=True article_id=27 draft_id=27",
                data={"ok": True, "final": True, "article_id": 27, "draft_id": 27},
            ),
        )
    ]

    action = await runtime.next_action(
        AgentRequest(goal="finalize article", inputs={"topic": "vibe-coding-ai"}),
        AgentContextPack(user_goal="finalize article", tool_manifest=[ToolSpec(name="ingest", description="Collect")]),
        steps,
    )

    assert transport.calls == 0
    assert action.type == CoreActionType.final
    assert action.reason == "final=True article_id=27 draft_id=27"


async def test_openai_runtime_forces_final_article_after_verified_claims():
    transport = FakeTransport(
        '{"type":"final","tool_name":null,"args":{},'
        '"reason":"claims verified","expected_result":null}'
    )
    runtime = OpenAICompatibleRuntime(
        "http://localhost:8317/v1",
        "sk-local-dev-key",
        "gpt-5.5",
        5,
        "collection_agent_system.md",
        transport=transport,
    )
    steps = [
        AgentStep(
            step=1,
            action=AgentAction(type=CoreActionType.tool_call, tool_name="verify_claims", reason="verify"),
            observation=AgentObservation(
                tool_name="verify_claims",
                ok=True,
                summary="checks=3 unsupported=0 missing_refs=0",
                data={"ok": True, "article_id": 42, "checks": [1, 2, 3]},
            ),
        )
    ]

    action = await runtime.next_action(
        AgentRequest(goal="final article", inputs={"topic": "vibe-coding-ai"}),
        AgentContextPack(user_goal="final article", tool_manifest=[ToolSpec(name="final_article", description="Finalize")]),
        steps,
    )

    assert action.type == CoreActionType.tool_call
    assert action.tool_name == "final_article"
    assert action.args["article_id"] == 42
