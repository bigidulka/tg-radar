from __future__ import annotations

import json

from tg_radar.agent import OpenAICompatibleCollectionRuntime
from tg_radar.agent_factory import build_runtime, build_tool_registry
from tg_radar.agent_runtime import LangGraphRuntime, OpenAICompatibleRuntime
from tg_radar.agent_tools import CollectionToolRegistry, CompositeToolRegistry
from tg_radar.config import Settings
from tg_radar.rule_loader import load_rules


class DummyAutoSearch:
    running = False
    active_count = 0


def test_factory_uses_collection_tools_by_default():
    registry = build_tool_registry(
        Settings(agent_enabled_tools="core_status,discover"),
        discoverer=object(),
        ingesting=object(),
        auto_search=DummyAutoSearch(),
    )

    assert isinstance(registry, CollectionToolRegistry)
    assert {tool.name for tool in registry.specs()} == {"core_status", "discover"}


def test_factory_composes_native_tools_when_enabled():
    registry = build_tool_registry(
        Settings(
            agent_enabled_tools="core_status",
            agent_native_tools_enabled=True,
            agent_native_workspace=".",
            agent_native_enabled_tools="read_file,search_text",
        ),
        discoverer=object(),
        ingesting=object(),
        auto_search=DummyAutoSearch(),
    )

    assert isinstance(registry, CompositeToolRegistry)
    assert {tool.name for tool in registry.specs()} == {"core_status", "read_file", "search_text"}


def test_factory_composes_mcp_tools_from_configured_manifest(tmp_path, monkeypatch):
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    (rules_dir / "agent_tool_rules.json").write_text(
        json.dumps(
            {
                "mcp_bridge": {
                    "tools": [
                        {
                            "name": "docs_search",
                            "description": "Search docs",
                            "input_schema": {"type": "object"},
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TG_RADAR_RULES_DIR", str(rules_dir))
    load_rules.cache_clear()

    registry = build_tool_registry(
        Settings(
            agent_enabled_tools="core_status",
            agent_mcp_tools_enabled=True,
            agent_mcp_endpoint="http://mcp.local",
            agent_mcp_enabled_tools="docs_search",
        ),
        discoverer=object(),
        ingesting=object(),
        auto_search=DummyAutoSearch(),
    )

    assert isinstance(registry, CompositeToolRegistry)
    assert {tool.name for tool in registry.specs()} == {"core_status", "docs_search"}
    load_rules.cache_clear()


def test_factory_builds_configured_openai_runtime():
    runtime = build_runtime(
        Settings(
            llm_base_url="http://localhost:8317/v1",
            agent_runtime_provider="openai_chat",
        )
    )

    assert isinstance(runtime, OpenAICompatibleCollectionRuntime)


def test_factory_builds_langgraph_runtime_without_required_dependency():
    runtime = build_runtime(Settings(agent_runtime_provider="langgraph"))

    assert isinstance(runtime, LangGraphRuntime)


def test_factory_builds_openai_responses_runtime():
    runtime = build_runtime(
        Settings(
            llm_base_url="http://localhost:8317/v1",
            agent_runtime_provider="openai_responses",
        )
    )

    assert isinstance(runtime, OpenAICompatibleRuntime)


def test_factory_skips_unknown_runtime_provider():
    runtime = build_runtime(
        Settings(
            llm_base_url="http://localhost:8317/v1",
            agent_runtime_provider="custom",
        )
    )

    assert runtime is None
