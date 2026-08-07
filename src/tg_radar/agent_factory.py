from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from tg_radar.agent import (
    CollectionContextBuilder,
    DataCollectionAgent,
    OpenAICompatibleCollectionRuntime,
)
from tg_radar.agent_runtime import LangGraphRuntime, OpenAITransportConfig, ResponsesTransport
from tg_radar.agent_runtime.openai_compatible import OpenAICompatibleRuntime
from tg_radar.agent_runtime.prompt_loader import PromptLoader
from tg_radar.agent_runtime.rule_based import RuleBasedCollectionRuntime
from tg_radar.agent_state import SqlAlchemyAgentStateStore
from tg_radar.agent_tools import CollectionToolRegistry, CompositeToolRegistry, ContentToolRegistry, HttpMcpToolClient, McpToolRegistry, NativeDevToolRegistry
from tg_radar.agent_core import (
    AgentVerifier,
    NoOpAgentVerifier,
    NoOpStepCompactor,
    StepCompactor,
    ToolResultVerifier,
    WindowStepCompactor,
)
from tg_radar.agent_core.ports import ToolRegistry
from tg_radar.auto_search import AutoSearchEngine
from tg_radar.config import Settings
from tg_radar.discovery import Discoverer
from tg_radar.rule_loader import load_rules
from tg_radar.service import IngestService


def build_collection_agent(
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    discoverer: Discoverer,
    ingesting: IngestService,
    auto_search: AutoSearchEngine,
) -> DataCollectionAgent:
    tool_rules = load_rules("agent_tool_rules.json").get("native_dev_tools", {})
    return DataCollectionAgent(
        CollectionContextBuilder(
            sessionmaker,
            auto_search,
            settings.agent_context_budget,
            settings.agent_recent_steps_limit,
            settings.agent_recent_tasks_limit,
            workspace_context_enabled=settings.agent_workspace_context_enabled or settings.agent_native_tools_enabled,
            workspace=settings.agent_native_workspace,
            workspace_exclude_dirs=set(tool_rules.get("exclude_dirs") or []),
            workspace_repo_map_enabled=settings.agent_workspace_repo_map_enabled,
            workspace_retrieval_enabled=settings.agent_workspace_retrieval_enabled,
            workspace_git_diff_enabled=settings.agent_workspace_git_diff_enabled,
            workspace_repo_map_max_entries=settings.agent_workspace_repo_map_max_entries,
            workspace_relevant_files_limit=settings.agent_workspace_relevant_files_limit,
            workspace_relevant_file_chars=settings.agent_workspace_relevant_file_chars,
            workspace_diff_chars=settings.agent_workspace_diff_chars,
            memory_context_enabled=settings.agent_memory_context_enabled,
            memory_state_dir=str(tool_rules.get("state_dir") or ".tg-radar-agent"),
            memory_file=str(tool_rules.get("memory_file") or "memory.json"),
            tasks_file=str(tool_rules.get("tasks_file") or "tasks.json"),
            memory_context_max_items=settings.agent_memory_context_max_items,
        ),
        build_tool_registry(settings, discoverer, ingesting, auto_search, sessionmaker=sessionmaker),
        build_runtime(settings),
        SqlAlchemyAgentStateStore(sessionmaker),
        build_compactor(settings),
        build_verifier(settings),
    )


def build_runtime(settings: Settings):
    if settings.agent_runtime_provider == "langgraph":
        return LangGraphRuntime(fallback=_core_runtime(settings) or RuleBasedCollectionRuntime())
    if settings.agent_runtime_provider == "openai_responses":
        return _core_responses_runtime(settings)
    if settings.agent_runtime_provider not in {"openai_compatible", "openai_chat"}:
        return None
    if not settings.llm_base_url:
        return None
    return _collection_openai_runtime(settings)


def _collection_openai_runtime(settings: Settings) -> OpenAICompatibleCollectionRuntime:
    return OpenAICompatibleCollectionRuntime(
        settings.llm_base_url,
        settings.llm_api_key or "sk-local-dev-key",
        settings.llm_model,
        settings.llm_timeout_seconds,
        settings.agent_collection_system_prompt,
        settings.agent_prompt_dir,
        settings.agent_model_temperature,
        settings.agent_model_max_tokens,
        settings.agent_planner_timeout_seconds,
    )


def _core_runtime(settings: Settings):
    if not settings.llm_base_url:
        return None
    return OpenAICompatibleRuntime(
        settings.llm_base_url,
        settings.llm_api_key or "sk-local-dev-key",
        settings.llm_model,
        settings.llm_timeout_seconds,
        settings.agent_collection_system_prompt,
        PromptLoader(override_dir=settings.agent_prompt_dir),
        settings.agent_model_temperature,
        settings.agent_model_max_tokens,
        settings.agent_planner_timeout_seconds,
    )


def _core_responses_runtime(settings: Settings):
    if not settings.llm_base_url:
        return None
    config = OpenAITransportConfig(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key or "sk-local-dev-key",
        model=settings.llm_model,
        timeout_seconds=settings.llm_timeout_seconds,
        temperature=settings.agent_model_temperature,
        max_tokens=settings.agent_model_max_tokens,
    )
    return OpenAICompatibleRuntime(
        settings.llm_base_url,
        settings.llm_api_key or "sk-local-dev-key",
        settings.llm_model,
        settings.llm_timeout_seconds,
        settings.agent_collection_system_prompt,
        PromptLoader(override_dir=settings.agent_prompt_dir),
        settings.agent_model_temperature,
        settings.agent_model_max_tokens,
        settings.agent_planner_timeout_seconds,
        transport=ResponsesTransport(config),
    )


def build_compactor(settings: Settings) -> StepCompactor:
    if not settings.agent_compaction_enabled:
        return NoOpStepCompactor()
    return WindowStepCompactor(settings.agent_compaction_recent_steps)


def build_verifier(settings: Settings) -> AgentVerifier:
    if not settings.agent_auto_verify_enabled:
        return NoOpAgentVerifier()
    return ToolResultVerifier(settings.agent_verify_tool_name)


def build_tool_registry(
    settings: Settings,
    discoverer: Discoverer,
    ingesting: IngestService,
    auto_search: AutoSearchEngine,
    sessionmaker: async_sessionmaker[AsyncSession] | None = None,
) -> ToolRegistry:
    collection_tools = CollectionToolRegistry(
        discoverer,
        ingesting,
        auto_search,
        _enabled_tools(settings.agent_enabled_tools),
    )
    registries: list[ToolRegistry] = [collection_tools]
    if settings.content_factory_enabled and sessionmaker is not None:
        registries.append(ContentToolRegistry(settings, sessionmaker))
    if settings.agent_native_tools_enabled:
        registries.append(
            NativeDevToolRegistry(
                settings.agent_native_workspace,
                enabled_tools=_enabled_tools(settings.agent_native_enabled_tools),
            )
        )
    if settings.agent_mcp_tools_enabled and settings.agent_mcp_endpoint:
        rules = load_rules("agent_tool_rules.json").get("mcp_bridge", {})
        registries.append(
            McpToolRegistry(
                HttpMcpToolClient(settings.agent_mcp_endpoint, settings.agent_mcp_timeout_seconds),
                enabled_tools=_enabled_tools(settings.agent_mcp_enabled_tools),
                initial_tools=list(rules.get("tools") or []),
            )
        )
    if len(registries) == 1:
        return collection_tools
    return CompositeToolRegistry(registries)


def _enabled_tools(value: str) -> set[str] | None:
    names = {name.strip() for name in value.split(",") if name.strip()}
    return names or None
