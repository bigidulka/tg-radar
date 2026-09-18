from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TG_RADAR_", env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://tg_radar:tg_radar@localhost:5432/tg_radar"
    vespa_endpoint: str = "http://localhost:8081"
    user_agent: str = "Mozilla/5.0 (compatible; TG-Radar/0.1)"
    requests_per_second: float = Field(default=1.0, gt=0, le=10)
    request_timeout_seconds: float = Field(default=20.0, gt=1)
    external_search_enabled: bool = True
    external_search_provider: str = "auto"
    brave_api_key: str | None = None
    serpapi_key: str | None = None
    searxng_endpoint: str | None = None
    tgstat_token: str | None = None
    telemetr_public_enabled: bool = True
    lyzem_enabled: bool = True
    open_websearch_url: str | None = None
    open_websearch_enabled: bool = True
    open_websearch_engines: str = "startpage,duckduckgo"
    telethon_enabled: bool = False
    telethon_home: str = ".pi-telethon"
    telethon_profile: str = "default"
    telethon_user_search_enabled: bool = False
    telethon_search_limit: int = Field(default=20, ge=1, le=100)
    telethon_flood_sleep_threshold_seconds: int = Field(default=3, ge=0, le=60)
    external_search_tier_concurrency: int = Field(default=4, ge=1, le=16)
    max_external_results_per_keyword: int = Field(default=20, ge=1, le=100)
    max_keyword_expansions: int = Field(default=6, ge=1, le=20)
    discovery_query_cache_ttl_seconds: int = Field(default=3600, ge=0, le=86400)
    ingest_concurrency: int = Field(default=4, ge=1, le=16)
    crawl_cooldown_seconds: int = Field(default=1800, ge=0, le=86400)
    topic_expansion_max_new_per_run: int = Field(default=5, ge=0, le=100)
    topic_expansion_min_messages: int = Field(default=10, ge=1, le=500)
    topic_expansion_min_quality_score: float = Field(default=0.55, ge=0.0, le=1.0)
    topic_expansion_min_match_rate: float = Field(default=0.55, ge=0.0, le=1.0)
    topic_expansion_min_pain_score: float = Field(default=0.45, ge=0.0, le=1.0)
    auto_worker_interval_seconds: float = Field(default=60.0, ge=5, le=3600)
    auto_task_timeout_seconds: float = Field(default=600.0, ge=30, le=3600)
    auto_worker_enabled: bool = True
    agent_run_worker_enabled: bool = True
    agent_run_worker_interval_seconds: float = Field(default=1.0, ge=0.1, le=60)
    agent_run_timeout_seconds: float = Field(default=600.0, ge=10, le=86400)
    api_token: str | None = None
    public_metrics: bool = False
    api_max_concurrent_agent_runs: int = Field(default=3, ge=1, le=100)
    api_max_daily_crawl_budget: int = Field(default=200, ge=1, le=100000)
    bot_token: str | None = None
    bot_api_base_url: str = "http://127.0.0.1:8080"
    bot_state_path: str = "var/bot/airbot.sqlite3"
    bot_generated_images_path: str = "tmp/leaderboard_images"
    bot_generated_images_base_url: str | None = None
    bot_image_model: str = "gpt-image-2"
    bot_image_size: str = "1024x1024"
    bot_image_timeout_seconds: float = Field(default=180.0, ge=5, le=600)
    bot_default_limit: int = Field(default=100, ge=1, le=300)
    bot_max_limit: int = Field(default=300, ge=1, le=300)
    bot_cooldown_seconds: int = Field(default=300, ge=0, le=86400)
    bot_cache_ttl_seconds: int = Field(default=86400, ge=0, le=604800)
    bot_history_ttl_seconds: int = Field(default=604800, ge=3600, le=2592000)
    bot_user_concurrency: int = Field(default=1, ge=1, le=10)
    bot_global_concurrency: int = Field(default=5, ge=1, le=50)
    bot_llm_chunk_size: int = Field(default=40, ge=5, le=100)
    bot_llm_chunk_concurrency: int = Field(default=1, ge=1, le=10)
    bot_llm_global_concurrency: int = Field(default=1, ge=1, le=10)
    bot_fight_max_tokens: int = Field(default=2400, ge=500, le=64000)
    bot_fight_global_concurrency: int = Field(default=5, ge=1, le=50)
    bot_fight_user_concurrency: int = Field(default=1, ge=1, le=10)
    bot_fight_cache_ttl_seconds: int = Field(default=86400, ge=0, le=604800)
    bot_fight_max_elo_gap: int = Field(default=2400, ge=0, le=4000)
    bot_fight_elo_gap_boost_start: int = Field(default=800, ge=0, le=4000)
    bot_fight_elo_gap_boost_max: float = Field(default=1.75, ge=1.0, le=5.0)
    bot_http_timeout_seconds: float = Field(default=180.0, ge=5, le=600)
    crawl_debug_html: bool = False
    embeddings_enabled: bool = False
    embedding_model: str = "BAAI/bge-m3"
    reranker_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    content_factory_enabled: bool = False
    rules_dir: str | None = None
    content_prompt_dir: str | None = None
    content_system_prompt: str = "content_factory_system.md"
    content_outline_task_prompt: str = "content_outline_task.md"
    content_draft_task_prompt: str = "content_draft_task.md"
    content_llm_temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    content_llm_max_tokens: int = Field(default=4096, ge=100, le=64000)
    story_catalog_source: str = "../publish-engine/docs/article-ideas.md"
    story_match_prompt: str = "story_match.md"
    story_match_window_days: int = Field(default=14, ge=1, le=365)
    story_match_signal_limit: int = Field(default=150, ge=1, le=1000)
    story_match_batch_size: int = Field(default=15, ge=1, le=100)
    story_match_concurrency: int = Field(default=3, ge=1, le=16)
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str = "gpt-5.5"
    llm_timeout_seconds: float = Field(default=90.0, ge=5, le=600)
    operator_quality_enabled: bool = False
    operator_quality_message_limit: int = Field(default=40, ge=5, le=200)
    operator_quality_min_messages: int = Field(default=5, ge=1, le=100)
    operator_quality_recheck_interval_days: int = Field(default=7, ge=1, le=365)
    operator_quality_max_per_backfill: int = Field(default=50, ge=0, le=1000)
    agent_runtime_provider: str = "openai_compatible"
    agent_prompt_dir: str | None = None
    agent_collection_system_prompt: str = "collection_agent_system.md"
    agent_context_budget: int = Field(default=120000, ge=1000, le=300000)
    agent_recent_steps_limit: int = Field(default=4, ge=1, le=20)
    agent_recent_tasks_limit: int = Field(default=8, ge=1, le=50)
    agent_compaction_enabled: bool = True
    agent_compaction_recent_steps: int = Field(default=4, ge=1, le=40)
    agent_auto_verify_enabled: bool = False
    agent_verify_tool_name: str = "run_tests"
    agent_workspace_context_enabled: bool = False
    agent_workspace_repo_map_enabled: bool = True
    agent_workspace_retrieval_enabled: bool = True
    agent_workspace_git_diff_enabled: bool = True
    agent_workspace_repo_map_max_entries: int = Field(default=200, ge=1, le=5000)
    agent_workspace_relevant_files_limit: int = Field(default=5, ge=1, le=50)
    agent_workspace_relevant_file_chars: int = Field(default=4000, ge=100, le=50000)
    agent_workspace_diff_chars: int = Field(default=12000, ge=100, le=200000)
    agent_memory_context_enabled: bool = True
    agent_memory_context_max_items: int = Field(default=20, ge=1, le=200)
    agent_model_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    agent_model_max_tokens: int = Field(default=1200, ge=100, le=64000)
    agent_planner_timeout_seconds: float = Field(default=40.0, ge=1.0, le=120.0)
    agent_enabled_tools: str = "core_status,discover,ingest"
    agent_native_tools_enabled: bool = False
    agent_native_workspace: str = "."
    agent_native_enabled_tools: str = (
        "list_files,read_file,search_text,search_symbols,memory_read,memory_write,task_tracker,"
        "checkpoint_create,checkpoint_list,checkpoint_restore,"
        "write_file,apply_patch,git_status,git_diff,run_shell,run_tests"
    )
    agent_mcp_tools_enabled: bool = False
    agent_mcp_endpoint: str | None = None
    agent_mcp_enabled_tools: str = ""
    agent_mcp_timeout_seconds: float = Field(default=30.0, ge=1.0, le=600.0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
