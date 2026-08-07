from __future__ import annotations

from alembic import op

from tg_radar.db import Base

revision = "0001_prod_hardening"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    Base.metadata.create_all(bind=op.get_bind())
    statements = [
        "ALTER TABLE channels ADD COLUMN IF NOT EXISTS last_crawled_at TIMESTAMP WITH TIME ZONE",
        "ALTER TABLE channels ADD COLUMN IF NOT EXISTS quality_score DOUBLE PRECISION DEFAULT 0 NOT NULL",
        "ALTER TABLE channels ADD COLUMN IF NOT EXISTS public_chat BOOLEAN DEFAULT false NOT NULL",
        "ALTER TABLE channels ADD COLUMN IF NOT EXISTS recent_jobs BOOLEAN DEFAULT false NOT NULL",
        "ALTER TABLE channels ADD COLUMN IF NOT EXISTS company_match BOOLEAN DEFAULT false NOT NULL",
        "ALTER TABLE channels ADD COLUMN IF NOT EXISTS ai_llm_match BOOLEAN DEFAULT false NOT NULL",
        "ALTER TABLE channels ADD COLUMN IF NOT EXISTS direct_contact BOOLEAN DEFAULT false NOT NULL",
        "ALTER TABLE channels ADD COLUMN IF NOT EXISTS source_trust DOUBLE PRECISION DEFAULT 0 NOT NULL",
        "ALTER TABLE channels ADD COLUMN IF NOT EXISTS quality_reasons VARCHAR[] DEFAULT '{}' NOT NULL",
        "ALTER TABLE channels ADD COLUMN IF NOT EXISTS crawl_priority INTEGER DEFAULT 100 NOT NULL",
        "ALTER TABLE channels ADD COLUMN IF NOT EXISTS crawl_interval_seconds INTEGER DEFAULT 1800 NOT NULL",
        "ALTER TABLE channels ADD COLUMN IF NOT EXISTS last_successful_crawl_at TIMESTAMP WITH TIME ZONE",
        "ALTER TABLE channels ADD COLUMN IF NOT EXISTS last_error_at TIMESTAMP WITH TIME ZONE",
        "ALTER TABLE channels ADD COLUMN IF NOT EXISTS last_error TEXT",
        "ALTER TABLE search_tasks ADD COLUMN IF NOT EXISTS auto_tune BOOLEAN DEFAULT true NOT NULL",
        "ALTER TABLE search_tasks ADD COLUMN IF NOT EXISTS total_valid_candidates INTEGER DEFAULT 0 NOT NULL",
        "ALTER TABLE search_tasks ADD COLUMN IF NOT EXISTS total_junk_candidates INTEGER DEFAULT 0 NOT NULL",
        "ALTER TABLE search_tasks ADD COLUMN IF NOT EXISTS last_candidates_found INTEGER DEFAULT 0 NOT NULL",
        "ALTER TABLE search_tasks ADD COLUMN IF NOT EXISTS last_valid_candidates INTEGER DEFAULT 0 NOT NULL",
        "ALTER TABLE search_tasks ADD COLUMN IF NOT EXISTS last_junk_candidates INTEGER DEFAULT 0 NOT NULL",
        "ALTER TABLE search_tasks ADD COLUMN IF NOT EXISTS last_channels_crawled INTEGER DEFAULT 0 NOT NULL",
        "ALTER TABLE search_tasks ADD COLUMN IF NOT EXISTS last_messages_saved INTEGER DEFAULT 0 NOT NULL",
        "ALTER TABLE search_tasks ADD COLUMN IF NOT EXISTS last_edges_saved INTEGER DEFAULT 0 NOT NULL",
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS last_seen_at TIMESTAMP WITH TIME ZONE",
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS edited_at TIMESTAMP WITH TIME ZONE",
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS deleted_or_missing BOOLEAN DEFAULT false NOT NULL",
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS last_refreshed_at TIMESTAMP WITH TIME ZONE",
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS source_snapshot_hash VARCHAR(64)",
        "ALTER TABLE discovered_channels ADD COLUMN IF NOT EXISTS task_name VARCHAR(128)",
        "ALTER TABLE discovered_channels ADD COLUMN IF NOT EXISTS state VARCHAR(32) DEFAULT 'raw_candidate' NOT NULL",
        "ALTER TABLE discovered_channels ADD COLUMN IF NOT EXISTS quality_score DOUBLE PRECISION DEFAULT 0 NOT NULL",
        "ALTER TABLE discovered_channels ADD COLUMN IF NOT EXISTS quality_reasons VARCHAR[] DEFAULT '{}' NOT NULL",
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS vacancy_key VARCHAR(64)",
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS detector_score DOUBLE PRECISION DEFAULT 0 NOT NULL",
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS detector_reasons VARCHAR[] DEFAULT '{}' NOT NULL",
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS cluster_id INTEGER REFERENCES vacancy_clusters(id) ON DELETE SET NULL",
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS sender_id BIGINT",
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS sender_name VARCHAR(512)",
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS thread_id BIGINT",
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS reply_to_msg_id BIGINT",
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS pain_score DOUBLE PRECISION DEFAULT 0 NOT NULL",
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS pain_type VARCHAR(64)",
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS intent VARCHAR(64)",
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS pain_reasons VARCHAR[] DEFAULT '{}' NOT NULL",
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS discussion_key VARCHAR(64)",
        "ALTER TABLE search_tasks ADD COLUMN IF NOT EXISTS crawl_mode VARCHAR(32) DEFAULT 'backfill' NOT NULL",
        "ALTER TABLE search_tasks ADD COLUMN IF NOT EXISTS freshness_days INTEGER",
        "ALTER TABLE search_tasks ADD COLUMN IF NOT EXISTS running_started_at TIMESTAMP WITH TIME ZONE",
        """
        CREATE TABLE IF NOT EXISTS agent_runs (
            run_id VARCHAR(128) PRIMARY KEY,
            status VARCHAR(32) DEFAULT 'queued' NOT NULL,
            request_json JSONB DEFAULT '{}'::jsonb NOT NULL,
            final TEXT,
            error TEXT,
            started_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
            finished_at TIMESTAMP WITH TIME ZONE,
            locked_at TIMESTAMP WITH TIME ZONE,
            worker_id VARCHAR(128),
            cancel_requested BOOLEAN DEFAULT false NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS task_runs (
            id SERIAL PRIMARY KEY,
            task_name VARCHAR(128) NOT NULL,
            status VARCHAR(32) NOT NULL,
            started_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
            finished_at TIMESTAMP WITH TIME ZONE,
            ingest_stats JSONB DEFAULT '{}'::jsonb NOT NULL,
            error TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS agent_eval_runs (
            id SERIAL PRIMARY KEY,
            model VARCHAR(128),
            prompt_hash VARCHAR(128),
            harness_config JSONB DEFAULT '{}'::jsonb NOT NULL,
            started_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
            finished_at TIMESTAMP WITH TIME ZONE,
            score DOUBLE PRECISION DEFAULT 0 NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS agent_eval_cases (
            id SERIAL PRIMARY KEY,
            eval_run_id INTEGER REFERENCES agent_eval_runs(id) ON DELETE CASCADE NOT NULL,
            case_name VARCHAR(256) NOT NULL,
            request JSONB DEFAULT '{}'::jsonb NOT NULL,
            expected_tools VARCHAR[] DEFAULT '{}' NOT NULL,
            actual_tools VARCHAR[] DEFAULT '{}' NOT NULL,
            final TEXT,
            failures VARCHAR[] DEFAULT '{}' NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS worker_heartbeats (
            worker_id VARCHAR(128) PRIMARY KEY,
            role VARCHAR(64) DEFAULT 'worker' NOT NULL,
            heartbeat_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
            payload JSONB DEFAULT '{}'::jsonb NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_channels_crawl_policy ON channels (crawl_priority, last_successful_crawl_at)",
        "CREATE INDEX IF NOT EXISTS idx_channels_last_crawled_at ON channels (last_crawled_at)",
        "CREATE INDEX IF NOT EXISTS idx_channels_quality_score ON channels (quality_score)",
        "CREATE INDEX IF NOT EXISTS idx_channels_quality_flags ON channels (public_chat, recent_jobs, company_match, ai_llm_match)",
        "CREATE INDEX IF NOT EXISTS idx_discovered_channels_task_name ON discovered_channels (task_name)",
        "CREATE INDEX IF NOT EXISTS idx_discovered_channels_state ON discovered_channels (state)",
        "CREATE INDEX IF NOT EXISTS idx_discovered_channels_quality_score ON discovered_channels (quality_score)",
        "CREATE INDEX IF NOT EXISTS idx_messages_vacancy_key ON messages (vacancy_key)",
        "CREATE INDEX IF NOT EXISTS idx_messages_detector_score ON messages (detector_score)",
        "CREATE INDEX IF NOT EXISTS idx_messages_cluster_id ON messages (cluster_id)",
        "CREATE INDEX IF NOT EXISTS idx_messages_thread_id ON messages (thread_id)",
        "CREATE INDEX IF NOT EXISTS idx_messages_reply_to_msg_id ON messages (reply_to_msg_id)",
        "CREATE INDEX IF NOT EXISTS idx_messages_pain_score ON messages (pain_score)",
        "CREATE INDEX IF NOT EXISTS idx_messages_pain_type ON messages (pain_type)",
        "CREATE INDEX IF NOT EXISTS idx_messages_intent ON messages (intent)",
        "CREATE INDEX IF NOT EXISTS idx_messages_discussion_key ON messages (discussion_key)",
        "CREATE INDEX IF NOT EXISTS idx_messages_last_seen_at ON messages (last_seen_at)",
        "CREATE INDEX IF NOT EXISTS idx_messages_deleted_or_missing ON messages (deleted_or_missing)",
        "CREATE INDEX IF NOT EXISTS idx_messages_last_refreshed_at ON messages (last_refreshed_at)",
        "CREATE INDEX IF NOT EXISTS idx_topic_channels_topic_score ON topic_channels (topic_id, score DESC)",
        "CREATE INDEX IF NOT EXISTS idx_topic_messages_topic_pain ON topic_messages (topic_id, pain_score DESC)",
        "CREATE INDEX IF NOT EXISTS idx_content_cards_topic_status ON content_cards (topic_slug, status, score DESC)",
        "CREATE INDEX IF NOT EXISTS idx_content_cards_type_score ON content_cards (card_type, score DESC)",
        "CREATE INDEX IF NOT EXISTS idx_article_sessions_topic_status ON article_sessions (topic_slug, status, updated_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_article_blocks_session_position ON article_blocks (session_id, position)",
        "CREATE INDEX IF NOT EXISTS idx_article_drafts_session_version ON article_drafts (session_id, version DESC)",
        "CREATE INDEX IF NOT EXISTS idx_agent_events_run_step ON agent_events (run_id, step_index, id)",
        "CREATE INDEX IF NOT EXISTS idx_messages_text_fts ON messages USING GIN (to_tsvector('simple', coalesce(text, '')))",
        "CREATE INDEX IF NOT EXISTS idx_messages_posted_channel ON messages (posted_at DESC, channel_id)",
        "CREATE INDEX IF NOT EXISTS idx_agent_runs_status_locked ON agent_runs (status, locked_at)",
        "CREATE INDEX IF NOT EXISTS idx_task_runs_task_started ON task_runs (task_name, started_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_agent_eval_runs_started ON agent_eval_runs (started_at DESC)",
    ]
    for statement in statements:
        op.execute(statement)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS worker_heartbeats")
    op.execute("DROP TABLE IF EXISTS agent_eval_cases")
    op.execute("DROP TABLE IF EXISTS agent_eval_runs")
    op.execute("DROP TABLE IF EXISTS task_runs")
    op.execute("DROP TABLE IF EXISTS agent_runs")
