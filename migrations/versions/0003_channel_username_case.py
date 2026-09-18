from __future__ import annotations

from alembic import op

from tg_radar.db import Base

revision = "0003_channel_username_case"
down_revision = "0002_operator_quality"
branch_labels = None
depends_on = None


def upgrade() -> None:
    Base.metadata.create_all(bind=op.get_bind())
    statements = [
        "ALTER TABLE search_tasks ADD COLUMN IF NOT EXISTS topic_slug VARCHAR(128)",
        "CREATE INDEX IF NOT EXISTS idx_search_tasks_topic_slug ON search_tasks (topic_slug)",
        # Telegram handles are case insensitive, so `neuralshit` and `NeuralShit` are one
        # channel. Keep the row that already holds the most messages and fold the rest in.
        """
        CREATE TEMP TABLE channel_merge_map ON COMMIT DROP AS
        WITH ranked AS (
            SELECT c.id,
                   lower(c.username) AS handle,
                   row_number() OVER (
                       PARTITION BY lower(c.username)
                       ORDER BY (SELECT count(*) FROM messages m WHERE m.channel_id = c.id) DESC, c.id
                   ) AS position
            FROM channels c
        )
        SELECT ranked.id AS duplicate_id, keep.id AS keep_id
        FROM ranked
        JOIN ranked AS keep ON keep.handle = ranked.handle AND keep.position = 1
        WHERE ranked.position > 1
        """,
        # Same channel and same tg_msg_id is the same post: the duplicate row is dropped,
        # everything else moves over so no message is lost.
        """
        DELETE FROM messages m
        USING channel_merge_map map
        WHERE m.channel_id = map.duplicate_id
          AND EXISTS (SELECT 1 FROM messages keep WHERE keep.channel_id = map.keep_id AND keep.tg_msg_id = m.tg_msg_id)
        """,
        "UPDATE messages m SET channel_id = map.keep_id FROM channel_merge_map map WHERE m.channel_id = map.duplicate_id",
        """
        DELETE FROM edges e
        USING channel_merge_map map
        WHERE e.src_channel_id = map.duplicate_id
          AND EXISTS (
              SELECT 1 FROM edges keep
              WHERE keep.src_channel_id = map.keep_id
                AND keep.dst_username = e.dst_username
                AND keep.edge_type = e.edge_type
                AND keep.msg_id IS NOT DISTINCT FROM e.msg_id
          )
        """,
        "UPDATE edges e SET src_channel_id = map.keep_id FROM channel_merge_map map WHERE e.src_channel_id = map.duplicate_id",
        """
        DELETE FROM topic_channels t
        USING channel_merge_map map
        WHERE t.channel_id = map.duplicate_id
          AND EXISTS (SELECT 1 FROM topic_channels keep WHERE keep.topic_id = t.topic_id AND keep.channel_id = map.keep_id)
        """,
        "UPDATE topic_channels t SET channel_id = map.keep_id FROM channel_merge_map map WHERE t.channel_id = map.duplicate_id",
        """
        DELETE FROM crawl_jobs j
        USING channel_merge_map map
        WHERE j.channel_id = map.duplicate_id
          AND EXISTS (SELECT 1 FROM crawl_jobs keep WHERE keep.channel_id = map.keep_id)
        """,
        "UPDATE crawl_jobs j SET channel_id = map.keep_id FROM channel_merge_map map WHERE j.channel_id = map.duplicate_id",
        # The surviving row must not lose crawl state that only the folded rows carried.
        """
        UPDATE channels keep
        SET title = coalesce(keep.title, folded.title),
            quality_score = GREATEST(keep.quality_score, folded.quality_score),
            source_trust = GREATEST(keep.source_trust, folded.source_trust),
            first_seen_at = LEAST(keep.first_seen_at, folded.first_seen_at),
            last_seen_at = GREATEST(keep.last_seen_at, folded.last_seen_at),
            last_crawled_at = GREATEST(keep.last_crawled_at, folded.last_crawled_at),
            last_successful_crawl_at = GREATEST(keep.last_successful_crawl_at, folded.last_successful_crawl_at),
            crawl_priority = LEAST(keep.crawl_priority, folded.crawl_priority),
            crawl_interval_seconds = LEAST(keep.crawl_interval_seconds, folded.crawl_interval_seconds),
            public_chat = keep.public_chat OR folded.public_chat,
            recent_jobs = keep.recent_jobs OR folded.recent_jobs,
            company_match = keep.company_match OR folded.company_match,
            ai_llm_match = keep.ai_llm_match OR folded.ai_llm_match,
            direct_contact = keep.direct_contact OR folded.direct_contact
        FROM (
            SELECT map.keep_id,
                   max(c.title) AS title,
                   max(c.quality_score) AS quality_score,
                   max(c.source_trust) AS source_trust,
                   min(c.first_seen_at) AS first_seen_at,
                   max(c.last_seen_at) AS last_seen_at,
                   max(c.last_crawled_at) AS last_crawled_at,
                   max(c.last_successful_crawl_at) AS last_successful_crawl_at,
                   min(c.crawl_priority) AS crawl_priority,
                   min(c.crawl_interval_seconds) AS crawl_interval_seconds,
                   bool_or(c.public_chat) AS public_chat,
                   bool_or(c.recent_jobs) AS recent_jobs,
                   bool_or(c.company_match) AS company_match,
                   bool_or(c.ai_llm_match) AS ai_llm_match,
                   bool_or(c.direct_contact) AS direct_contact
            FROM channel_merge_map map
            JOIN channels c ON c.id = map.duplicate_id
            GROUP BY map.keep_id
        ) folded
        WHERE keep.id = folded.keep_id
        """,
        "DELETE FROM channels c USING channel_merge_map map WHERE c.id = map.duplicate_id",
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_channels_username_lower ON channels (lower(username))",
    ]
    for statement in statements:
        op.execute(statement)


def downgrade() -> None:
    # The channel merge is not reversible; only the schema changes are dropped.
    statements = [
        "DROP INDEX IF EXISTS uq_channels_username_lower",
        "DROP INDEX IF EXISTS idx_search_tasks_topic_slug",
        "ALTER TABLE search_tasks DROP COLUMN IF EXISTS topic_slug",
    ]
    for statement in statements:
        op.execute(statement)
