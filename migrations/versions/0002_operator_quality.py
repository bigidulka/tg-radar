from __future__ import annotations

from alembic import op

from tg_radar.db import Base

revision = "0002_operator_quality"
down_revision = "0001_prod_hardening"
branch_labels = None
depends_on = None


def upgrade() -> None:
    Base.metadata.create_all(bind=op.get_bind())
    statements = [
        "ALTER TABLE channels ADD COLUMN IF NOT EXISTS operator_niche_category VARCHAR(32)",
        "ALTER TABLE channels ADD COLUMN IF NOT EXISTS operator_income_authenticity VARCHAR(32)",
        "ALTER TABLE channels ADD COLUMN IF NOT EXISTS operator_niche_tags VARCHAR[] DEFAULT '{}' NOT NULL",
        "ALTER TABLE channels ADD COLUMN IF NOT EXISTS operator_niche_confidence DOUBLE PRECISION DEFAULT 0 NOT NULL",
        "ALTER TABLE channels ADD COLUMN IF NOT EXISTS operator_niche_evidence JSONB DEFAULT '{}'::jsonb NOT NULL",
        "ALTER TABLE channels ADD COLUMN IF NOT EXISTS operator_niche_notes VARCHAR[] DEFAULT '{}' NOT NULL",
        "ALTER TABLE channels ADD COLUMN IF NOT EXISTS operator_niche_updated_at TIMESTAMP WITH TIME ZONE",
        "ALTER TABLE discovered_channels ADD COLUMN IF NOT EXISTS operator_niche_category VARCHAR(32)",
        "ALTER TABLE discovered_channels ADD COLUMN IF NOT EXISTS operator_niche_tags VARCHAR[] DEFAULT '{}' NOT NULL",
        "ALTER TABLE discovered_channels ADD COLUMN IF NOT EXISTS operator_niche_confidence DOUBLE PRECISION DEFAULT 0 NOT NULL",
        "CREATE INDEX IF NOT EXISTS idx_channels_operator_niche_category ON channels (operator_niche_category)",
        "CREATE INDEX IF NOT EXISTS idx_channels_operator_niche_updated_at ON channels (operator_niche_updated_at)",
        "CREATE INDEX IF NOT EXISTS idx_discovered_channels_operator_niche_category ON discovered_channels (operator_niche_category)",
    ]
    for statement in statements:
        op.execute(statement)


def downgrade() -> None:
    statements = [
        "DROP INDEX IF EXISTS idx_discovered_channels_operator_niche_category",
        "DROP INDEX IF EXISTS idx_channels_operator_niche_updated_at",
        "DROP INDEX IF EXISTS idx_channels_operator_niche_category",
        "ALTER TABLE discovered_channels DROP COLUMN IF EXISTS operator_niche_confidence",
        "ALTER TABLE discovered_channels DROP COLUMN IF EXISTS operator_niche_tags",
        "ALTER TABLE discovered_channels DROP COLUMN IF EXISTS operator_niche_category",
        "ALTER TABLE channels DROP COLUMN IF EXISTS operator_niche_updated_at",
        "ALTER TABLE channels DROP COLUMN IF EXISTS operator_niche_notes",
        "ALTER TABLE channels DROP COLUMN IF EXISTS operator_niche_evidence",
        "ALTER TABLE channels DROP COLUMN IF EXISTS operator_niche_confidence",
        "ALTER TABLE channels DROP COLUMN IF EXISTS operator_income_authenticity",
        "ALTER TABLE channels DROP COLUMN IF EXISTS operator_niche_category",
    ]
    for statement in statements:
        op.execute(statement)
