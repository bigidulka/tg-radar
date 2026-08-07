from __future__ import annotations

from typing import Any

from tg_radar.rule_loader import load_rules


def load_content_rules() -> dict[str, Any]:
    return load_rules("content_factory_rules.json")
