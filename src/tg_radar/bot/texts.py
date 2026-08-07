from __future__ import annotations

from typing import Any

from tg_radar.rule_loader import load_rules


def bot_texts() -> dict[str, Any]:
    load_rules.cache_clear()
    return load_rules("airbot_texts.json")


def text(name: str, **values: object) -> str:
    value = bot_texts()
    for part in name.split("."):
        next_value = value.get(part) if isinstance(value, dict) else None
        if next_value is None:
            return name
        value = next_value
    if not isinstance(value, str):
        return name
    return value.format(**values)
