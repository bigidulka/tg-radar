from __future__ import annotations

import json
import os
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any


@lru_cache
def load_rules(name: str) -> dict[str, Any]:
    override_dir = os.environ.get("TG_RADAR_RULES_DIR") or _env_file_value("TG_RADAR_RULES_DIR")
    if override_dir:
        path = Path(override_dir) / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    text = files("tg_radar.config_data").joinpath(name).read_text(encoding="utf-8")
    return json.loads(text)


def _env_file_value(key: str, env_path: str = ".env") -> str | None:
    path = Path(env_path)
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        if name.strip() == key:
            return value.strip().strip("\"'")
    return None
