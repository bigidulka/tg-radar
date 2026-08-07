from __future__ import annotations

from importlib.resources import files
from pathlib import Path


class PromptLoader:
    def __init__(self, package: str = "tg_radar.prompts", override_dir: str | None = None) -> None:
        self.package = package
        self.override_dir = Path(override_dir) if override_dir else None

    def load(self, name: str) -> str:
        if self.override_dir:
            path = self.override_dir / name
            if path.exists():
                return path.read_text(encoding="utf-8")
        return files(self.package).joinpath(name).read_text(encoding="utf-8")
