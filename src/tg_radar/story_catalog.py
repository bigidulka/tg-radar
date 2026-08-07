from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

from tg_radar.rule_loader import load_rules


CATALOG_FILE = "story_catalog.json"
RULES_FILE = "story_rules.json"
CATALOG_PATH = Path(__file__).parent / "config_data" / CATALOG_FILE


@dataclass(frozen=True)
class StoryCase:
    number: int
    title: str
    format: str
    section: str
    body: str
    takeaway: str | None = None


def load_story_rules() -> dict:
    return load_rules(RULES_FILE)


def parse_catalog(markdown: str) -> list[StoryCase]:
    rules = load_story_rules()["catalog"]
    takeaway_prefix = str(rules["takeaway_prefix"])
    separator = str(rules["heading_separator"])
    skip_prefixes = tuple(rules["skip_prefixes"])
    cases: list[StoryCase] = []
    section = ""
    heading: dict | None = None
    body: list[str] = []
    takeaway: str | None = None

    def flush() -> None:
        nonlocal heading, body, takeaway
        if heading:
            cases.append(
                StoryCase(
                    number=int(heading["number"]),
                    title=str(heading["title"]),
                    format=str(heading["format"]),
                    section=section,
                    body=" ".join(body).strip(),
                    takeaway=takeaway,
                )
            )
        heading, body, takeaway = None, [], None

    for raw in markdown.splitlines():
        line = raw.strip()
        case_heading = _parse_heading(line, separator)
        if case_heading:
            flush()
            heading = case_heading
            continue
        if line.startswith("#"):
            flush()
            section = line.lstrip("#").strip()
            continue
        if heading is None:
            continue
        if line.startswith(takeaway_prefix):
            takeaway = line[len(takeaway_prefix) :].strip()
            continue
        if not line or line.startswith(skip_prefixes):
            continue
        body.append(line)
    flush()
    return cases


def _parse_heading(line: str, separator: str) -> dict | None:
    if not line.startswith("**"):
        return None
    end = line.find("**", 2)
    if end < 0:
        return None
    number, dot, title = line[2:end].partition(". ")
    if not dot or not number.isdigit() or not title.strip():
        return None
    rest = line[end + 2 :].strip()
    if not rest.startswith(separator):
        return None
    format_name = rest[len(separator) :].strip()
    if len(format_name) < 3 or not format_name.startswith("*") or not format_name.endswith("*"):
        return None
    return {"number": int(number), "title": title.strip(), "format": format_name.strip("*").strip()}


def catalog_payload(cases: list[StoryCase], source: str) -> dict:
    return {
        "source": source,
        "case_count": len(cases),
        "cases": [asdict(case) for case in cases],
    }


def write_catalog(cases: list[StoryCase], source: str, path: Path = CATALOG_PATH) -> Path:
    path.write_text(
        json.dumps(catalog_payload(cases, source), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def regenerate_catalog(source_path: str, path: Path = CATALOG_PATH) -> list[StoryCase]:
    source = Path(source_path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"story catalog source not found: {source}")
    cases = parse_catalog(source.read_text(encoding="utf-8"))
    if not cases:
        raise ValueError(f"no cases parsed from {source}")
    write_catalog(cases, str(source), path)
    return cases


def load_story_catalog() -> list[StoryCase]:
    payload = load_rules(CATALOG_FILE)
    return [
        StoryCase(
            number=int(item["number"]),
            title=str(item["title"]),
            format=str(item.get("format") or ""),
            section=str(item.get("section") or ""),
            body=str(item.get("body") or ""),
            takeaway=item.get("takeaway"),
        )
        for item in payload.get("cases", [])
    ]
