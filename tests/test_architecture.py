import json
from pathlib import Path

from tg_radar.agent_runtime.prompt_loader import PromptLoader
from tg_radar.config import Settings


SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "tg_radar"
CONFIG_PATH_PARTS = {"config_data", "prompts"}
REGEX_MARKERS = (
    "re.compile",
    "re.findall",
    "re.sub",
    "re.search",
    "re.fullmatch",
    "re.match",
    "re.split",
    "Field(pattern",
)


def iter_python_sources():
    for path in SRC_ROOT.rglob("*.py"):
        if any(part in CONFIG_PATH_PARTS for part in path.parts):
            continue
        yield path


def test_runtime_code_has_no_cyrillic_constants():
    offenders = []
    for path in iter_python_sources():
        text = path.read_text(encoding="utf-8")
        if any("А" <= char <= "я" or char == "ё" or char == "Ё" for char in text):
            offenders.append(path.relative_to(SRC_ROOT.parent).as_posix())

    assert offenders == []


def test_runtime_code_has_no_regex_usage():
    offenders = []
    for path in iter_python_sources():
        text = path.read_text(encoding="utf-8")
        lines = [line.strip() for line in text.splitlines()]
        has_regex_import = any(line == "import re" or line.startswith("from re ") for line in lines)
        if has_regex_import or any(marker in text for marker in REGEX_MARKERS):
            offenders.append(path.relative_to(SRC_ROOT.parent).as_posix())

    assert offenders == []


def test_default_prompts_are_external_files():
    prompt_names = {
        Settings.model_fields["agent_collection_system_prompt"].default,
        Settings.model_fields["content_system_prompt"].default,
        Settings.model_fields["content_outline_task_prompt"].default,
        Settings.model_fields["content_draft_task_prompt"].default,
        Settings.model_fields["story_match_prompt"].default,
        "airbot_chunk_analysis.md",
        "airbot_fight.md",
        "airbot_merge_analysis.md",
        "operator_quality_classification.md",
    }
    loader = PromptLoader()

    for name in prompt_names:
        assert name.endswith(".md")
        assert (SRC_ROOT / "prompts" / name).is_file()
        assert loader.load(name).strip()


def test_default_rule_configs_are_external_json_files():
    rule_names = {
        "airbot_texts.json",
        "agent_tool_rules.json",
        "cli_rules.json",
        "content_factory_rules.json",
        "discovery_rules.json",
        "parser_rules.json",
        "scoring_rules.json",
        "search_provider_rules.json",
        "story_catalog.json",
        "story_rules.json",
        "topical_rules.json",
    }

    for name in rule_names:
        path = SRC_ROOT / "config_data" / name
        assert path.is_file()
        assert json.loads(path.read_text(encoding="utf-8"))


def test_airbot_ranks_are_single_config_source():
    config = json.loads((SRC_ROOT / "config_data" / "airbot_texts.json").read_text(encoding="utf-8"))
    ranks = config["elo_ranks"]
    prompts = "\n".join(path.read_text(encoding="utf-8") for path in (SRC_ROOT / "prompts").glob("airbot_*_analysis.md"))
    allowed_emoji = {"🫧", "🌬️", "💨", "🌫️", "🌪️", "🌀"}

    assert ranks
    for rank in ranks:
        assert rank["emoji"] in allowed_emoji
        assert 1 <= int(rank["unit_value"]) <= 1000
        assert "->" in rank["tag"]
        assert rank["tag"] not in prompts
        assert rank["unit"] not in prompts


def test_agent_core_has_no_storage_dependency():
    offenders = []
    for path in (SRC_ROOT / "agent_core").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "tg_radar.db" in text or "tg_radar.agent_state" in text:
            offenders.append(path.relative_to(SRC_ROOT.parent).as_posix())

    assert offenders == []


def test_runtime_adapters_do_not_own_http_transport():
    offenders = []
    allowed = {"openai_transport.py"}
    for path in (SRC_ROOT / "agent_runtime").rglob("*.py"):
        if path.name in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        if "import httpx" in text or "from httpx" in text:
            offenders.append(path.relative_to(SRC_ROOT.parent).as_posix())

    assert offenders == []


def test_mcp_tool_adapter_has_no_runtime_or_storage_dependency():
    path = SRC_ROOT / "agent_tools" / "mcp.py"
    text = path.read_text(encoding="utf-8")

    assert "tg_radar.db" not in text
    assert "tg_radar.agent_runtime" not in text
