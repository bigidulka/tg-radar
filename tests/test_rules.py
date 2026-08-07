import json

from tg_radar.rule_loader import load_rules


def test_rules_can_be_overridden_from_env_file(tmp_path, monkeypatch):
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    (rules_dir / "cli_rules.json").write_text(json.dumps({"truthy_values": ["y"]}), encoding="utf-8")
    (tmp_path / ".env").write_text(f"TG_RADAR_RULES_DIR={rules_dir}\n", encoding="utf-8")

    monkeypatch.delenv("TG_RADAR_RULES_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    load_rules.cache_clear()

    assert load_rules("cli_rules.json") == {"truthy_values": ["y"]}

    load_rules.cache_clear()
