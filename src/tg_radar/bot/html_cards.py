from __future__ import annotations

import base64
from html import escape
import json
from importlib.resources import files
from typing import Any

from tg_radar.bot.ratings import elo_rank, elo_visual_score


CARD_WIDTH = 1122
CARD_HEIGHT = 1402


def render_airbot_card_html(result: dict[str, Any], avatar_url: str | None = None) -> str:
    profile = result.get("meme_profile") if isinstance(result.get("meme_profile"), dict) else {}
    channel = str(result.get("channel") or "")
    elo = _int(result.get("air_elo"), 0)
    rank = elo_rank(elo) if elo else {}
    rank_name = str(rank.get("tag") or profile.get("rank_name") or result.get("humorous_label") or "")
    measure = _rank_measure(result, profile, rank)
    template = _asset_text("card_template.html")
    css = _asset_text("card.css")
    return template.format(
        title=escape(f"airbot-{channel}"),
        css=css,
        backplate=_backplate_data_uri(elo),
        meter=min(max(elo_visual_score(elo), 0), 200) / 2,
        channel=escape(channel),
        avatar=_avatar_html(channel, avatar_url),
        elo_label=escape(str(result.get("air_elo_label") or f"{elo} ELO")),
        rank=escape(rank_name),
        measure=escape(measure),
        custom_tag=escape(str(result.get("custom_tag") or profile.get("custom_tag") or "")),
        one_liner=escape(str(profile.get("one_liner") or result.get("summary") or "")),
        sections=_sections(result),
    )


def _sections(result: dict[str, Any]) -> str:
    niche = _chips_section("niche", result.get("niche_tags"))
    traits = _chips_section("traits", result.get("core_traits"))
    parts = [
        _chips_pair_section(niche, traits),
        _fun_metrics_section(result.get("fun_metrics")),
        _metrics_section(result.get("metrics")),
        _text_section("summary", result.get("summary")),
        _objects_section("tactics", result.get("tactics"), ("name", "hint", "why_it_keeps_attention", "code")),
        _objects_section("evidence", result.get("evidence"), ("snippet", "comment", "url", "metric")),
        _objects_section("counter_signals", result.get("counter_signals"), ("name", "comment")),
        _objects_section("text_generation_hooks", [result.get("text_generation_hooks")], ("roast_angles", "running_jokes", "safe_phrases")),
        _json_section(result),
    ]
    return "".join(part for part in parts if part)


def _chips_pair_section(left: str, right: str) -> str:
    if not left and not right:
        return ""
    if not left:
        return right
    if not right:
        return left
    return f'<div class="chips-pair">{left}{right}</div>'


def _chips_section(name: str, value: Any) -> str:
    items = value if isinstance(value, list) else []
    if not items:
        return ""
    chips = "".join(f'<span class="chip">{escape(str(item))}</span>' for item in items if item is not None)
    return f'<section class="section"><div class="section-title">{escape(name)}</div><div class="chips">{chips}</div></section>'


def _fun_metrics_section(value: Any) -> str:
    items = value if isinstance(value, list) else []
    if not items:
        return ""
    cells = []
    for raw in items:
        item = raw if isinstance(raw, dict) else {}
        cells.append(_kv(str(item.get("label") or item.get("key") or ""), str(item.get("display") or item.get("value") or "")))
    return f'<section class="section"><div class="section-title">fun_metrics</div><div class="grid">{''.join(cells)}</div></section>'


def _metrics_section(value: Any) -> str:
    metrics = value if isinstance(value, dict) else {}
    if not metrics:
        return ""
    rows = []
    for key, raw in sorted(metrics.items(), key=lambda item: _metric_score(item[1]), reverse=True):
        item = raw if isinstance(raw, dict) else {}
        score = _int(item.get("score"), 0)
        rows.append(
            '<div class="kv">'
            f'<div class="kv-label">{escape(str(key))}</div>'
            f'<div class="kv-value">{score}/100</div>'
            "</div>"
        )
    return f'<section class="section"><div class="section-title">metrics</div><div class="grid metric-grid">{''.join(rows)}</div></section>'


def _text_section(name: str, value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    return f'<section class="section"><div class="section-title">{escape(name)}</div><div class="summary">{escape(text)}</div></section>'


def _objects_section(name: str, value: Any, keys: tuple[str, ...]) -> str:
    items = value if isinstance(value, list) else []
    if not items:
        return ""
    rows = []
    for raw in items:
        item = raw if isinstance(raw, dict) else {}
        values = []
        for key in keys:
            current = item.get(key)
            if isinstance(current, list):
                current = ", ".join(str(part) for part in current)
            if current:
                values.append(f"{key}: {current}")
        if values:
            rows.append(f'<li class="list-item">{escape(" | ".join(values))}</li>')
    if not rows:
        return ""
    return f'<section class="section"><div class="section-title">{escape(name)}</div><ul class="list">{''.join(rows)}</ul></section>'


def _json_section(result: dict[str, Any]) -> str:
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    return f'<section class="section"><div class="section-title">raw_json</div><pre class="json-dump">{escape(payload)}</pre></section>'


def _kv(label: str, value: str) -> str:
    return f'<div class="kv"><div class="kv-label">{escape(label)}</div><div class="kv-value">{escape(value)}</div></div>'


def _avatar_html(channel: str, avatar_url: str | None) -> str:
    if avatar_url:
        return f'<img class="avatar" src="{escape(avatar_url, quote=True)}" alt="">'
    initials = "".join(char for char in channel if char.isalnum())[:2].upper() or "@"
    return f'<div class="avatar avatar-fallback">{escape(initials)}</div>'


def _rank_measure(result: dict[str, Any], profile: dict[str, Any], rank: dict[str, Any]) -> str:
    value = result.get("air_rank_measure") or profile.get("rank_measure")
    if value:
        return str(value)
    unit_value = rank.get("unit_value")
    unit = str(rank.get("unit") or "")
    if unit_value is None or not unit:
        return ""
    return f"{unit_value} {unit}"


def _metric_score(value: Any) -> int:
    item = value if isinstance(value, dict) else {}
    return _int(item.get("score"), 0)


def _backplate_data_uri(elo: int) -> str:
    name = _backplate_name(elo)
    data = files("tg_radar.assets.airbot_backplates").joinpath(name).read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _backplate_name(elo: int) -> str:
    manifest = json.loads(files("tg_radar.assets.airbot_backplates").joinpath("manifest.json").read_text(encoding="utf-8"))
    items = [item for item in manifest if isinstance(item, dict) and item.get("file") and item.get("elo") is not None]
    if not items:
        raise ValueError("missing backplate manifest")
    selected = min(items, key=lambda item: abs(_int(item.get("elo"), 0) - elo))
    return str(selected["file"])


def _asset_text(name: str) -> str:
    return files("tg_radar.assets.airbot_card").joinpath(name).read_text(encoding="utf-8")


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default
