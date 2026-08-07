from __future__ import annotations

from typing import Any

from tg_radar.bot.texts import bot_texts


def air_elo(metrics: dict[str, dict[str, Any]]) -> int:
    model = _elo_model()
    min_elo = _int(model.get("min_elo"))
    max_elo = _int(model.get("max_elo"))
    if max_elo <= min_elo:
        return min_elo
    floor = _float(model.get("trait_floor"))
    ceiling = _float(model.get("trait_ceiling"))
    if ceiling <= floor:
        return min_elo
    weights = model.get("weights") if isinstance(model.get("weights"), dict) else {}
    total_weight = 0.0
    weighted_total = 0.0
    for name, weight_value in weights.items():
        weight = _float(weight_value)
        if weight <= 0:
            continue
        item = metrics.get(str(name)) if isinstance(metrics.get(str(name)), dict) else {}
        score = _clamp(_float(item.get("score")), 0.0, 100.0)
        weighted_total += score * weight
        total_weight += weight
    if total_weight <= 0:
        return min_elo
    average = weighted_total / total_weight
    pressure = _clamp((average - floor) / (ceiling - floor), 0.0, 1.0)
    curve = _float(model.get("curve")) or 1.0
    base = min_elo + round((max_elo - min_elo) * (pressure**curve))
    bonus = _rating_bonus(metrics, model, "critical_threshold", "critical_bonus_per_trait", "critical_bonus_cap")
    bonus += _rating_bonus(metrics, model, "stack_threshold", "stack_bonus_per_trait", "stack_bonus_cap")
    bonus += _combo_bonus(metrics, model)
    return _clamp_int(base + bonus, min_elo, max_elo)


def elo_visual_score(elo: int) -> int:
    model = _elo_model()
    min_elo = _int(model.get("min_elo"))
    max_elo = _int(model.get("max_elo"))
    visual_max = _int(model.get("visual_score_max"))
    if max_elo <= min_elo or visual_max <= 0:
        return 0
    pressure = _clamp((float(elo) - min_elo) / (max_elo - min_elo), 0.0, 1.0)
    return round(pressure * visual_max)


def rank_bucket(elo: int) -> int:
    model = _elo_model()
    min_elo = _int(model.get("min_elo"))
    max_elo = _int(model.get("max_elo"))
    step = _int(model.get("bucket_step"))
    if step <= 0:
        return _clamp_int(elo, min_elo, max_elo)
    clipped = _clamp_int(elo, min_elo, max_elo)
    bucket = ((clipped + step // 2) // step) * step
    return _clamp_int(bucket, min_elo, max_elo)


def elo_rank(elo: int) -> dict[str, Any]:
    ranks = bot_texts().get("elo_ranks")
    if not isinstance(ranks, list):
        return {}
    bucket = rank_bucket(elo)
    fallback: dict[str, Any] = {}
    for item in ranks:
        if not isinstance(item, dict):
            continue
        if int(item.get("elo") or 0) == bucket:
            return item
        if not fallback:
            fallback = item
    return fallback


def rank_emoji(elo: int) -> str:
    return str(elo_rank(elo).get("emoji") or "")


def rank_name(elo: int) -> str:
    return str(elo_rank(elo).get("tag") or "")


def rank_summary(elo: int) -> str:
    return str(elo_rank(elo).get("summary") or "")


def _rating_bonus(metrics: dict[str, dict[str, Any]], model: dict[str, Any], threshold_key: str, step_key: str, cap_key: str) -> int:
    threshold = _float(model.get(threshold_key))
    per_trait = _int(model.get(step_key))
    cap = _int(model.get(cap_key))
    if per_trait <= 0 or cap <= 0:
        return 0
    count = 0
    for item in metrics.values():
        raw = item if isinstance(item, dict) else {}
        if _float(raw.get("score")) >= threshold:
            count += 1
    return min(cap, count * per_trait)


def _combo_bonus(metrics: dict[str, dict[str, Any]], model: dict[str, Any]) -> int:
    combos = model.get("combo_bonuses")
    if not isinstance(combos, list):
        return 0
    total = 0
    for raw_combo in combos:
        combo = raw_combo if isinstance(raw_combo, dict) else {}
        names = combo.get("metrics")
        if not isinstance(names, list):
            continue
        scores = [_metric_score(metrics, str(name)) for name in names]
        if not scores:
            continue
        threshold = _float(combo.get("threshold"))
        if min(scores) < threshold:
            continue
        multiplier = _float(combo.get("multiplier"))
        floor = _int(combo.get("floor"))
        cap = _int(combo.get("cap"))
        value = max(0, floor + round(((sum(scores) / len(scores)) - threshold) * multiplier))
        total += min(value, cap) if cap > 0 else value
    global_cap = _int(model.get("combo_bonus_cap"))
    return min(total, global_cap) if global_cap > 0 else total


def _metric_score(metrics: dict[str, dict[str, Any]], name: str) -> float:
    item = metrics.get(name) if isinstance(metrics.get(name), dict) else {}
    return _clamp(_float(item.get("score")), 0.0, 100.0)


def _elo_model() -> dict[str, Any]:
    model = bot_texts().get("elo_model")
    return model if isinstance(model, dict) else {}


def _int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _clamp_int(value: int, low: int, high: int) -> int:
    return min(max(value, low), high)
