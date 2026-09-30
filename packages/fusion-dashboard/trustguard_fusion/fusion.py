"""Trust fusion: one per-device trust score (0-100) from both layers.

I built this so an operator gets a single number per device with an
explainable breakdown, instead of scattered warnings from separate tools.
Every penalty term is named, so the score can always be drilled into.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from trustguard_data.taxonomy import HealthState, Layer1Result
from trustguard_model.monitors import Layer2Result

DEFAULT_WEIGHTS = {"layer1": 0.6, "layer2": 0.4}


@dataclass
class TrustReport:
    device_id: str
    trust_score: float            # 0-100, higher = more trustworthy
    layer1_score: float           # 0-100
    layer2_score: float | None    # 0-100, None when no model is monitored
    contributions: dict[str, float]  # named penalty terms (negative numbers)
    worst_channels: list[str]     # channels with the lowest health, worst first
    timestamp_s: float = 0.0


def _check_score(name: str, value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    v = float(value)
    if v != v:  # NaN
        raise ValueError(f"{name} must not be NaN")
    if not 0.0 <= v <= 100.0:
        raise ValueError(f"{name} must be in [0, 100], got {v}")
    return v


def fuse_trust(
    device_id: str,
    l1: Layer1Result,
    l2: Layer2Result | None,
    weights: dict | None = None,
    timestamp_s: float = 0.0,
) -> TrustReport:
    """Fuse Layer 1 and Layer 2 health into one per-device trust score.

    trust = 100 + sum(contributions); every contribution is a named,
    non-positive penalty so the score is fully explainable.
    """
    if not isinstance(device_id, str) or not device_id:
        raise ValueError("device_id must be a non-empty string")
    if not isinstance(l1, Layer1Result):
        raise TypeError("l1 must be a Layer1Result")
    if l2 is not None and not isinstance(l2, Layer2Result):
        raise TypeError("l2 must be a Layer2Result or None")
    if not isinstance(timestamp_s, (int, float)) or isinstance(timestamp_s, bool):
        raise TypeError("timestamp_s must be a number")
    if timestamp_s != timestamp_s or timestamp_s < 0:
        raise ValueError("timestamp_s must be a non-negative number")

    w = dict(DEFAULT_WEIGHTS)
    if weights is not None:
        if not isinstance(weights, dict):
            raise TypeError("weights must be a dict or None")
        for k in ("layer1", "layer2"):
            if k in weights:
                v = weights[k]
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    raise TypeError(f"weights[{k!r}] must be a number")
                if v < 0:
                    raise ValueError(f"weights[{k!r}] must be non-negative")
                w[k] = float(v)
    if l2 is None:
        w = {"layer1": 1.0, "layer2": 0.0}
    else:
        total = w["layer1"] + w["layer2"]
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        w = {k: v / total for k, v in w.items()}

    l1_score = _check_score("l1.layer_score", l1.layer_score)
    l2_score = _check_score("l2.health", l2.health) if l2 is not None else None

    contributions: dict[str, float] = {}

    # Layer 1: each channel contributes a share of the layer-1 penalty
    # proportional to its unhealthiness.
    channels = l1.channels or {}
    if channels:
        per_channel_w = w["layer1"] / len(channels)
        for name, ch in channels.items():
            ch_score = _check_score(f"channel {name!r} score", ch.score)
            penalty = -(100.0 - ch_score) * per_channel_w
            state = ch.state.value if isinstance(ch.state, HealthState) else str(ch.state)
            contributions[f"layer1:{name}:{state}"] = round(penalty, 4)

    # Layer 2: penalty shared equally across monitored signals.
    if l2 is not None and l2.signals:
        per_signal_w = w["layer2"] / len(l2.signals)
        for name in l2.signals:
            penalty = -(100.0 - l2_score) * per_signal_w
            contributions[f"layer2:{name}"] = round(penalty, 4)

    trust = 100.0 + sum(contributions.values())
    trust = max(0.0, min(100.0, trust))

    ranked = sorted(channels.items(), key=lambda kv: kv[1].score)
    worst = [name for name, ch in ranked[:3] if ch.score < 100.0]

    return TrustReport(
        device_id=device_id,
        trust_score=round(trust, 2),
        layer1_score=round(l1_score, 2),
        layer2_score=round(l2_score, 2) if l2_score is not None else None,
        contributions=contributions,
        worst_channels=worst,
        timestamp_s=float(timestamp_s),
    )
