"""Health taxonomy: states, per-channel health, and the classification rule."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np


class HealthState(enum.Enum):
    """Channel health states for Layer 1."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DRIFTING = "drifting"
    STUCK = "stuck"
    DROPOUT = "dropout"
    MISSING = "missing"


@dataclass
class ChannelHealth:
    """Health verdict for one channel in one window."""

    channel: str
    state: HealthState
    score: float  # 0-100, higher = healthier
    details: dict = field(default_factory=dict)  # detector outputs behind the decision


@dataclass
class Layer1Result:
    """Layer 1 verdict for one window across all its channels."""

    device_id: str
    start_s: float
    end_s: float
    channels: dict[str, ChannelHealth]
    layer_score: float  # 0-100, mean of channel scores


#: Default decision thresholds for :func:`classify_channel`. Override per key
#: via the ``thresholds`` argument or the pipeline config.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "dropout_missing_fraction": 0.20,  # >= -> DROPOUT
    "dropout_longest_run_s": 10.0,  # >= -> DROPOUT
    "range_out_fraction": 0.05,  # >= -> DEGRADED
    "ks_pvalue": 0.01,  # < -> DRIFTING (vs baseline), gated by ks_min_d
    "ks_min_d": 0.0,  # KS statistic must also reach this (effect-size gate;
    # guards the p-value against huge-N false positives; 0.0 = gate off)
    "psi_drift": 0.25,  # >= -> DRIFTING
    "psi_watch": 0.10,  # >= (and < psi_drift) -> DEGRADED
    "trend_pvalue": 0.01,  # < (with meaningful swing) -> DRIFTING
    "trend_min_relative": 1.0,  # |slope|*window_s must exceed this * std
    "stuck_min_duration_s": 5.0,  # flat run must last this long -> STUCK
    "acc_min_correlation": 0.5,  # used by the pipeline for acc pairs
}


def _is_drifting(ks: dict | None, psiv: float | None, trend: dict,
                 window_s: float, std: float, thr: dict) -> bool:
    # KS p-values collapse to ~0 at large N even for negligible shifts, so the
    # p-value rule is gated on the effect size (KS D statistic). This is the
    # standard guard against huge-N false positives in drift detection.
    if ks is not None and float(ks.get("pvalue", 1.0)) < thr["ks_pvalue"] \
            and float(ks.get("statistic", 0.0)) >= thr["ks_min_d"]:
        return True
    if (
        isinstance(psiv, (int, float))
        and np.isfinite(psiv)
        and psiv >= thr["psi_drift"]
    ):
        return True
    if trend:
        p = float(trend.get("pvalue", 1.0))
        swing = abs(float(trend.get("slope_per_s", 0.0))) * window_s
        if p < thr["trend_pvalue"] and window_s > 0 and std > 0 \
                and swing > thr["trend_min_relative"] * std:
            return True
    return False


def _drift_reason(ks: dict | None, psiv: float | None, trend: dict, thr: dict) -> str:
    parts = []
    if ks is not None and float(ks.get("pvalue", 1.0)) < thr["ks_pvalue"]:
        parts.append(f"ks pvalue={ks['pvalue']:.2e}")
    if isinstance(psiv, (int, float)) and np.isfinite(psiv) and psiv >= thr["psi_drift"]:
        parts.append(f"psi={psiv:.3f}")
    if trend and float(trend.get("pvalue", 1.0)) < thr["trend_pvalue"]:
        parts.append(f"trend slope={trend.get('slope_per_s', 0.0):.4g}/s")
    return "drift: " + ", ".join(parts) if parts else "drift"


def _is_degraded(outputs: dict, rng: dict, psiv: float | None, thr: dict) -> bool:
    if rng and float(rng.get("fraction_out_of_range", 0.0) or 0.0) >= thr["range_out_fraction"]:
        return True
    if (
        isinstance(psiv, (int, float))
        and np.isfinite(psiv)
        and thr["psi_watch"] <= psiv < thr["psi_drift"]
    ):
        return True
    if outputs.get("schema_ok") is False:
        return True
    if outputs.get("consistency_issue"):
        return True
    return False


def _degraded_reason(outputs: dict, rng: dict, psiv: float | None, thr: dict) -> str:
    parts = []
    if rng and float(rng.get("fraction_out_of_range", 0.0) or 0.0) >= thr["range_out_fraction"]:
        parts.append(
            f"range violations frac={rng['fraction_out_of_range']:.3f}"
        )
    if (
        isinstance(psiv, (int, float))
        and np.isfinite(psiv)
        and thr["psi_watch"] <= psiv < thr["psi_drift"]
    ):
        parts.append(f"psi watch band={psiv:.3f}")
    if outputs.get("schema_ok") is False:
        parts.append("schema check failed")
    if outputs.get("consistency_issue"):
        parts.append("cross-channel consistency issue")
    return "degraded: " + ", ".join(parts) if parts else "degraded"


def classify_channel(
    name: str,
    detector_outputs: dict,
    thresholds: dict | None = None,
) -> ChannelHealth:
    """Map detector outputs to a :class:`HealthState`.

    Decision logic — first match wins, in this order:

    1. **MISSING** — ``missing`` is True (channel absent or all-NaN).
    2. **DROPOUT** — ``dropout`` shows ``missing_fraction`` >= threshold or the
       longest NaN run >= threshold seconds.
    3. **STUCK** — ``stuck["is_stuck"]`` is True.
    4. **DRIFTING** — distribution drift vs baseline: KS ``pvalue`` below
       threshold *with the KS statistic above the ``ks_min_d`` effect-size
       gate* (so large-N p-values alone can't false-positive), or PSI >=
       drift threshold, or a statistically significant
       trend whose total swing over the window exceeds
       ``trend_min_relative`` × channel std (default 1.0: the fitted line must
       move the signal by at least one standard deviation, so tiny-but-
       "significant" slopes on long noisy windows don't false-positive).
    5. **DEGRADED** — range violations above threshold, PSI in the watch band,
       a failed schema check, or a cross-channel ``consistency_issue``.
    6. **HEALTHY** — none of the above.

    Scores start at 100 with documented penalties: DROPOUT loses
    ``50 + 50*missing_fraction``; STUCK loses ``60 + 20*stuck_fraction``;
    DRIFTING loses ``30 + 20*min(1, psi)``; DEGRADED scores 80; MISSING
    scores 0; HEALTHY scores 100. Scores are clamped to [0, 100].

    Expected ``detector_outputs`` keys (as produced by
    :class:`~trustguard_data.pipeline.DataTrustPipeline`): ``missing``,
    ``schema_ok``, ``dropout``, ``range``, ``stuck``, ``ks``, ``psi``,
    ``trend``, ``window_s``, ``std``, ``consistency_issue``. Missing keys are
    treated as "no evidence".

    Raises:
        TypeError/ValueError: On bad input.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string.")
    if not isinstance(detector_outputs, dict):
        raise TypeError(
            f"detector_outputs must be a dict, got {type(detector_outputs).__name__}."
        )
    thr = dict(DEFAULT_THRESHOLDS)
    if thresholds is not None:
        if not isinstance(thresholds, dict):
            raise TypeError("thresholds must be a dict or None.")
        thr.update(thresholds)

    o = detector_outputs
    if o.get("missing"):
        state, score = HealthState.MISSING, 0.0
        reason = "channel absent or all-NaN"
    else:
        dropout = o.get("dropout") or {}
        miss_frac = float(dropout.get("missing_fraction", 0.0) or 0.0)
        longest_s = float(dropout.get("longest_nan_run_s", 0.0) or 0.0)
        stuck = o.get("stuck") or {}
        ks = o.get("ks")
        psiv = o.get("psi")
        trend = o.get("trend") or {}
        rng = o.get("range") or {}
        window_s = float(o.get("window_s", 0.0) or 0.0)
        std = float(o.get("std", 0.0) or 0.0)

        if (
            miss_frac >= thr["dropout_missing_fraction"]
            or longest_s >= thr["dropout_longest_run_s"]
        ):
            state = HealthState.DROPOUT
            score = max(0.0, 100.0 - 50.0 - 50.0 * min(1.0, miss_frac))
            reason = (
                f"dropout: missing_fraction={miss_frac:.3f}, "
                f"longest_nan_run_s={longest_s:.2f}"
            )
        elif stuck.get("is_stuck"):
            state = HealthState.STUCK
            sf = float(stuck.get("stuck_fraction", 0.0) or 0.0)
            score = max(0.0, 100.0 - 60.0 - 20.0 * min(1.0, sf))
            reason = f"stuck: longest_stuck_s={stuck.get('longest_stuck_s', 0.0):.2f}"
        elif _is_drifting(ks, psiv, trend, window_s, std, thr):
            state = HealthState.DRIFTING
            p = (
                min(1.0, psiv)
                if isinstance(psiv, (int, float)) and np.isfinite(psiv)
                else 0.0
            )
            score = max(0.0, 100.0 - 30.0 - 20.0 * p)
            reason = _drift_reason(ks, psiv, trend, thr)
        elif _is_degraded(o, rng, psiv, thr):
            state = HealthState.DEGRADED
            score = 80.0
            reason = _degraded_reason(o, rng, psiv, thr)
        else:
            state = HealthState.HEALTHY
            score = 100.0
            reason = "all detectors nominal"

    details = {"decision_reason": reason}
    details.update(o)
    return ChannelHealth(
        channel=name, state=state, score=float(score), details=details
    )
