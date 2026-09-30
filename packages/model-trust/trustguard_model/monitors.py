"""Deployed-model degradation monitoring (Layer 2).

:class:`ModelTrustMonitor` watches the *output* distribution of a deployed
binary/multiclass classifier without needing ground-truth labels for most
signals:

  * ``psi_proba`` — Population Stability Index of the per-window model score
    vs the reference score distribution given at construction (breach: above).
  * ``ks_proba`` — two-sample Kolmogorov–Smirnov statistic on the same
    scores (breach: above).
  * ``mean_confidence_drop`` — reference mean confidence minus current mean
    confidence (breach: above; a positive drop means the model got less
    sure).
  * ``ece_drift`` — only when ``y_true`` is supplied: ``|ECE_now - ECE_ref|``.
    The reference ECE is bootstrapped from the first labeled update unless
    ``config["ece_ref"]`` provides it (breach: above).
  * ``rolling_accuracy`` — only when ``y_true`` is supplied (binary models):
    mean batch accuracy over the last ``accuracy_window`` labeled updates
    (breach: below).

The per-window "score" is the canonical 1D summary of ``proba``: the raw
probability for 1D (binary P(class 1)) input, or the max class probability
for 2D input.  PSI/KS compare score distributions; confidence is
``max(p, 1-p)`` for 1D input or the max class probability for 2D input.

Every signal is guarded by a :class:`DegradationAlert` (threshold +
hysteresis): once breached, a signal must record ``hysteresis_streak`` (K,
default 3) consecutive clear updates before it un-breaches, so noisy
batches cannot flap the ``degraded`` flag.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import ks_2samp

from .alerting import DEFAULT_CLEAR_STREAK, DegradationAlert
from .baseline import expected_calibration_error

__all__ = [
    "ModelSignal",
    "Layer2Result",
    "ModelTrustMonitor",
    "DEFAULT_CONFIG",
]


@dataclass
class ModelSignal:
    """One monitored quantity with its threshold and breach state."""

    name: str
    value: float
    threshold: float
    breached: bool
    details: dict = field(default_factory=dict)


@dataclass
class Layer2Result:
    """Result of one :meth:`ModelTrustMonitor.update` call."""

    model_id: str
    n_windows: int  # cumulative windows seen by the monitor
    signals: dict[str, ModelSignal]
    degraded: bool  # True if any signal is breached (hysteresis applied)
    health: float  # 0-100; 100 = no breached signals


#: Documented defaults; any other key in ``config`` raises ValueError.
DEFAULT_CONFIG: dict = {
    "psi_threshold": 0.25,  # PSI > 0.25 is the conventional "significant shift"
    "psi_bins": 10,
    "ks_threshold": 0.20,
    "conf_drop_threshold": 0.10,  # absolute drop in mean confidence
    "ece_drift_threshold": 0.05,
    "ece_ref": None,  # or a float: skip bootstrapping from first labeled batch
    "accuracy_threshold": 0.80,
    "accuracy_window": 5,  # rolling mean over this many labeled updates
    "hysteresis_streak": DEFAULT_CLEAR_STREAK,  # K consecutive clears to heal
    "hysteresis_margin": 0.0,
    "signal_weights": None,  # or dict name -> weight for the health score
}

_KNOWN_SIGNALS = (
    "psi_proba",
    "ks_proba",
    "mean_confidence_drop",
    "ece_drift",
    "rolling_accuracy",
)


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index with reference-quantile bins (deterministic)."""
    expected = np.asarray(expected, dtype=float).ravel()
    actual = np.asarray(actual, dtype=float).ravel()
    if expected.size == 0 or actual.size == 0:
        raise ValueError("_psi: expected and actual must be non-empty")
    if not isinstance(bins, int) or isinstance(bins, bool) or bins < 2:
        raise ValueError("_psi: bins must be an int >= 2")
    edges = np.unique(np.percentile(expected, np.linspace(0, 100, bins + 1)))
    if edges.size < 2:
        return 0.0  # constant reference distribution: no measurable shift
    edges = np.concatenate(([-np.inf], edges[1:-1], [np.inf]))
    n_bins = edges.size - 1
    eps = 1e-4
    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)
    e = (e_counts + eps) / (e_counts.sum() + eps * n_bins)
    a = (a_counts + eps) / (a_counts.sum() + eps * n_bins)
    return float(np.sum((a - e) * np.log(a / e)))


def _canonical_score(proba: np.ndarray) -> np.ndarray:
    """1D per-window score: p for 1D input, max class proba for 2D input."""
    if proba.ndim == 1:
        return proba
    return proba.max(axis=1)


def _confidence(proba: np.ndarray, confidences: np.ndarray | None) -> np.ndarray:
    """Per-window confidence: explicit confidences, else max(p, 1-p) / max class."""
    if confidences is not None:
        return confidences
    if proba.ndim == 1:
        return np.maximum(proba, 1.0 - proba)
    return proba.max(axis=1)


class ModelTrustMonitor:
    """Stateful degradation monitor for one deployed model.

    Parameters
    ----------
    model_id:
        Non-empty string identifying the deployed model.
    reference_proba:
        Predicted probabilities from a trusted reference period: 1D array of
        P(class 1) for binary models, or 2D ``(n_windows, n_classes)`` with
        rows summing to 1.  Values must lie in [0, 1].
    config:
        Optional dict overriding :data:`DEFAULT_CONFIG`.  Unknown keys raise
        ``ValueError``.
    """

    def __init__(
        self, model_id: str, reference_proba: np.ndarray, config: dict | None = None
    ) -> None:
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("model_id must be a non-empty string")
        self.model_id = model_id
        self.config = self._validate_config(config)
        self._ref_proba = self._validate_proba(
            reference_proba, "reference_proba", allow_ndim=None
        )
        self._ref_ndim = self._ref_proba.ndim
        self._ref_score = _canonical_score(self._ref_proba)
        self._ref_mean_conf = float(_confidence(self._ref_proba, None).mean())

        k = self.config["hysteresis_streak"]
        margin = self.config["hysteresis_margin"]
        self._alerts: dict[str, DegradationAlert] = {
            "psi_proba": DegradationAlert(
                "psi_proba", self.config["psi_threshold"], "above", k, margin
            ),
            "ks_proba": DegradationAlert(
                "ks_proba", self.config["ks_threshold"], "above", k, margin
            ),
            "mean_confidence_drop": DegradationAlert(
                "mean_confidence_drop",
                self.config["conf_drop_threshold"],
                "above",
                k,
                margin,
            ),
            "ece_drift": DegradationAlert(
                "ece_drift", self.config["ece_drift_threshold"], "above", k, margin
            ),
            "rolling_accuracy": DegradationAlert(
                "rolling_accuracy",
                self.config["accuracy_threshold"],
                "below",
                k,
                margin,
            ),
        }
        self._ref_ece: float | None = self.config["ece_ref"]
        self._acc_history: deque[float] = deque(
            maxlen=self.config["accuracy_window"]
        )
        self._n_windows = 0

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _validate_config(config: dict | None) -> dict:
        merged = dict(DEFAULT_CONFIG)
        if config is None:
            return merged
        if not isinstance(config, dict):
            raise TypeError("config must be a dict or None")
        unknown = set(config) - set(DEFAULT_CONFIG)
        if unknown:
            raise ValueError(
                f"unknown config keys: {sorted(unknown)}; "
                f"allowed: {sorted(DEFAULT_CONFIG)}"
            )
        merged.update(config)
        if merged["ece_ref"] is not None:
            ece_ref = merged["ece_ref"]
            if (
                not isinstance(ece_ref, (int, float))
                or isinstance(ece_ref, bool)
                or not 0.0 <= float(ece_ref) <= 1.0
            ):
                raise ValueError("config['ece_ref'] must be None or a float in [0, 1]")
            merged["ece_ref"] = float(ece_ref)
        for key in (
            "psi_threshold",
            "ks_threshold",
            "conf_drop_threshold",
            "ece_drift_threshold",
        ):
            v = merged[key]
            if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0:
                raise ValueError(f"config['{key}'] must be a non-negative number")
            merged[key] = float(v)
        if (
            not isinstance(merged["psi_bins"], int)
            or isinstance(merged["psi_bins"], bool)
            or merged["psi_bins"] < 2
        ):
            raise ValueError("config['psi_bins'] must be an int >= 2")
        if not 0.0 <= float(merged["accuracy_threshold"]) <= 1.0:
            raise ValueError("config['accuracy_threshold'] must lie in [0, 1]")
        merged["accuracy_threshold"] = float(merged["accuracy_threshold"])
        if (
            not isinstance(merged["accuracy_window"], int)
            or isinstance(merged["accuracy_window"], bool)
            or merged["accuracy_window"] < 1
        ):
            raise ValueError("config['accuracy_window'] must be a positive int")
        if (
            not isinstance(merged["hysteresis_streak"], int)
            or isinstance(merged["hysteresis_streak"], bool)
            or merged["hysteresis_streak"] < 1
        ):
            raise ValueError("config['hysteresis_streak'] (K) must be a positive int")
        if merged["hysteresis_margin"] < 0:
            raise ValueError("config['hysteresis_margin'] must be non-negative")
        merged["hysteresis_margin"] = float(merged["hysteresis_margin"])
        sw = merged["signal_weights"]
        if sw is not None:
            if not isinstance(sw, dict) or not sw:
                raise ValueError("config['signal_weights'] must be None or a non-empty dict")
            unknown_signals = set(sw) - set(_KNOWN_SIGNALS)
            if unknown_signals:
                raise ValueError(
                    f"config['signal_weights'] has unknown signals: "
                    f"{sorted(unknown_signals)}"
                )
            for name, w in sw.items():
                if (
                    not isinstance(w, (int, float))
                    or isinstance(w, bool)
                    or w < 0
                ):
                    raise ValueError(
                        f"config['signal_weights']['{name}'] must be a non-negative number"
                    )
        return merged

    @staticmethod
    def _validate_proba(proba, where: str, allow_ndim: int | None) -> np.ndarray:
        arr = np.asarray(proba, dtype=float)
        if arr.ndim not in (1, 2):
            raise ValueError(
                f"{where}: proba must be 1D (binary P(class 1)) or 2D "
                f"(n_windows, n_classes); got ndim={arr.ndim}"
            )
        if arr.shape[0] == 0:
            raise ValueError(f"{where}: proba must contain at least one window")
        if allow_ndim is not None and arr.ndim != allow_ndim:
            raise ValueError(
                f"{where}: proba ndim ({arr.ndim}) does not match the reference "
                f"proba ndim ({allow_ndim})"
            )
        if np.any(~np.isfinite(arr)):
            raise ValueError(f"{where}: proba must be finite")
        if np.any((arr < 0.0) | (arr > 1.0)):
            raise ValueError(f"{where}: proba values must lie in [0, 1]")
        if arr.ndim == 2:
            if arr.shape[1] < 2:
                raise ValueError(
                    f"{where}: 2D proba needs >= 2 classes, got {arr.shape[1]}"
                )
            if not np.allclose(arr.sum(axis=1), 1.0, atol=1e-6):
                raise ValueError(f"{where}: 2D proba rows must sum to 1")
        return arr

    # ----------------------------------------------------------------- update
    def update(
        self,
        proba: np.ndarray,
        y_true: np.ndarray | None = None,
        confidences: np.ndarray | None = None,
    ) -> Layer2Result:
        """Score one batch of deployed predictions.

        Parameters
        ----------
        proba:
            1D (binary) or 2D (n_windows, n_classes) predicted probabilities,
            same ndim as the reference proba.
        y_true:
            Optional ground-truth labels (binary int labels when the ECE /
            accuracy signals are used).  Enables ``ece_drift`` and
            ``rolling_accuracy``.
        confidences:
            Optional per-window confidences in [0, 1]; defaults to
            ``max(p, 1-p)`` (1D) or the max class probability (2D).

        Returns
        -------
        Layer2Result
        """
        proba = self._validate_proba(proba, "proba", allow_ndim=self._ref_ndim)
        n = proba.shape[0]
        score = _canonical_score(proba)

        if confidences is not None:
            confidences = np.asarray(confidences, dtype=float).ravel()
            if confidences.shape[0] != n:
                raise ValueError(
                    f"confidences has length {confidences.shape[0]} but proba "
                    f"has {n} windows"
                )
            if np.any(~np.isfinite(confidences)) or np.any(
                (confidences < 0.0) | (confidences > 1.0)
            ):
                raise ValueError("confidences must be finite values in [0, 1]")

        yt = None
        if y_true is not None:
            yt = np.asarray(y_true).ravel()
            if yt.shape[0] != n:
                raise ValueError(
                    f"y_true has length {yt.shape[0]} but proba has {n} windows"
                )
            n_classes = 2 if proba.ndim == 1 else proba.shape[1]
            if not np.issubdtype(yt.dtype, np.integer) or np.any(
                (yt < 0) | (yt >= n_classes)
            ):
                raise ValueError(
                    f"y_true must be integer labels in [0, {n_classes - 1}]"
                )
            yt = yt.astype(int)

        signals: dict[str, ModelSignal] = {}

        # --- label-free signals -------------------------------------------
        psi_value = _psi(self._ref_score, score, bins=self.config["psi_bins"])
        breached = self._alerts["psi_proba"].update(psi_value)
        signals["psi_proba"] = ModelSignal(
            name="psi_proba",
            value=psi_value,
            threshold=self.config["psi_threshold"],
            breached=breached,
            details={
                "bins": self.config["psi_bins"],
                "n_reference": int(self._ref_score.size),
                "n_current": int(n),
            },
        )

        ks_res = ks_2samp(self._ref_score, score)
        ks_stat = float(ks_res.statistic)
        ks_p = float(ks_res.pvalue)
        breached = self._alerts["ks_proba"].update(ks_stat)
        signals["ks_proba"] = ModelSignal(
            name="ks_proba",
            value=ks_stat,
            threshold=self.config["ks_threshold"],
            breached=breached,
            details={"pvalue": ks_p},
        )

        cur_mean_conf = float(_confidence(proba, confidences).mean())
        drop = self._ref_mean_conf - cur_mean_conf
        breached = self._alerts["mean_confidence_drop"].update(drop)
        signals["mean_confidence_drop"] = ModelSignal(
            name="mean_confidence_drop",
            value=drop,
            threshold=self.config["conf_drop_threshold"],
            breached=breached,
            details={
                "ref_mean_confidence": self._ref_mean_conf,
                "cur_mean_confidence": cur_mean_conf,
            },
        )

        # --- label-gated signals ------------------------------------------
        if yt is not None:
            if proba.ndim == 1:
                p1 = proba
                pred = (proba >= 0.5).astype(int)
            elif proba.shape[1] == 2:
                p1 = proba[:, 1]
                pred = proba.argmax(axis=1)
            else:
                raise ValueError(
                    "ece_drift and rolling_accuracy require a binary model; "
                    f"got {proba.shape[1]} classes"
                )

            cur_ece = expected_calibration_error(yt, p1)
            if self._ref_ece is None:
                # Bootstrap the reference from the first labeled batch.
                self._ref_ece = cur_ece
                drift = 0.0
                note = "reference ECE bootstrapped from this batch"
            else:
                drift = abs(cur_ece - self._ref_ece)
                note = "reference ECE from earlier labeled data"
            breached = self._alerts["ece_drift"].update(drift)
            signals["ece_drift"] = ModelSignal(
                name="ece_drift",
                value=drift,
                threshold=self.config["ece_drift_threshold"],
                breached=breached,
                details={
                    "ref_ece": self._ref_ece,
                    "cur_ece": cur_ece,
                    "note": note,
                },
            )

            batch_acc = float((pred == yt).mean())
            self._acc_history.append(batch_acc)
            rolling_acc = float(np.mean(list(self._acc_history)))
            breached = self._alerts["rolling_accuracy"].update(rolling_acc)
            signals["rolling_accuracy"] = ModelSignal(
                name="rolling_accuracy",
                value=rolling_acc,
                threshold=self.config["accuracy_threshold"],
                breached=breached,
                details={
                    "batch_accuracy": batch_acc,
                    "batches_in_window": len(self._acc_history),
                },
            )

        self._n_windows += n
        degraded = any(s.breached for s in signals.values())
        health = self._health(signals)
        return Layer2Result(
            model_id=self.model_id,
            n_windows=self._n_windows,
            signals=signals,
            degraded=degraded,
            health=health,
        )

    def _health(self, signals: dict[str, ModelSignal]) -> float:
        weights_cfg = self.config["signal_weights"]
        total = 0.0
        penalty = 0.0
        for name, sig in signals.items():
            w = 1.0 if weights_cfg is None else float(weights_cfg.get(name, 1.0))
            total += w
            if sig.breached:
                penalty += w
        if total == 0.0:
            return 100.0
        return float(100.0 * (1.0 - penalty / total))

    def reset(self) -> None:
        """Clear breach history, labeled reference state, and window count."""
        for alert in self._alerts.values():
            alert.reset()
        self._acc_history.clear()
        if self.config["ece_ref"] is None:
            self._ref_ece = None
        else:
            self._ref_ece = self.config["ece_ref"]
        self._n_windows = 0

    @property
    def n_windows(self) -> int:
        """Cumulative windows processed."""
        return self._n_windows
