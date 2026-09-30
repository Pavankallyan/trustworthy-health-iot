"""Label-free degradation proxies.

When ground-truth labels are unavailable (the normal case for a deployed
health-IoT model), two cheap proxies still carry signal about model health:

* :class:`EntropyMonitor` — tracks the mean predictive entropy of ``proba``.
  Rising entropy means the model is getting less decisive, a classic
  precursor of degradation under distribution shift.
* :class:`SurrogateAgreementMonitor` — compares the deployed model's
  predicted labels against a tiny, cheap heuristic classifier (a callable
  supplied by the user, e.g. a threshold rule on one feature).  The surrogate
  is *not* expected to be accurate; what matters is that its agreement with
  the model is stable in-distribution, so a drop in agreement flags that the
  model started deciding differently.

Both are deterministic and reuse :class:`DegradationAlert` for
threshold-plus-hysteresis alerting.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .alerting import DEFAULT_CLEAR_STREAK, DegradationAlert
from .monitors import ModelSignal

__all__ = ["EntropyMonitor", "SurrogateAgreementMonitor"]

_EPS = 1e-12


def _entropy(proba: np.ndarray) -> np.ndarray:
    """Per-window Shannon entropy in nats (binary or multiclass)."""
    arr = np.asarray(proba, dtype=float)
    if arr.ndim == 1:
        p = np.clip(arr, _EPS, 1.0 - _EPS)
        return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    p = np.clip(arr, _EPS, 1.0)
    pn = p / p.sum(axis=1, keepdims=True)
    return -np.sum(pn * np.log(np.clip(pn, _EPS, 1.0)), axis=1)


def _validate_proba(proba, where: str, ref_ndim: int | None = None) -> np.ndarray:
    arr = np.asarray(proba, dtype=float)
    if arr.ndim not in (1, 2):
        raise ValueError(f"{where}: proba must be 1D or 2D, got ndim={arr.ndim}")
    if arr.shape[0] == 0:
        raise ValueError(f"{where}: proba must contain at least one window")
    if ref_ndim is not None and arr.ndim != ref_ndim:
        raise ValueError(
            f"{where}: proba ndim ({arr.ndim}) does not match reference ndim "
            f"({ref_ndim})"
        )
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{where}: proba must be finite")
    if np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError(f"{where}: proba values must lie in [0, 1]")
    if arr.ndim == 2:
        if arr.shape[1] < 2:
            raise ValueError(f"{where}: 2D proba needs >= 2 classes")
        if not np.allclose(arr.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError(f"{where}: 2D proba rows must sum to 1")
    return arr


class EntropyMonitor:
    """Watch mean predictive entropy against a trusted reference level.

    Parameters
    ----------
    reference_proba:
        Probabilities from a trusted reference period (1D binary or 2D
        multiclass, rows summing to 1).  Sets the reference mean entropy.
    threshold:
        Breach when ``mean_entropy_now - mean_entropy_ref > threshold``.
        Non-negative float.
    clear_streak_required:
        K — consecutive clear updates needed to un-breach (default 3).
    margin:
        Hysteresis band width (non-negative float).
    """

    def __init__(
        self,
        reference_proba: np.ndarray,
        threshold: float = 0.15,
        clear_streak_required: int = DEFAULT_CLEAR_STREAK,
        margin: float = 0.0,
    ) -> None:
        self._ref = _validate_proba(reference_proba, "reference_proba")
        if (
            not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or threshold < 0
        ):
            raise ValueError("threshold must be a non-negative number")
        self.threshold = float(threshold)
        self._ref_mean_entropy = float(_entropy(self._ref).mean())
        self._alert = DegradationAlert(
            "prediction_entropy",
            self.threshold,
            direction="above",
            clear_streak_required=clear_streak_required,
            margin=margin,
        )

    @property
    def reference_mean_entropy(self) -> float:
        """Mean entropy of the reference period."""
        return self._ref_mean_entropy

    def update(self, proba: np.ndarray) -> ModelSignal:
        """Score one batch; return a :class:`ModelSignal`.

        ``value`` is the *increase* in mean entropy vs the reference (positive
        means the model grew less decisive).
        """
        proba = _validate_proba(proba, "proba", ref_ndim=self._ref.ndim)
        cur = float(_entropy(proba).mean())
        increase = cur - self._ref_mean_entropy
        breached = self._alert.update(increase)
        return ModelSignal(
            name="prediction_entropy",
            value=increase,
            threshold=self.threshold,
            breached=breached,
            details={
                "ref_mean_entropy": self._ref_mean_entropy,
                "cur_mean_entropy": cur,
            },
        )

    def reset(self) -> None:
        """Clear breach history."""
        self._alert.reset()


class SurrogateAgreementMonitor:
    """Watch agreement between the model and a cheap heuristic surrogate.

    Parameters
    ----------
    surrogate:
        Callable ``surrogate(X) -> np.ndarray`` mapping a 2D feature matrix
        ``(n_windows, n_features)`` to 1D predicted integer labels.  A tiny
        threshold rule is enough — e.g. ``lambda X: (X[:, 0] > 0).astype(int)``.
    threshold:
        Breach when the agreement fraction drops *below* this value
        (default 0.80).
    clear_streak_required:
        K — consecutive clear updates needed to un-breach (default 3).
    margin:
        Hysteresis band width (non-negative float).
    """

    def __init__(
        self,
        surrogate: Callable[[np.ndarray], np.ndarray],
        threshold: float = 0.80,
        clear_streak_required: int = DEFAULT_CLEAR_STREAK,
        margin: float = 0.0,
    ) -> None:
        if not callable(surrogate):
            raise TypeError("surrogate must be a callable taking X -> labels")
        if (
            not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not 0.0 <= float(threshold) <= 1.0
        ):
            raise ValueError("threshold must be a number in [0, 1]")
        self.surrogate = surrogate
        self.threshold = float(threshold)
        self._alert = DegradationAlert(
            "surrogate_agreement",
            self.threshold,
            direction="below",
            clear_streak_required=clear_streak_required,
            margin=margin,
        )

    def update(self, proba: np.ndarray, X: np.ndarray) -> ModelSignal:
        """Score one batch; return a :class:`ModelSignal`.

        Model labels are ``(p >= 0.5)`` for 1D proba or ``argmax`` for 2D.
        ``value`` is the fraction of windows where model and surrogate agree.
        """
        proba = _validate_proba(proba, "proba")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n_windows, n_features), got ndim={X.ndim}")
        if X.shape[0] != proba.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} rows but proba has {proba.shape[0]} windows"
            )
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one window")
        try:
            surr = np.asarray(self.surrogate(X)).ravel()
        except Exception as exc:
            raise ValueError(f"surrogate(X) raised: {exc}") from exc
        if surr.shape[0] != proba.shape[0]:
            raise ValueError(
                f"surrogate returned {surr.shape[0]} labels for "
                f"{proba.shape[0]} windows"
            )
        if not np.issubdtype(surr.dtype, np.integer):
            raise ValueError("surrogate must return integer labels")
        model_labels = (
            (proba >= 0.5).astype(int) if proba.ndim == 1 else proba.argmax(axis=1)
        )
        agreement = float((model_labels == surr.astype(int)).mean())
        breached = self._alert.update(agreement)
        return ModelSignal(
            name="surrogate_agreement",
            value=agreement,
            threshold=self.threshold,
            breached=breached,
            details={
                "n_windows": int(proba.shape[0]),
                "n_agree": int((model_labels == surr.astype(int)).sum()),
            },
        )

    def reset(self) -> None:
        """Clear breach history."""
        self._alert.reset()
