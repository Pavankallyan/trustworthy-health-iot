"""NaN / missing-run (dropout) detection."""

from __future__ import annotations

import numpy as np


def _as_1d(x: np.ndarray, label: str) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        raise TypeError(f"{label} must be a numpy array, got {type(x).__name__}.")
    if x.ndim != 1:
        raise ValueError(f"{label} must be 1D, got shape {x.shape}.")
    if x.size == 0:
        raise ValueError(f"{label} must not be empty.")
    return x


def _as_fs(fs: float) -> float:
    if not isinstance(fs, (int, float)) or not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be a positive number, got {fs!r}.")
    return float(fs)


def _nan_runs(nan: np.ndarray) -> list[int]:
    """Lengths (in samples) of consecutive-True runs in a boolean mask."""
    runs: list[int] = []
    count = 0
    for flag in nan:
        if flag:
            count += 1
        elif count:
            runs.append(count)
            count = 0
    if count:
        runs.append(count)
    return runs


def detect_dropout(x: np.ndarray, fs: float) -> dict:
    """Describe missing (NaN) runs in a channel.

    Args:
        x: 1D array of samples; NaN marks a missing sample.
        fs: Sampling rate in Hz (> 0).

    Returns:
        ``{"n_total", "n_missing", "missing_fraction", "n_runs",
        "longest_nan_run_samples", "longest_nan_run_s"}``.

    Raises:
        TypeError/ValueError: On bad input.
    """
    x = _as_1d(x, "x")
    fs = _as_fs(fs)
    nan = np.isnan(x)
    n_missing = int(np.sum(nan))
    runs = _nan_runs(nan)
    longest = max(runs) if runs else 0
    return {
        "n_total": int(x.size),
        "n_missing": n_missing,
        "missing_fraction": float(n_missing / x.size),
        "n_runs": len(runs),
        "longest_nan_run_samples": int(longest),
        "longest_nan_run_s": float(longest / fs),
    }
