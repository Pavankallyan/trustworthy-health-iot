"""Stuck-at (frozen value) detection."""

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


def detect_stuck(
    x: np.ndarray, fs: float, tol: float = 1e-9, min_duration_s: float = 5.0
) -> dict:
    """Detect runs where the signal is frozen within ``tol``.

    Consecutive samples whose absolute difference is ``<= tol`` form a flat
    run; NaN samples break runs (missing data is a dropout, not stuck). A run
    counts as stuck when it lasts at least ``min_duration_s`` seconds.

    Args:
        x: 1D array of samples.
        fs: Sampling rate in Hz (> 0).
        tol: Maximum absolute sample-to-sample change still considered flat.
        min_duration_s: Minimum stuck duration in seconds (> 0).

    Returns:
        ``{"is_stuck", "longest_stuck_s", "longest_stuck_samples",
        "stuck_fraction", "n_stuck_runs", "value"}`` where ``stuck_fraction``
        is the fraction of non-NaN samples inside qualifying stuck runs and
        ``value`` is the frozen value of the longest run (None if not stuck).

    Raises:
        TypeError/ValueError: On bad input.
    """
    x = _as_1d(x, "x")
    fs = _as_fs(fs)
    if not isinstance(tol, (int, float)) or not np.isfinite(tol) or tol < 0:
        raise ValueError(f"tol must be a non-negative finite number, got {tol!r}.")
    if (
        not isinstance(min_duration_s, (int, float))
        or not np.isfinite(min_duration_s)
        or min_duration_s <= 0
    ):
        raise ValueError(
            f"min_duration_s must be a positive number, got {min_duration_s!r}."
        )

    valid = ~np.isnan(x)
    n_valid = int(np.sum(valid))
    empty = {
        "is_stuck": False,
        "longest_stuck_s": 0.0,
        "longest_stuck_samples": 0,
        "stuck_fraction": 0.0,
        "n_stuck_runs": 0,
        "value": None,
    }
    if n_valid < 2:
        return empty

    idx = np.nonzero(valid)[0]
    diffs = np.abs(np.diff(x[valid]))
    # A diff counts as flat only if both samples are real and adjacent in time.
    flat = (diffs <= tol) & (np.diff(idx) == 1)

    min_samples = max(2, int(round(min_duration_s * fs)))
    stuck_mask = np.zeros(x.size, dtype=bool)
    run_lengths: list[int] = []
    count = 0
    for i, is_flat in enumerate(flat):
        if is_flat:
            count += 1
        else:
            if count:
                run_lengths.append(count)
                if count + 1 >= min_samples:
                    stuck_mask[idx[i - count] : idx[i] + 1] = True
                count = 0
    if count:
        run_lengths.append(count)
        if count + 1 >= min_samples:
            stuck_mask[idx[len(flat) - count] : idx[len(flat)] + 1] = True

    qualifying = [c + 1 for c in run_lengths if c + 1 >= min_samples]
    if not qualifying:
        return empty

    longest = max(qualifying)
    # Frozen value of the longest qualifying run: first sample of that run.
    value = None
    # Re-scan to find the start index of a longest run: a run of `count`
    # flat diffs ending at diff index i covers samples idx[i-count+1..i+1].
    count = 0
    for i, is_flat in enumerate(flat):
        if is_flat:
            count += 1
            if count + 1 == longest:
                value = float(x[idx[i - count + 1]])
                break
        else:
            count = 0

    return {
        "is_stuck": True,
        "longest_stuck_s": float(longest / fs),
        "longest_stuck_samples": int(longest),
        "stuck_fraction": float(np.sum(stuck_mask) / n_valid),
        "n_stuck_runs": len(qualifying),
        "value": value,
    }
