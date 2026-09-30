"""Cross-channel consistency checks.

Two checks live here:

- :func:`acc_consistency`: compare two accelerometers by signal magnitude
  (Pearson correlation plus cross-correlation lag / desync flag).
- :func:`hr_agreement`: estimate heart rate from two pulsatile signals via
  peak detection and check agreement.

Channel names and roles are always caller-supplied parameters — nothing is
hardcoded.
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def _as_1d(x: np.ndarray, label: str) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        raise TypeError(f"{label} must be a numpy array, got {type(x).__name__}.")
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"{label} must be 1D, got shape {a.shape}.")
    if a.size == 0:
        raise ValueError(f"{label} must not be empty.")
    return a


def _as_fs(fs: float, label: str) -> float:
    if not isinstance(fs, (int, float)) or not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"{label} must be a positive number, got {fs!r}.")
    return float(fs)


def _interp_nan(x: np.ndarray, label: str) -> np.ndarray:
    """Linearly interpolate NaNs (edges filled with nearest valid value)."""
    nan = np.isnan(x)
    if not np.any(nan):
        return x.copy()
    if np.all(nan):
        raise ValueError(f"{label} is all-NaN; cannot interpolate.")
    idx = np.arange(x.size)
    return np.interp(idx, idx[~nan], x[~nan])


def acc_consistency(
    acc_a: dict[str, np.ndarray],
    fs_a: float,
    acc_b: dict[str, np.ndarray],
    fs_b: float,
    desync_lag_s: float = 0.5,
) -> dict:
    """Compare two accelerometers via signal magnitude.

    Each dict maps an axis name to a 1D array (e.g. ``{"x": ..., "y": ...,
    "z": ...}``); axis names are caller-chosen, never hardcoded. Per-device
    magnitude is ``sqrt(sum(axis**2))``. Magnitudes are resampled to a common
    rate (the lower of the two) over their shared time span, then compared by
    Pearson correlation and by the lag of peak cross-correlation.

    Args:
        acc_a: First device's axis dict; ``fs_a`` its sampling rate in Hz.
        acc_b: Second device's axis dict; ``fs_b`` its sampling rate in Hz.
        desync_lag_s: ``|lag_s|`` above this raises the desync flag.

    Returns:
        ``{"correlation", "lag_s", "lag_samples", "desync", "common_fs",
        "n_samples"}``. ``lag_s`` is the lag of the second device's signal
        relative to the first (positive = second device delayed).

    Raises:
        TypeError/ValueError: On bad input or insufficient overlap.
    """
    for d, label in ((acc_a, "acc_a"), (acc_b, "acc_b")):
        if not isinstance(d, dict) or not d:
            raise ValueError(f"{label} must be a non-empty dict of axis -> array.")
        lengths = set()
        for axis, arr in d.items():
            a = _as_1d(arr, f"{label}['{axis}']")
            d[axis] = a  # normalize in place for magnitude computation
            lengths.add(a.size)
        if len(lengths) != 1:
            raise ValueError(
                f"All axes in {label} must have the same length, got {sorted(lengths)}."
            )
    fs_a = _as_fs(fs_a, "fs_a")
    fs_b = _as_fs(fs_b, "fs_b")
    if (
        not isinstance(desync_lag_s, (int, float))
        or not np.isfinite(desync_lag_s)
        or desync_lag_s < 0
    ):
        raise ValueError(
            f"desync_lag_s must be a non-negative number, got {desync_lag_s!r}."
        )

    mag_a = np.sqrt(sum(_interp_nan(a, "acc_a axis") ** 2 for a in acc_a.values()))
    mag_b = np.sqrt(sum(_interp_nan(b, "acc_b axis") ** 2 for b in acc_b.values()))

    common_fs = min(fs_a, fs_b)
    duration_s = min(mag_a.size / fs_a, mag_b.size / fs_b)
    n = int(duration_s * common_fs)
    if n < 8:
        raise ValueError(
            f"Not enough overlapping samples for comparison (got {n}, need >= 8)."
        )
    ra = signal.resample(mag_a, n)
    rb = signal.resample(mag_b, n)

    ma, mb = float(np.mean(ra)), float(np.mean(rb))
    sa, sb = float(np.std(ra)), float(np.std(rb))
    if sa == 0.0 or sb == 0.0:
        correlation = 1.0 if abs(ma - mb) <= 1e-12 else 0.0
    else:
        correlation = float(np.corrcoef(ra, rb)[0, 1])
        if not np.isfinite(correlation):
            correlation = 0.0

    a0, b0 = ra - ma, rb - mb
    if np.all(a0 == 0.0) or np.all(b0 == 0.0):
        lag_samples, lag_s = 0, 0.0
    else:
        cc = signal.correlate(a0, b0, mode="full")
        lags = signal.correlation_lags(n, n, mode="full")
        # correlate(a, b) peaks at -d when b is delayed by d vs a; negate so
        # that a positive lag means the second device is delayed.
        lag_samples = -int(lags[int(np.argmax(cc))])
        lag_s = float(lag_samples / common_fs)

    return {
        "correlation": correlation,
        "lag_s": lag_s,
        "lag_samples": lag_samples,
        "desync": bool(abs(lag_s) > desync_lag_s),
        "common_fs": float(common_fs),
        "n_samples": int(n),
    }


def _estimate_hr_bpm(x: np.ndarray, fs: float, label: str) -> dict:
    """Peak-based heart-rate estimate; NaN-tolerant.

    Returns ``{"hr_bpm", "n_peaks", "reason"}`` with ``hr_bpm`` None when no
    reliable estimate exists.
    """
    x = _as_1d(x, label)
    fs = _as_fs(fs, f"fs for {label}")
    if float(np.mean(np.isnan(x))) > 0.5:
        return {"hr_bpm": None, "n_peaks": 0, "reason": "too much missing data"}
    xi = _interp_nan(x, label)
    # Peak bar from the signal's own robust range: prominence rejects
    # broad noise, while the height gate rejects small bumps sitting in the
    # deep valleys next to tall spikes (their prominence is inflated).
    hi = float(np.percentile(xi, 99.5))
    lo = float(np.median(xi))
    peak_range = hi - lo
    if peak_range <= 0.0 or not np.isfinite(peak_range):
        return {"hr_bpm": None, "n_peaks": 0, "reason": "flat signal"}
    distance = max(1, int(round(0.4 * fs)))  # refractory period: max 150 bpm
    peaks, _ = signal.find_peaks(
        xi,
        distance=distance,
        prominence=0.3 * peak_range,
        height=lo + 0.5 * peak_range,
    )
    if peaks.size < 3:
        return {
            "hr_bpm": None,
            "n_peaks": int(peaks.size),
            "reason": "too few peaks",
        }
    rr_s = np.diff(peaks) / fs
    hr_bpm = 60.0 / float(np.median(rr_s))
    return {"hr_bpm": hr_bpm, "n_peaks": int(peaks.size), "reason": "ok"}


def hr_agreement(
    ecg: np.ndarray,
    fs_ecg: float,
    bvp: np.ndarray,
    fs_bvp: float,
    tol_bpm: float = 5.0,
) -> dict:
    """Estimate HR from two pulsatile signals via peak detection and compare.

    The two inputs are positional parameters — callers pass whichever channels
    they configured (typically ECG and BVP); nothing is hardcoded.

    Args:
        ecg: First pulsatile signal (1D array).
        fs_ecg: Its sampling rate in Hz.
        bvp: Second pulsatile signal (1D array).
        fs_bvp: Its sampling rate in Hz.
        tol_bpm: Agreement tolerance in beats per minute.

    Returns:
        ``{"hr_ecg_bpm", "hr_bvp_bpm", "abs_diff_bpm", "agreement",
        "n_peaks_ecg", "n_peaks_bvp"}``. HR values are None when no reliable
        peak-based estimate exists; ``agreement`` is then False.

    Raises:
        TypeError/ValueError: On bad input.
    """
    if (
        not isinstance(tol_bpm, (int, float))
        or not np.isfinite(tol_bpm)
        or tol_bpm < 0
    ):
        raise ValueError(f"tol_bpm must be a non-negative number, got {tol_bpm!r}.")
    e = _estimate_hr_bpm(ecg, fs_ecg, "ecg")
    b = _estimate_hr_bpm(bvp, fs_bvp, "bvp")
    if e["hr_bpm"] is None or b["hr_bpm"] is None:
        diff, agreement = None, False
    else:
        diff = abs(e["hr_bpm"] - b["hr_bpm"])
        agreement = diff <= tol_bpm
    return {
        "hr_ecg_bpm": e["hr_bpm"],
        "hr_bvp_bpm": b["hr_bpm"],
        "abs_diff_bpm": diff,
        "agreement": bool(agreement),
        "n_peaks_ecg": e["n_peaks"],
        "n_peaks_bvp": b["n_peaks"],
    }
