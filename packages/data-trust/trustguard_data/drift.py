"""Distribution-drift detectors: two-sample KS test, PSI, and trend slope."""

from __future__ import annotations

import numpy as np
from scipy import stats


def _clean_1d(x: np.ndarray, label: str) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        raise TypeError(f"{label} must be a numpy array, got {type(x).__name__}.")
    if x.ndim != 1:
        raise ValueError(f"{label} must be 1D, got shape {x.shape}.")
    return x[~np.isnan(x)]


def ks_drift(x: np.ndarray, baseline_x: np.ndarray) -> dict:
    """Two-sample Kolmogorov-Smirnov test of ``x`` against a baseline.

    NaNs are dropped from both inputs before testing.

    Returns:
        ``{"statistic", "pvalue", "n", "n_baseline"}``. A small ``pvalue``
        means the window's distribution differs from the baseline.

    Raises:
        TypeError/ValueError: On bad input or fewer than 3 valid samples.
    """
    xv = _clean_1d(x, "x")
    bv = _clean_1d(baseline_x, "baseline_x")
    if xv.size < 3 or bv.size < 3:
        raise ValueError(
            "ks_drift needs at least 3 valid samples in each input "
            f"(got {xv.size} and {bv.size})."
        )
    res = stats.ks_2samp(xv, bv)
    return {
        "statistic": float(res.statistic),
        "pvalue": float(res.pvalue),
        "n": int(xv.size),
        "n_baseline": int(bv.size),
    }


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index of ``actual`` relative to ``expected``.

    Bin edges are quantiles of ``expected`` (extended to ``-inf``/``inf`` so
    every ``actual`` sample is counted); bin counts get epsilon smoothing.
    Returns 0.0 for identical distributions; larger values mean more drift
    (rule of thumb: < 0.1 negligible, 0.1-0.25 watch, >= 0.25 drift).
    Returns ``inf`` when ``expected`` is degenerate (constant) but ``actual``
    is not. NaNs are dropped from both inputs.

    Raises:
        TypeError/ValueError: On bad input or empty inputs after NaN removal.
    """
    e = _clean_1d(expected, "expected")
    a = _clean_1d(actual, "actual")
    if not isinstance(bins, int) or bins < 2:
        raise ValueError(f"bins must be an int >= 2, got {bins!r}.")
    if e.size == 0 or a.size == 0:
        raise ValueError("psi needs at least one valid sample in each input.")

    edges = np.unique(np.percentile(e, np.linspace(0, 100, bins + 1)))
    if edges.size < 2:
        # Degenerate (constant) expected distribution.
        return 0.0 if np.all(a == e[0]) else float("inf")
    edges = np.concatenate(([-np.inf], edges[1:-1], [np.inf]))

    eps = 1e-8
    e_counts, _ = np.histogram(e, bins=edges)
    a_counts, _ = np.histogram(a, bins=edges)
    e_p = np.clip(e_counts / e_counts.sum(), eps, None)
    a_p = np.clip(a_counts / a_counts.sum(), eps, None)
    e_p /= e_p.sum()
    a_p /= a_p.sum()
    return float(np.sum((a_p - e_p) * np.log(a_p / e_p)))


def trend_slope(x: np.ndarray, fs: float) -> dict:
    """Least-squares slope of the signal in units per second.

    NaNs are dropped before fitting. A perfectly flat signal returns a slope
    of 0 with ``pvalue`` 1.0.

    Returns:
        ``{"slope_per_s", "intercept", "rvalue", "pvalue", "stderr", "n"}``.
        ``pvalue`` (from ``scipy.stats.linregress``) is the significance
        proxy: a small p-value means the trend is unlikely to be noise.

    Raises:
        TypeError/ValueError: On bad input, non-positive ``fs``, or fewer
            than 3 valid samples.
    """
    xv = _clean_1d(x, "x")
    if not isinstance(fs, (int, float)) or not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be a positive number, got {fs!r}.")
    if not np.isfinite(xv).all():
        raise ValueError("x contains non-finite (inf) values after NaN removal.")
    # Keep original sample times so NaN gaps don't distort the slope.
    mask = ~np.isnan(np.asarray(x, dtype=np.float64).ravel())
    t = np.arange(mask.size, dtype=np.float64)[mask] / fs
    xv = np.asarray(x, dtype=np.float64).ravel()[mask]
    if xv.size < 3:
        raise ValueError(
            f"trend_slope needs at least 3 valid samples, got {xv.size}."
        )
    if np.std(xv) == 0.0:
        return {
            "slope_per_s": 0.0,
            "intercept": float(xv[0]),
            "rvalue": 0.0,
            "pvalue": 1.0,
            "stderr": 0.0,
            "n": int(xv.size),
        }
    res = stats.linregress(t, xv)
    return {
        "slope_per_s": float(res.slope),
        "intercept": float(res.intercept),
        "rvalue": float(res.rvalue),
        "pvalue": float(res.pvalue),
        "stderr": float(res.stderr),
        "n": int(xv.size),
    }
