"""Time-domain feature extraction: windows -> model-ready feature matrix.

``extract_features`` turns a list of :class:`Window` objects into a pandas
DataFrame with one row per window.  For every requested channel it computes
six time-domain statistics: mean, std, min, max, RMS, peak-to-peak, plus a
zero-crossing count of the demeaned signal.  Column names are
``"<channel>__<statistic>"``; column order follows the ``channels`` argument,
so output is fully deterministic.

NaN handling (documented, per contract):
  * NaNs are treated as missing samples and *excluded* from each statistic
    (statistics are computed over the non-NaN samples only).
  * If a channel is entirely NaN inside a window, that window's feature
    values for the channel are NaN in the output frame.  The model never sees
    NaNs because :func:`trustguard_model.baseline.train_stress_model` builds a
    pipeline whose first step is a median ``SimpleImputer``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._compat import Window  # noqa: F401  (re-exported for convenience)

__all__ = ["extract_features", "FEATURE_STATS"]

#: Statistics computed per channel, in output column order.
FEATURE_STATS = (
    "mean",
    "std",
    "min",
    "max",
    "rms",
    "peak_to_peak",
    "zero_crossings",
)


def _zero_crossings(valid: np.ndarray) -> float:
    """Count sign changes of the demeaned signal (NaN-free input)."""
    centered = valid - valid.mean()
    signs = np.sign(centered)
    signs = signs[signs != 0]
    if signs.size < 2:
        return 0.0
    return float(np.sum(signs[1:] != signs[:-1]))


def _channel_features(x: np.ndarray) -> dict[str, float]:
    """Compute the seven time-domain stats for one 1D channel array."""
    valid = x[~np.isnan(x)]
    if valid.size == 0:
        return {stat: float("nan") for stat in FEATURE_STATS}
    vmin = float(valid.min())
    vmax = float(valid.max())
    return {
        "mean": float(valid.mean()),
        "std": float(valid.std()),
        "min": vmin,
        "max": vmax,
        "rms": float(np.sqrt(np.mean(valid**2))),
        "peak_to_peak": vmax - vmin,
        "zero_crossings": _zero_crossings(valid),
    }


def _validate_window_channels(window, channels: list[str], index: int) -> None:
    if not hasattr(window, "channels") or not isinstance(window.channels, dict):
        raise TypeError(
            f"windows[{index}]: expected a Window-like object with a "
            f"'channels' dict, got {type(window).__name__}"
        )
    for ch in channels:
        if ch not in window.channels:
            raise ValueError(
                f"windows[{index}]: unknown channel '{ch}' "
                f"(available: {sorted(window.channels)})"
            )
        arr = window.channels[ch]
        try:
            arr = np.asarray(arr, dtype=np.float64).ravel()
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"windows[{index}]: channel '{ch}' could not be converted to a "
                f"float64 array: {exc}"
            ) from exc
        if arr.size == 0:
            raise ValueError(
                f"windows[{index}]: channel '{ch}' is empty; channels must "
                "contain at least one sample"
            )


def extract_features(windows: list, channels: list[str]) -> pd.DataFrame:
    """Extract per-channel time-domain features, one row per window.

    Parameters
    ----------
    windows:
        Non-empty list of ``Window`` objects (Layer 1 type from
        ``trustguard_data.stream``; anything exposing a ``channels`` dict of
        1D float arrays is accepted so tests can run before/without the
        sibling package).
    channels:
        Non-empty list of channel names.  Every channel must exist in every
        window.

    Returns
    -------
    pandas.DataFrame
        Shape ``(len(windows), len(channels) * 7)`` with columns
        ``"<channel>__<statistic>"``.  Deterministic: no randomness involved.
    """
    if not isinstance(windows, list) or len(windows) == 0:
        raise ValueError("windows must be a non-empty list of Window objects")
    if not isinstance(channels, list) or len(channels) == 0:
        raise ValueError("channels must be a non-empty list of channel names")
    for ch in channels:
        if not isinstance(ch, str) or not ch:
            raise ValueError("channel names must be non-empty strings")

    rows: list[dict[str, float]] = []
    for i, window in enumerate(windows):
        _validate_window_channels(window, channels, i)
        row: dict[str, float] = {}
        for ch in channels:
            x = np.asarray(window.channels[ch], dtype=np.float64).ravel()
            for stat, value in _channel_features(x).items():
                row[f"{ch}__{stat}"] = value
        rows.append(row)

    columns = [f"{ch}__{stat}" for ch in channels for stat in FEATURE_STATS]
    return pd.DataFrame(rows, columns=columns)
