"""Schema and range validators for Layer 1 windows."""

from __future__ import annotations

import numpy as np

from .stream import Window, validate_window


def check_schema(window: Window, expected: dict[str, tuple]) -> list[dict]:
    """Check that expected channels exist with the right dtype kind and length.

    Args:
        window: The window under test.
        expected: channel name -> ``(dtype_kind, min_len)`` tuple, e.g.
            ``{"ecg": ("f", 100)}``. ``dtype_kind`` is a numpy dtype ``kind``
            character (``"f"`` float, ``"i"`` int, ...).

    Returns:
        One result dict per expected channel: ``{"channel", "check": "schema",
        "ok", "message", "dtype", "length"}``. Channels listed in ``expected``
        but absent from the window yield ``ok=False`` entries; channels in the
        window but not in ``expected`` are ignored.

    Raises:
        TypeError/ValueError: On bad input.
    """
    validate_window(window)
    if not isinstance(expected, dict) or not expected:
        raise ValueError(
            "expected must be a non-empty dict of channel -> (dtype_kind, min_len)."
        )
    results: list[dict] = []
    for name, spec in expected.items():
        if not isinstance(spec, tuple) or len(spec) != 2:
            raise ValueError(
                f"Schema spec for channel '{name}' must be a "
                f"(dtype_kind, min_len) tuple, got {spec!r}."
            )
        dtype_kind, min_len = spec
        if not isinstance(dtype_kind, str) or len(dtype_kind) != 1:
            raise ValueError(
                f"dtype_kind for channel '{name}' must be a single character, "
                f"got {dtype_kind!r}."
            )
        if not isinstance(min_len, int) or min_len < 1:
            raise ValueError(
                f"min_len for channel '{name}' must be a positive int, "
                f"got {min_len!r}."
            )
        if name not in window.channels:
            results.append(
                {
                    "channel": name,
                    "check": "schema",
                    "ok": False,
                    "message": f"channel '{name}' missing from window.",
                    "dtype": None,
                    "length": None,
                }
            )
            continue
        arr = window.channels[name]
        if arr.dtype.kind != dtype_kind:
            ok, msg = False, (
                f"dtype kind '{arr.dtype.kind}' != expected '{dtype_kind}'."
            )
        elif arr.size < min_len:
            ok, msg = False, f"length {arr.size} < min_len {min_len}."
        else:
            ok, msg = True, "ok"
        results.append(
            {
                "channel": name,
                "check": "schema",
                "ok": bool(ok),
                "message": msg,
                "dtype": arr.dtype.kind,
                "length": int(arr.size),
            }
        )
    return results


def check_range(window: Window, ranges: dict[str, tuple[float, float]]) -> list[dict]:
    """Check that samples fall inside plausible per-channel ranges.

    NaN samples are *not* range violations; they are reported separately as
    ``n_nan`` and handled by the dropout detector.

    Args:
        window: The window under test.
        ranges: channel name -> ``(lo, hi)`` plausible range, ``lo < hi``.

    Returns:
        One result dict per ranged channel: ``{"channel", "check": "range",
        "ok", "message", "fraction_out_of_range", "n_out", "n_total", "n_nan",
        "lo", "hi"}``. Channels listed in ``ranges`` but absent from the
        window yield ``ok=False`` entries.

    Raises:
        TypeError/ValueError: On bad input.
    """
    validate_window(window)
    if not isinstance(ranges, dict) or not ranges:
        raise ValueError("ranges must be a non-empty dict of channel -> (lo, hi).")
    results: list[dict] = []
    for name, bounds in ranges.items():
        if not isinstance(bounds, tuple) or len(bounds) != 2:
            raise ValueError(
                f"Range for channel '{name}' must be a (lo, hi) tuple, "
                f"got {bounds!r}."
            )
        lo, hi = bounds
        for value, label in ((lo, "lo"), (hi, "hi")):
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(
                    f"Range {label} for channel '{name}' must be finite, "
                    f"got {value!r}."
                )
        if lo >= hi:
            raise ValueError(
                f"Range for channel '{name}' must satisfy lo < hi, "
                f"got ({lo}, {hi})."
            )
        if name not in window.channels:
            results.append(
                {
                    "channel": name,
                    "check": "range",
                    "ok": False,
                    "message": f"channel '{name}' missing from window.",
                    "fraction_out_of_range": None,
                    "n_out": None,
                    "n_total": None,
                    "n_nan": None,
                    "lo": float(lo),
                    "hi": float(hi),
                }
            )
            continue
        arr = window.channels[name]
        nan_mask = np.isnan(arr)
        valid = arr[~nan_mask]
        n_out = int(np.sum((valid < lo) | (valid > hi)))
        frac = float(n_out / valid.size) if valid.size else 0.0
        results.append(
            {
                "channel": name,
                "check": "range",
                "ok": frac == 0.0,
                "message": "ok"
                if frac == 0.0
                else f"{n_out}/{valid.size} samples out of [{lo}, {hi}].",
                "fraction_out_of_range": frac,
                "n_out": n_out,
                "n_total": int(valid.size),
                "n_nan": int(np.sum(nan_mask)),
                "lo": float(lo),
                "hi": float(hi),
            }
        )
    return results
