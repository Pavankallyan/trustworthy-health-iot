"""Layer 1 pipeline: baseline fitting and per-window scoring."""

from __future__ import annotations

import numpy as np

from .consistency import acc_consistency, hr_agreement
from .drift import ks_drift, psi, trend_slope
from .dropout import detect_dropout
from .stream import Window, validate_window
from .stuck import detect_stuck
from .taxonomy import (
    ChannelHealth,
    HealthState,  # noqa: F401  (re-exported for convenience)
    Layer1Result,
    classify_channel,
    DEFAULT_THRESHOLDS,
)
from .validators import check_range, check_schema


class DataTrustPipeline:
    """Score windows of health-IoT sensor data for silent corruption.

    The pipeline runs every detector on every channel, adds cross-channel
    consistency checks, and maps the detector outputs to
    :class:`~trustguard_data.taxonomy.HealthState` via
    :func:`~trustguard_data.taxonomy.classify_channel`.

    Config keys (all optional):

    - ``ranges``: channel -> ``(lo, hi)`` plausible range.
    - ``schema``: channel -> ``(dtype_kind, min_len)``.
    - ``acc_groups``: group name -> list of channel names forming one
      accelerometer (e.g. ``{"chest": ["acc_x", "acc_y", "acc_z"]}``).
    - ``acc_pairs``: list of ``(group_a, group_b)`` tuples to cross-check.
    - ``hr_pair``: ``(channel_a, channel_b)`` whose peak-based heart rates
      must agree (e.g. ``("ecg", "bvp")``).
    - ``thresholds``: overrides for
      :data:`~trustguard_data.taxonomy.DEFAULT_THRESHOLDS`.
    - ``desync_lag_s``: lag threshold for the accelerometer desync flag.
    - ``baseline_max_samples``: cap on stored baseline samples per channel.
    - ``baseline_min_samples``: minimum valid samples needed for drift
      scoring and baseline fitting.

    Baselines are optional: without :meth:`fit_baseline`, KS/PSI drift checks
    are skipped and classification relies on the remaining detectors.
    """

    def __init__(self, config: dict | None = None):
        if config is None:
            config = {}
        if not isinstance(config, dict):
            raise TypeError(
                f"config must be a dict or None, got {type(config).__name__}."
            )
        self.ranges = self._validate_ranges(config.get("ranges", {}))
        self.schema = self._validate_schema(config.get("schema", {}))
        self.acc_groups = self._validate_acc_groups(config.get("acc_groups", {}))
        self.acc_pairs = self._validate_acc_pairs(config.get("acc_pairs", []))
        self.hr_pair = self._validate_hr_pair(config.get("hr_pair"))
        self.thresholds = self._validate_thresholds(config.get("thresholds"))
        self.desync_lag_s = self._validate_desync_lag(config.get("desync_lag_s", 0.5))
        self.baseline_max_samples = self._validate_positive_int(
            config.get("baseline_max_samples", 10000), "baseline_max_samples"
        )
        self.baseline_min_samples = self._validate_positive_int(
            config.get("baseline_min_samples", 30), "baseline_min_samples"
        )
        self._baselines: dict[str, np.ndarray] = {}
        self._fitted = False

    # -- config validation ------------------------------------------------

    @staticmethod
    def _validate_ranges(ranges) -> dict:
        if not isinstance(ranges, dict):
            raise TypeError("config['ranges'] must be a dict.")
        for ch, bounds in ranges.items():
            if (
                not isinstance(bounds, tuple)
                or len(bounds) != 2
                or not all(
                    isinstance(v, (int, float)) and np.isfinite(v) for v in bounds
                )
                or bounds[0] >= bounds[1]
            ):
                raise ValueError(
                    f"config['ranges']['{ch}'] must be a (lo, hi) tuple with "
                    f"lo < hi, got {bounds!r}."
                )
        return dict(ranges)

    @staticmethod
    def _validate_schema(schema) -> dict:
        if not isinstance(schema, dict):
            raise TypeError("config['schema'] must be a dict.")
        for ch, spec in schema.items():
            if (
                not isinstance(spec, tuple)
                or len(spec) != 2
                or not isinstance(spec[0], str)
                or not isinstance(spec[1], int)
            ):
                raise ValueError(
                    f"config['schema']['{ch}'] must be a (dtype_kind, min_len) "
                    f"tuple, got {spec!r}."
                )
        return dict(schema)

    @staticmethod
    def _validate_acc_groups(groups) -> dict:
        if not isinstance(groups, dict):
            raise TypeError("config['acc_groups'] must be a dict.")
        for g, axes in groups.items():
            if (
                not isinstance(axes, (list, tuple))
                or not axes
                or not all(isinstance(a, str) for a in axes)
            ):
                raise ValueError(
                    f"config['acc_groups']['{g}'] must be a non-empty list of "
                    f"channel names, got {axes!r}."
                )
        return {g: list(a) for g, a in groups.items()}

    @staticmethod
    def _validate_acc_pairs(pairs) -> list:
        if not isinstance(pairs, (list, tuple)):
            raise TypeError("config['acc_pairs'] must be a list of (a, b) tuples.")
        out = []
        for p in pairs:
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                raise ValueError(
                    f"config['acc_pairs'] entries must be (group_a, group_b) "
                    f"tuples, got {p!r}."
                )
            out.append((p[0], p[1]))
        return out

    @staticmethod
    def _validate_hr_pair(pair):
        if pair is None:
            return None
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(
                f"config['hr_pair'] must be a (channel_a, channel_b) tuple or "
                f"None, got {pair!r}."
            )
        return (pair[0], pair[1])

    @staticmethod
    def _validate_thresholds(thresholds) -> dict:
        if thresholds is None:
            return {}
        if not isinstance(thresholds, dict):
            raise TypeError("config['thresholds'] must be a dict or None.")
        unknown = set(thresholds) - set(DEFAULT_THRESHOLDS)
        if unknown:
            raise ValueError(
                f"Unknown threshold keys: {sorted(unknown)}. Valid keys: "
                f"{sorted(DEFAULT_THRESHOLDS)}."
            )
        return dict(thresholds)

    @staticmethod
    def _validate_desync_lag(value) -> float:
        if (
            not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError(
                f"config['desync_lag_s'] must be a non-negative number, "
                f"got {value!r}."
            )
        return float(value)

    @staticmethod
    def _validate_positive_int(value, label: str) -> int:
        if not isinstance(value, int) or value < 1:
            raise ValueError(
                f"config['{label}'] must be a positive int, got {value!r}."
            )
        return value

    # -- state ------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit_baseline` has been called successfully."""
        return self._fitted

    @property
    def baseline_channels(self) -> list[str]:
        """Channel names that have a fitted baseline."""
        return sorted(self._baselines)

    def _merged_thresholds(self) -> dict:
        thr = dict(DEFAULT_THRESHOLDS)
        thr.update(self.thresholds)
        return thr

    # -- baseline fitting -------------------------------------------------

    def fit_baseline(self, windows: list[Window], seed: int | None = None) -> None:
        """Fit per-channel baselines from CLEAN windows.

        Windows with > 5% missing samples on a channel are skipped for that
        channel (documented safeguard — callers should still pass clean
        windows). Per channel, valid samples are pooled and deterministically
        subsampled to ``baseline_max_samples`` with ``seed``.

        Raises:
            TypeError/ValueError: On bad input, or when no channel has enough
                clean data.
        """
        if not isinstance(windows, list) or not windows:
            raise ValueError("windows must be a non-empty list of Window.")
        for w in windows:
            validate_window(w)
        if seed is not None and not isinstance(seed, (int, np.integer)):
            raise TypeError(f"seed must be an int or None, got {seed!r}.")

        rng = np.random.default_rng(seed)
        pooled: dict[str, list[np.ndarray]] = {}
        for w in windows:
            for name, arr in w.channels.items():
                nan = np.isnan(arr)
                if float(np.mean(nan)) > 0.05:
                    continue  # not clean enough for a baseline
                valid = arr[~nan]
                if valid.size:
                    pooled.setdefault(name, []).append(valid)

        baselines: dict[str, np.ndarray] = {}
        for name, parts in pooled.items():
            data = np.concatenate(parts)
            if data.size < self.baseline_min_samples:
                continue
            if data.size > self.baseline_max_samples:
                idx = np.sort(
                    rng.choice(data.size, self.baseline_max_samples, replace=False)
                )
                data = data[idx]
            baselines[name] = data

        if not baselines:
            raise ValueError("No channel had enough clean data to fit a baseline.")
        self._baselines = baselines
        self._fitted = True

    # -- scoring ----------------------------------------------------------

    def score_window(self, window: Window) -> Layer1Result:
        """Score one window; returns a :class:`Layer1Result`.

        Channels are scored in window order, followed by any schema-expected
        channels absent from the window (classified MISSING). Cross-channel
        consistency results are folded into the involved channels before
        classification.
        """
        validate_window(window)
        thr = self._merged_thresholds()

        ordered: list[str] = []
        for name in list(window.channels.keys()) + list(self.schema.keys()):
            if name not in ordered:
                ordered.append(name)

        outputs = {name: self._score_channel(window, name, thr)
                   for name in ordered}
        flags = self._consistency_flags(window, thr)

        channels: dict[str, ChannelHealth] = {}
        for name in ordered:
            out = dict(outputs[name])
            out["consistency_issue"] = bool(flags.get(name, False))
            channels[name] = classify_channel(name, out, thresholds=thr)

        layer_score = float(np.mean([c.score for c in channels.values()]))
        return Layer1Result(
            device_id=window.device_id,
            start_s=window.start_s,
            end_s=window.end_s,
            channels=channels,
            layer_score=layer_score,
        )

    def score_stream(self, windows: list[Window]) -> list[Layer1Result]:
        """Score a list of windows in order; returns one result per window."""
        if not isinstance(windows, list) or not windows:
            raise ValueError("windows must be a non-empty list of Window.")
        return [self.score_window(w) for w in windows]

    # -- internals --------------------------------------------------------

    def _score_channel(self, window: Window, name: str,
                       thr: dict | None = None) -> dict:
        out: dict = {"window_s": window.duration_s}
        if name not in window.channels:
            out["missing"] = True
            return out
        x = window.channels[name]
        fs = window.fs[name]
        valid = x[~np.isnan(x)]
        out["missing"] = bool(valid.size == 0)
        out["std"] = float(np.std(valid)) if valid.size else 0.0
        if out["missing"]:
            return out

        out["schema_ok"] = True
        if name in self.schema:
            schema_res = check_schema(window, {name: self.schema[name]})[0]
            out["schema"] = schema_res
            out["schema_ok"] = schema_res["ok"]
        if name in self.ranges:
            out["range"] = check_range(window, {name: self.ranges[name]})[0]

        out["dropout"] = detect_dropout(x, fs)
        # The stuck flat-run duration is a calibrated threshold: real wearable
        # sensors can hold a constant value for seconds while the wearer is
        # still (ADC quantization), so the minimum stuck duration is tuned on
        # clean validation data instead of using a fixed 5 s.
        _thr = thr if thr is not None else self._merged_thresholds()
        out["stuck"] = detect_stuck(
            x, fs, min_duration_s=_thr["stuck_min_duration_s"])
        out["trend"] = trend_slope(x, fs) if valid.size >= 3 else None

        baseline = self._baselines.get(name)
        if baseline is not None and valid.size >= self.baseline_min_samples:
            out["ks"] = ks_drift(x, baseline)
            out["psi"] = psi(baseline, x)
        else:
            out["ks"] = None
            out["psi"] = None
        return out

    def _consistency_flags(self, window: Window, thr: dict) -> dict[str, bool]:
        flags: dict[str, bool] = {}
        for group_a, group_b in self.acc_pairs:
            axes_a = self.acc_groups.get(group_a, [])
            axes_b = self.acc_groups.get(group_b, [])
            if not axes_a or not axes_b:
                continue
            if any(a not in window.channels for a in axes_a + axes_b):
                continue
            try:
                res = acc_consistency(
                    {a: window.channels[a] for a in axes_a},
                    window.fs[axes_a[0]],
                    {b: window.channels[b] for b in axes_b},
                    window.fs[axes_b[0]],
                    desync_lag_s=self.desync_lag_s,
                )
            except (TypeError, ValueError):
                continue
            bad = bool(
                res["desync"] or res["correlation"] < thr["acc_min_correlation"]
            )
            for a in axes_a + axes_b:
                flags[a] = flags.get(a, False) or bad

        if self.hr_pair is not None:
            ch_a, ch_b = self.hr_pair
            if ch_a in window.channels and ch_b in window.channels:
                try:
                    res = hr_agreement(
                        window.channels[ch_a],
                        window.fs[ch_a],
                        window.channels[ch_b],
                        window.fs[ch_b],
                    )
                except (TypeError, ValueError):
                    res = None
                if res is not None and not res["agreement"]:
                    flags[ch_a] = True
                    flags[ch_b] = True
        return flags
