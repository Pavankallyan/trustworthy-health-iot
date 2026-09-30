"""Fault injection with logged ground truth for evaluation.

Faults are *plausible*: after injection, touched channels are clipped to a
plausible range (explicit per-channel ranges when given, otherwise the
channel's observed min/max in the source window), so injections pass plain
range checks and only the statistical detectors should fire.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .stream import Window, validate_window

#: Fault types supported by :class:`FaultInjector`.
FAULT_TYPES = (
    "gradual_drift",
    "stuck_at",
    "random_dropout",
    "quantization_noise",
    "desync",
)


@dataclass
class InjectionResult:
    """Result of :meth:`FaultInjector.inject`.

    Supports tuple unpacking: ``new_window, ground_truth = inject(...)``.
    """

    window: Window
    ground_truth: list[dict]

    def __iter__(self):
        yield self.window
        yield self.ground_truth


class FaultInjector:
    """Inject plausible sensor faults into a :class:`Window`.

    Args:
        seed: Seed for the internal RNG. Pass an int for deterministic,
            reproducible injections; ``None`` uses fresh entropy.
        plausible_ranges: Optional channel -> ``(lo, hi)`` ranges used to clip
            injected values. When absent for a channel, the channel's observed
            min/max in the source window is used.

    Example:
        >>> injector = FaultInjector(seed=7)
        >>> new_window, truth = injector.inject(window, [
        ...     {"type": "stuck_at", "channel": "ecg",
        ...      "start": 2.0, "end": 8.0, "params": {}},
        ... ])
    """

    def __init__(
        self,
        seed: int | None = None,
        plausible_ranges: dict[str, tuple[float, float]] | None = None,
    ):
        if seed is not None and not isinstance(seed, (int, np.integer)):
            raise TypeError(f"seed must be an int or None, got {seed!r}.")
        self.seed = None if seed is None else int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.plausible_ranges: dict[str, tuple[float, float]] = {}
        if plausible_ranges is not None:
            if not isinstance(plausible_ranges, dict):
                raise TypeError("plausible_ranges must be a dict or None.")
            for ch, bounds in plausible_ranges.items():
                if (
                    not isinstance(bounds, tuple)
                    or len(bounds) != 2
                    or not all(
                        isinstance(v, (int, float)) and np.isfinite(v)
                        for v in bounds
                    )
                    or bounds[0] >= bounds[1]
                ):
                    raise ValueError(
                        f"plausible_ranges['{ch}'] must be a (lo, hi) tuple "
                        f"with lo < hi, got {bounds!r}."
                    )
                self.plausible_ranges[ch] = (float(bounds[0]), float(bounds[1]))

    def inject(self, window: Window, faults: list[dict]) -> InjectionResult:
        """Apply faults to a copy of ``window`` and log ground truth.

        Each fault dict: ``{"type": one of FAULT_TYPES, "channel": str,
        "start": float, "end": float, "params": dict}`` where ``start``/``end``
        are seconds relative to the window start.

        Fault semantics (``params``):

        - ``gradual_drift``: adds a linear ramp ``0 -> magnitude`` over the
          fault span. ``params["magnitude"]`` (default: 2 × channel std).
        - ``stuck_at``: freezes the span to ``params["value"]`` (default: the
          last good sample before ``start``, else the channel median).
        - ``random_dropout``: sets a random ``params["fraction"]`` (default
          0.3) of the span's samples to NaN, drawn with the injector RNG.
        - ``quantization_noise``: rounds the span to steps of
          ``params["step"]`` (default: span range / ``params["n_levels"]``,
          ``n_levels`` default 8).
        - ``desync``: time-shifts the span by ``params["shift_s"]`` seconds
          (default 1.0); vacated edge samples are filled with the nearest
          edge value.

        Returns:
            :class:`InjectionResult` with the corrupted window and
            ``ground_truth``: one dict per fault with ``type``, ``channel``,
            ``start_s``/``end_s``, ``start_sample``/``end_sample`` (sample
            indices into the channel array), and ``params``.

        Raises:
            TypeError/ValueError: On bad input.
        """
        validate_window(window)
        if not isinstance(faults, list) or not faults:
            raise ValueError("faults must be a non-empty list of fault dicts.")

        duration = window.duration_s
        channels = {name: arr.copy() for name, arr in window.channels.items()}
        ground_truth: list[dict] = []

        for i, fault in enumerate(faults):
            ftype, channel, s0, s1, params = self._validate_fault(
                fault, i, window, duration
            )
            x = channels[channel]
            if ftype == "gradual_drift":
                self._apply_gradual_drift(x, s0, s1, params)
            elif ftype == "stuck_at":
                self._apply_stuck_at(x, s0, s1, params)
            elif ftype == "random_dropout":
                self._apply_random_dropout(x, s0, s1, params)
            elif ftype == "quantization_noise":
                self._apply_quantization_noise(x, s0, s1, params)
            elif ftype == "desync":
                self._apply_desync(x, s0, s1, params, window.fs[channel])

            ground_truth.append(
                {
                    "type": ftype,
                    "channel": channel,
                    "start_s": float(fault["start"]),
                    "end_s": float(fault["end"]),
                    "start_sample": int(s0),
                    "end_sample": int(s1),
                    "params": dict(params),
                }
            )

        # Plausibility clip: keep every touched channel inside its plausible
        # range so injections pass plain range checks.
        touched = {g["channel"] for g in ground_truth}
        for name in touched:
            original = window.channels[name]
            valid = original[~np.isnan(original)]
            if valid.size == 0:
                continue
            lo, hi = self.plausible_ranges.get(
                name, (float(np.min(valid)), float(np.max(valid)))
            )
            channels[name] = np.clip(channels[name], lo, hi)

        new_window = Window(
            device_id=window.device_id,
            start_s=window.start_s,
            end_s=window.end_s,
            channels=channels,
            fs=dict(window.fs),
        )
        return InjectionResult(window=new_window, ground_truth=ground_truth)

    # -- validation -----------------------------------------------------

    def _validate_fault(self, fault: dict, i: int, window: Window, duration: float):
        if not isinstance(fault, dict):
            raise TypeError(f"faults[{i}] must be a dict, got {type(fault).__name__}.")
        for key in ("type", "channel", "start", "end"):
            if key not in fault:
                raise ValueError(f"faults[{i}] is missing required key '{key}'.")
        ftype = fault["type"]
        if ftype not in FAULT_TYPES:
            raise ValueError(
                f"faults[{i}]['type'] must be one of {FAULT_TYPES}, got {ftype!r}."
            )
        channel = fault["channel"]
        if channel not in window.channels:
            raise ValueError(
                f"faults[{i}]['channel'] '{channel}' is not in the window."
            )
        start, end = fault["start"], fault["end"]
        for value, label in ((start, "start"), (end, "end")):
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(
                    f"faults[{i}]['{label}'] must be a finite number, got {value!r}."
                )
        if not (0.0 <= start < end <= duration):
            raise ValueError(
                f"faults[{i}] must satisfy 0 <= start < end <= window duration "
                f"({duration}s), got start={start}, end={end}."
            )
        params = fault.get("params", {})
        if not isinstance(params, dict):
            raise TypeError(f"faults[{i}]['params'] must be a dict.")
        fs = window.fs[channel]
        s0 = int(start * fs)
        s1 = min(int(end * fs), window.channels[channel].size)
        if s1 <= s0:
            raise ValueError(
                f"faults[{i}] covers no samples on channel '{channel}' "
                f"(fs={fs} Hz)."
            )
        return ftype, channel, s0, s1, params

    # -- fault implementations ------------------------------------------

    def _apply_gradual_drift(self, x: np.ndarray, s0: int, s1: int, params: dict):
        valid = x[~np.isnan(x)]
        default_mag = 2.0 * float(np.std(valid)) if valid.size else 1.0
        magnitude = params.get("magnitude", default_mag)
        if not isinstance(magnitude, (int, float)) or not np.isfinite(magnitude):
            raise ValueError(
                f"gradual_drift params['magnitude'] must be finite, got {magnitude!r}."
            )
        ramp = np.linspace(0.0, float(magnitude), s1 - s0)
        x[s0:s1] = np.where(np.isnan(x[s0:s1]), np.nan, x[s0:s1] + ramp)

    def _apply_stuck_at(self, x: np.ndarray, s0: int, s1: int, params: dict):
        value = params.get("value", None)
        if value is None:
            if s0 > 0 and not np.isnan(x[s0 - 1]):
                value = float(x[s0 - 1])
            else:
                valid = x[~np.isnan(x)]
                if valid.size == 0:
                    raise ValueError("stuck_at needs a value: channel is all-NaN.")
                value = float(np.median(valid))
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(
                f"stuck_at params['value'] must be finite, got {value!r}."
            )
        x[s0:s1] = float(value)

    def _apply_random_dropout(self, x: np.ndarray, s0: int, s1: int, params: dict):
        fraction = params.get("fraction", 0.3)
        if (
            not isinstance(fraction, (int, float))
            or not np.isfinite(fraction)
            or not 0.0 < fraction < 1.0
        ):
            raise ValueError(
                f"random_dropout params['fraction'] must be in (0, 1), "
                f"got {fraction!r}."
            )
        k = max(1, int(round((s1 - s0) * float(fraction))))
        idx = self.rng.choice(np.arange(s0, s1), size=min(k, s1 - s0), replace=False)
        x[idx] = np.nan

    def _apply_quantization_noise(self, x: np.ndarray, s0: int, s1: int, params: dict):
        seg = x[s0:s1]
        step = params.get("step", None)
        if step is None:
            n_levels = params.get("n_levels", 8)
            if not isinstance(n_levels, int) or n_levels < 2:
                raise ValueError(
                    f"quantization_noise params['n_levels'] must be an int >= 2, "
                    f"got {n_levels!r}."
                )
            valid = seg[~np.isnan(seg)]
            span = float(np.max(valid) - np.min(valid)) if valid.size else 0.0
            step = span / n_levels if span > 0 else 1.0
        if not isinstance(step, (int, float)) or not np.isfinite(step) or step <= 0:
            raise ValueError(
                f"quantization_noise params['step'] must be positive, got {step!r}."
            )
        q = np.round(seg / step) * step
        x[s0:s1] = np.where(np.isnan(seg), np.nan, q)

    def _apply_desync(self, x: np.ndarray, s0: int, s1: int, params: dict, fs: float):
        shift_s = params.get("shift_s", 1.0)
        if not isinstance(shift_s, (int, float)) or not np.isfinite(shift_s):
            raise ValueError(
                f"desync params['shift_s'] must be finite, got {shift_s!r}."
            )
        shift = int(round(float(shift_s) * fs))
        if shift == 0:
            raise ValueError(
                f"desync shift_s={shift_s}s is less than one sample at fs={fs} Hz."
            )
        if abs(shift) >= (s1 - s0):
            raise ValueError(
                f"desync shift of {shift} samples covers the whole fault span."
            )
        seg = x[s0:s1].copy()
        rolled = np.roll(seg, shift)
        if shift > 0:
            rolled[:shift] = seg[0]
        else:
            rolled[shift:] = seg[-1]
        # Preserve NaNs that were already missing (don't fabricate data).
        rolled[np.isnan(seg)] = np.nan
        x[s0:s1] = rolled
