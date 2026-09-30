"""Streaming primitives for TrustGuard-IoT Layer 1 (data trust).

Layer 1 owns these core types: the :class:`Window` dataclass plus helpers to
validate windows and slice multi-channel streams into windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Window:
    """One fixed-duration slice of synchronized sensor channels.

    Attributes:
        device_id: Identifier of the device that produced the data.
        start_s: Window start time in seconds.
        end_s: Window end time in seconds (must be > ``start_s``).
        channels: Mapping of channel name -> 1D float64 array. NaN marks a
            missing sample.
        fs: Mapping of channel name -> sampling rate in Hz. Keys must match
            ``channels`` exactly.
    """

    device_id: str
    start_s: float
    end_s: float
    channels: dict[str, np.ndarray] = field(default_factory=dict)
    fs: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Normalize every channel to a 1D float64 array, then validate.
        normalized: dict[str, np.ndarray] = {}
        for name, arr in self.channels.items():
            a = np.asarray(arr, dtype=np.float64)
            if a.ndim != 1:
                raise ValueError(
                    f"Window channel '{name}' must be a 1D array, got shape {a.shape}."
                )
            normalized[name] = a
        self.channels = normalized
        validate_window(self)

    @property
    def duration_s(self) -> float:
        """Window duration in seconds."""
        return self.end_s - self.start_s


def validate_window(window: "Window") -> None:
    """Validate a :class:`Window` against the Layer 1 contract.

    Raises:
        TypeError: If the window or its fields have the wrong types.
        ValueError: If any contract rule is violated (empty id, end <= start,
            empty channels, fs keys not matching channel keys, non-positive
            sampling rates, non-1D or empty channel arrays).
    """
    if not isinstance(window, Window):
        raise TypeError(f"Expected a Window, got {type(window).__name__}.")
    if not isinstance(window.device_id, str) or not window.device_id:
        raise ValueError("Window.device_id must be a non-empty string.")
    if not isinstance(window.start_s, (int, float)) or not isinstance(
        window.end_s, (int, float)
    ):
        raise TypeError("Window start_s/end_s must be numbers.")
    if not np.isfinite(window.start_s) or not np.isfinite(window.end_s):
        raise ValueError("Window start_s/end_s must be finite.")
    if window.end_s <= window.start_s:
        raise ValueError(
            f"Window end_s ({window.end_s}) must be greater than "
            f"start_s ({window.start_s})."
        )
    if not isinstance(window.channels, dict) or not window.channels:
        raise ValueError("Window.channels must be a non-empty dict.")
    if not isinstance(window.fs, dict):
        raise TypeError("Window.fs must be a dict.")
    if set(window.fs.keys()) != set(window.channels.keys()):
        raise ValueError(
            "Window.fs keys must match Window.channels keys exactly. "
            f"channels={sorted(window.channels)}, fs={sorted(window.fs)}."
        )
    for name, rate in window.fs.items():
        if (
            not isinstance(rate, (int, float))
            or not np.isfinite(rate)
            or rate <= 0
        ):
            raise ValueError(
                f"Sampling rate for channel '{name}' must be a positive number, "
                f"got {rate!r}."
            )
    for name, arr in window.channels.items():
        if not isinstance(arr, np.ndarray) or arr.ndim != 1:
            raise ValueError(f"Channel '{name}' must be a 1D numpy array.")
        if arr.size == 0:
            raise ValueError(f"Channel '{name}' must not be empty.")


def split_stream(
    device_id: str,
    channels: dict[str, np.ndarray],
    fs: dict[str, float],
    window_s: float,
    start_s: float = 0.0,
) -> list[Window]:
    """Slice synchronized channels into consecutive non-overlapping windows.

    Channels may have different sampling rates; every window covers the same
    time span on each channel. Trailing samples that do not fill a full window
    are dropped (documented, deterministic).

    Args:
        device_id: Device identifier stamped on every window.
        channels: Channel name -> 1D array-like of samples.
        fs: Channel name -> sampling rate in Hz (keys must match ``channels``).
        window_s: Window length in seconds.
        start_s: Timestamp of the first window's start.

    Returns:
        List of :class:`Window`, in time order.

    Raises:
        TypeError/ValueError: On bad input, or when no channel holds enough
            samples for a single full window.
    """
    if not isinstance(device_id, str) or not device_id:
        raise ValueError("device_id must be a non-empty string.")
    if not isinstance(channels, dict) or not channels:
        raise ValueError("channels must be a non-empty dict.")
    if not isinstance(fs, dict) or set(fs.keys()) != set(channels.keys()):
        raise ValueError("fs must be a dict with exactly the same keys as channels.")
    if (
        not isinstance(window_s, (int, float))
        or not np.isfinite(window_s)
        or window_s <= 0
    ):
        raise ValueError(f"window_s must be a positive number, got {window_s!r}.")
    if not isinstance(start_s, (int, float)) or not np.isfinite(start_s):
        raise ValueError(f"start_s must be a finite number, got {start_s!r}.")

    arrays: dict[str, np.ndarray] = {}
    for name, arr in channels.items():
        a = np.asarray(arr, dtype=np.float64)
        if a.ndim != 1:
            raise ValueError(
                f"Channel '{name}' must be 1D, got shape {a.shape}."
            )
        rate = fs[name]
        if (
            not isinstance(rate, (int, float))
            or not np.isfinite(rate)
            or rate <= 0
        ):
            raise ValueError(
                f"Sampling rate for channel '{name}' must be a positive number, "
                f"got {rate!r}."
            )
        arrays[name] = a

    per_window = {name: int(round(window_s * fs[name])) for name in arrays}
    for name, n in per_window.items():
        if n < 1:
            raise ValueError(
                f"window_s={window_s}s is shorter than one sample for channel "
                f"'{name}' (fs={fs[name]} Hz)."
            )
    n_windows = min(len(arrays[name]) // per_window[name] for name in arrays)
    if n_windows < 1:
        raise ValueError(
            "No channel holds enough samples for a single full window."
        )

    windows: list[Window] = []
    for i in range(n_windows):
        w_start = start_s + i * window_s
        w_end = w_start + window_s
        w_channels = {
            name: arrays[name][i * per_window[name] : (i + 1) * per_window[name]].copy()
            for name in arrays
        }
        windows.append(
            Window(
                device_id=device_id,
                start_s=w_start,
                end_s=w_end,
                channels=w_channels,
                fs=dict(fs),
            )
        )
    return windows
