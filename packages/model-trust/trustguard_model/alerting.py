"""Threshold-plus-hysteresis alerting primitive.

A raw threshold flips the instant a noisy signal crosses it, which makes
deployment monitors flap. :class:`DegradationAlert` adds hysteresis: once a
signal breaches, it must stay clear for ``clear_streak_required`` (K)
consecutive updates before the alert un-breaches.  An optional ``margin``
widens the clear band so values hovering at the threshold do not flap.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DegradationAlert", "DEFAULT_CLEAR_STREAK"]

#: Default K: a breached signal must clear for this many consecutive updates
#: before it is considered healthy again.
DEFAULT_CLEAR_STREAK = 3


@dataclass
class AlertSnapshot:
    """Small, unambiguous snapshot of an alert's state."""

    name: str
    breached: bool
    clear_streak: int
    clear_streak_required: int


class DegradationAlert:
    """Stateful threshold alert with hysteresis on the clear path.

    Parameters
    ----------
    name:
        Human-readable signal name (used in snapshots and error messages).
    threshold:
        Breach boundary.  Must be a finite float.
    direction:
        ``"above"`` breaches when ``value > threshold`` (e.g. PSI, KS);
        ``"below"`` breaches when ``value < threshold`` (e.g. accuracy,
        agreement fraction).
    clear_streak_required:
        K — number of consecutive *clear* updates required to un-breach.
        Must be a positive int.  Default 3.
    margin:
        Hysteresis band width (non-negative float).  With ``direction="above"``
        a breached alert only counts an update as clear when
        ``value <= threshold - margin``; with ``direction="below"`` when
        ``value >= threshold + margin``.  Values inside the band neither
        breach nor advance the clear streak (the previous state holds).
    """

    def __init__(
        self,
        name: str,
        threshold: float,
        direction: str = "above",
        clear_streak_required: int = DEFAULT_CLEAR_STREAK,
        margin: float = 0.0,
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            raise TypeError("threshold must be a finite number")
        import math

        if not math.isfinite(float(threshold)):
            raise ValueError("threshold must be finite")
        if direction not in ("above", "below"):
            raise ValueError("direction must be 'above' or 'below'")
        if (
            not isinstance(clear_streak_required, int)
            or isinstance(clear_streak_required, bool)
            or clear_streak_required < 1
        ):
            raise ValueError("clear_streak_required (K) must be a positive int")
        if not isinstance(margin, (int, float)) or isinstance(margin, bool):
            raise TypeError("margin must be a non-negative number")
        if margin < 0 or not math.isfinite(float(margin)):
            raise ValueError("margin must be a finite non-negative number")

        self.name = name
        self.threshold = float(threshold)
        self.direction = direction
        self.clear_streak_required = clear_streak_required
        self.margin = float(margin)
        self._breached = False
        self._clear_streak = 0

    @property
    def breached(self) -> bool:
        """Current alert state (True = degraded)."""
        return self._breached

    @property
    def clear_streak(self) -> int:
        """Consecutive clear updates seen while breached."""
        return self._clear_streak

    def _classify(self, value: float) -> str:
        """Return 'breach', 'clear', or 'hold' for one observation."""
        if self.direction == "above":
            if value > self.threshold:
                return "breach"
            if value <= self.threshold - self.margin:
                return "clear"
            return "hold"
        if value < self.threshold:
            return "breach"
        if value >= self.threshold + self.margin:
            return "clear"
        return "hold"

    def update(self, value: float) -> bool:
        """Feed one observation; return the (possibly changed) breach state.

        * Not breached + ``"breach"`` observation -> breached immediately.
        * Breached + ``"clear"`` observation -> clear streak grows; un-breach
          only when the streak reaches ``clear_streak_required`` (K).
        * ``"hold"`` observations (inside the hysteresis band) reset the clear
          streak but do not change the breach state.
        """
        import math

        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(
                f"alert '{self.name}': value must be a number, got {type(value).__name__}"
            )
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"alert '{self.name}': value must be finite, got {value}")

        outcome = self._classify(value)
        if not self._breached:
            if outcome == "breach":
                self._breached = True
                self._clear_streak = 0
        else:
            if outcome == "clear":
                self._clear_streak += 1
                if self._clear_streak >= self.clear_streak_required:
                    self._breached = False
                    self._clear_streak = 0
            else:
                # re-breach or band-hover: streak restarts
                self._clear_streak = 0
        return self._breached

    def reset(self) -> None:
        """Return to the healthy state, forgetting breach history."""
        self._breached = False
        self._clear_streak = 0

    def snapshot(self) -> AlertSnapshot:
        """Return a small dataclass describing current state."""
        return AlertSnapshot(
            name=self.name,
            breached=self._breached,
            clear_streak=self._clear_streak,
            clear_streak_required=self.clear_streak_required,
        )
