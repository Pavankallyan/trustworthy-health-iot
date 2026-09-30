"""Severity-tiered alert routing for TrustGuard-IoT.

Alerts are deterministic: same report in, same alerts out. The dispatcher
adds cooldown-based dedup and a JSONL audit log so every alert is traceable.
"""

from __future__ import annotations

import enum
import json
import os
import time
from dataclasses import dataclass, field

from trustguard_data.taxonomy import HealthState, Layer1Result
from trustguard_fusion.fusion import TrustReport

try:
    from trustguard_model.monitors import Layer2Result
except Exception:  # pragma: no cover - keeps fusion importable standalone
    Layer2Result = None


class Severity(enum.Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    device_id: str
    severity: Severity
    title: str
    detail: str
    source: str            # "layer1" | "layer2" | "fusion"
    timestamp_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "device_id": self.device_id,
            "severity": self.severity.value,
            "title": self.title,
            "detail": self.detail,
            "source": self.source,
            "timestamp_s": self.timestamp_s,
        }


# States that always deserve at least a warning on their own.
_ALERT_STATES = {
    HealthState.STUCK: Severity.WARNING,
    HealthState.DROPOUT: Severity.WARNING,
    HealthState.MISSING: Severity.WARNING,
    HealthState.DRIFTING: Severity.INFO,
    HealthState.DEGRADED: Severity.INFO,
}

CRITICAL_TRUST = 40.0
WARNING_TRUST = 70.0
MAX_CHANNEL_ALERTS = 5


def route_alerts(
    report: TrustReport,
    l1: Layer1Result | None = None,
    l2: object | None = None,
) -> list[Alert]:
    """Build the deterministic alert list for one trust report."""
    if not isinstance(report, TrustReport):
        raise TypeError("report must be a TrustReport")
    ts = report.timestamp_s
    alerts: list[Alert] = []

    # Fusion-level trust tiers.
    if report.trust_score < CRITICAL_TRUST:
        alerts.append(Alert(report.device_id, Severity.CRITICAL,
                            "Device trust critical",
                            f"Trust score {report.trust_score:.1f} is below "
                            f"{CRITICAL_TRUST:.0f}. Immediate review recommended.",
                            "fusion", ts))
    elif report.trust_score < WARNING_TRUST:
        alerts.append(Alert(report.device_id, Severity.WARNING,
                            "Device trust degraded",
                            f"Trust score {report.trust_score:.1f} is below "
                            f"{WARNING_TRUST:.0f}.",
                            "fusion", ts))

    # Layer 1 channel states.
    if l1 is not None:
        if not isinstance(l1, Layer1Result):
            raise TypeError("l1 must be a Layer1Result or None")
        n = 0
        for name, ch in sorted(l1.channels.items(), key=lambda kv: kv[1].score):
            sev = _ALERT_STATES.get(ch.state)
            if sev is None or n >= MAX_CHANNEL_ALERTS:
                continue
            n += 1
            alerts.append(Alert(
                report.device_id, sev,
                f"Sensor {name}: {ch.state.value}",
                f"Channel health {ch.score:.1f}/100. "
                + str(ch.details.get("reason", "")),
                "layer1", ts))

    # Layer 2 model degradation.
    if l2 is not None and getattr(l2, "degraded", False):
        breached = [s for s, sig in l2.signals.items()
                    if getattr(sig, "breached", False)]
        detail = ("Breached signals: " + ", ".join(breached)) if breached \
            else "Model health below limit."
        alerts.append(Alert(report.device_id, Severity.WARNING,
                            "Model degradation detected", detail,
                            "layer2", ts))

    order = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2}
    alerts.sort(key=lambda a: (order[a.severity], a.title))
    return alerts


class AlertDispatcher:
    """Collect alerts with cooldown dedup and a JSONL audit log."""

    def __init__(self, log_path: str | None = None, cooldown_s: float = 3600.0):
        if cooldown_s < 0:
            raise ValueError("cooldown_s must be non-negative")
        self.log_path = log_path
        self.cooldown_s = float(cooldown_s)
        self._last_seen: dict[tuple[str, str], float] = {}
        self.dispatched: list[Alert] = []

    def dispatch(self, alerts: list[Alert], now_s: float | None = None) -> list[Alert]:
        """Return the alerts that are new (outside cooldown); log them."""
        if now_s is None:
            now_s = time.time()
        fresh: list[Alert] = []
        for a in alerts:
            if not isinstance(a, Alert):
                raise TypeError("alerts must contain Alert objects")
            key = (a.device_id, a.title)
            last = self._last_seen.get(key)
            if last is not None and (now_s - last) < self.cooldown_s:
                continue
            self._last_seen[key] = now_s
            fresh.append(a)
            self.dispatched.append(a)
        if fresh and self.log_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                for a in fresh:
                    f.write(json.dumps(a.to_dict()) + "\n")
        return fresh
