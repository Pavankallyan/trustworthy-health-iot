"""Tests for trustguard_fusion.alerts and dashboard."""

import pytest

from trustguard_data.taxonomy import (
    ChannelHealth,
    HealthState,
    Layer1Result,
)
from trustguard_fusion.alerts import (
    Alert,
    AlertDispatcher,
    Severity,
    route_alerts,
)
from trustguard_fusion.dashboard import render_html
from trustguard_fusion.fusion import TrustReport, fuse_trust
from trustguard_model.monitors import Layer2Result, ModelSignal


def _report(trust=100.0, l1_score=100.0, l2_score=100.0):
    return TrustReport(device_id="dev1", trust_score=trust,
                       layer1_score=l1_score, layer2_score=l2_score,
                       contributions={}, worst_channels=[], timestamp_s=0.0)


def _l1_with(state, score=20.0):
    ch = ChannelHealth(channel="ecg", state=state, score=score,
                       details={"reason": "test"})
    return Layer1Result(device_id="dev1", start_s=0.0, end_s=60.0,
                        channels={"ecg": ch}, layer_score=score)


def _l2_degraded():
    sig = ModelSignal(name="prediction_psi", value=0.5, threshold=0.2,
                      breached=True, details={})
    return Layer2Result(model_id="m", n_windows=5,
                        signals={"prediction_psi": sig},
                        degraded=True, health=60.0)


def test_no_alerts_when_healthy():
    assert route_alerts(_report()) == []


def test_critical_tier():
    alerts = route_alerts(_report(trust=30.0))
    assert alerts[0].severity is Severity.CRITICAL
    assert "critical" in alerts[0].title.lower()


def test_warning_tier():
    alerts = route_alerts(_report(trust=60.0))
    assert any(a.severity is Severity.WARNING for a in alerts)


def test_channel_state_alerts():
    alerts = route_alerts(_report(trust=90.0), l1=_l1_with(HealthState.STUCK))
    assert any(a.severity is Severity.WARNING and "ecg" in a.title
               for a in alerts)
    alerts = route_alerts(_report(trust=90.0), l1=_l1_with(HealthState.DRIFTING))
    assert any(a.severity is Severity.INFO for a in alerts)


def test_layer2_degradation_alert():
    alerts = route_alerts(_report(trust=90.0), l2=_l2_degraded())
    assert any(a.source == "layer2" and a.severity is Severity.WARNING
               for a in alerts)


def test_alerts_sorted_by_severity():
    alerts = route_alerts(_report(trust=10.0), l1=_l1_with(HealthState.DRIFTING),
                          l2=_l2_degraded())
    sevs = [a.severity for a in alerts]
    assert sevs == sorted(sevs, key=lambda s: ("critical", "warning", "info").index(s.value))


def test_route_alerts_validates():
    with pytest.raises(TypeError):
        route_alerts("nope")


def test_dispatcher_cooldown(tmp_path):
    d = AlertDispatcher(log_path=str(tmp_path / "alerts.jsonl"), cooldown_s=60.0)
    a = Alert("dev1", Severity.WARNING, "t", "d", "fusion", 0.0)
    assert d.dispatch([a], now_s=0.0) == [a]
    assert d.dispatch([a], now_s=10.0) == []       # within cooldown
    assert d.dispatch([a], now_s=61.0) == [a]      # cooldown expired
    lines = (tmp_path / "alerts.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2


def test_dashboard_renders():
    r = fuse_trust("dev1",
                   Layer1Result(device_id="dev1", start_s=0, end_s=60,
                                channels={"ecg": ChannelHealth(
                                    "ecg", HealthState.HEALTHY, 100.0, {})},
                                layer_score=100.0),
                   None)
    alerts = route_alerts(r)
    html_out = render_html([r], alerts, trend={"dev1": [98.0, 99.0, 100.0]})
    assert "<html" in html_out and "dev1" in html_out
    assert "100" in html_out and "<svg" in html_out
    assert "No external" not in html_out  # sanity: no such claim needed


def test_dashboard_requires_reports():
    with pytest.raises(ValueError):
        render_html([])
