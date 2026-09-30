"""Tests for trustguard_fusion.fusion."""

import numpy as np
import pytest

from trustguard_data.taxonomy import (
    ChannelHealth,
    HealthState,
    Layer1Result,
)
from trustguard_fusion.fusion import TrustReport, fuse_trust
from trustguard_model.monitors import Layer2Result, ModelSignal


def _ch(name, state, score):
    return ChannelHealth(channel=name, state=state, score=score, details={})


def _l1(scores=(100.0, 100.0)):
    channels = {
        "ecg": _ch("ecg", HealthState.HEALTHY, scores[0]),
        "bvp": _ch("bvp", HealthState.HEALTHY, scores[1]),
    }
    return Layer1Result(device_id="dev1", start_s=0.0, end_s=60.0,
                        channels=channels,
                        layer_score=float(np.mean(scores)))


def _l2(health=100.0, degraded=False, breached=()):
    signals = {}
    for name in ("prediction_psi", "mean_confidence"):
        signals[name] = ModelSignal(name=name, value=0.0, threshold=1.0,
                                    breached=name in breached, details={})
    return Layer2Result(model_id="m", n_windows=10, signals=signals,
                        degraded=degraded, health=health)


def test_perfect_health_gives_100():
    r = fuse_trust("dev1", _l1(), _l2())
    assert r.trust_score == 100.0
    assert r.layer2_score == 100.0
    assert r.worst_channels == []


def test_layer1_penalty_math():
    # one channel at 50/100, default weights 0.6/0.4, l2 perfect:
    # penalty = -(100-50) * 0.6/2 = -15 -> trust 85
    r = fuse_trust("dev1", _l1(scores=(50.0, 100.0)), _l2())
    assert r.trust_score == pytest.approx(85.0)
    assert any(k.startswith("layer1:ecg") for k in r.contributions)
    assert r.worst_channels[0] == "ecg"


def test_layer2_none_renormalizes():
    r = fuse_trust("dev1", _l1(scores=(50.0, 100.0)), None)
    assert r.layer2_score is None
    # full weight on layer 1: -(50)*1.0/2 = -25 -> 75
    assert r.trust_score == pytest.approx(75.0)


def test_layer2_penalty():
    r = fuse_trust("dev1", _l1(), _l2(health=50.0, degraded=True,
                                      breached=("prediction_psi",)))
    # -(100-50)*0.4 = -20 -> 80
    assert r.trust_score == pytest.approx(80.0)
    assert sum(v for v in r.contributions.values()) == pytest.approx(-20.0)


def test_clamped_at_zero():
    bad = {n: _ch(n, HealthState.STUCK, 0.0) for n in ("a", "b")}
    l1 = Layer1Result(device_id="d", start_s=0, end_s=1, channels=bad,
                      layer_score=0.0)
    r = fuse_trust("d", l1, _l2(health=0.0, degraded=True))
    assert r.trust_score == 0.0


def test_custom_weights():
    r = fuse_trust("dev1", _l1(scores=(0.0, 100.0)), _l2(),
                   weights={"layer1": 1.0, "layer2": 3.0})
    # renormalized to 0.25/0.75: -(100)*0.25/2 = -12.5 -> 87.5
    assert r.trust_score == pytest.approx(87.5)


def test_contributions_sum_to_penalty():
    r = fuse_trust("dev1", _l1(scores=(70.0, 90.0)),
                   _l2(health=80.0, degraded=True))
    assert 100.0 + sum(r.contributions.values()) == pytest.approx(r.trust_score)
    assert all(v <= 0 for v in r.contributions.values())


def test_input_validation():
    l1, l2 = _l1(), _l2()
    with pytest.raises(ValueError):
        fuse_trust("", l1, l2)
    with pytest.raises(TypeError):
        fuse_trust("d", "not-a-result", l2)
    with pytest.raises(TypeError):
        fuse_trust("d", l1, "not-a-result")
    with pytest.raises(ValueError):
        fuse_trust("d", l1, l2, weights={"layer1": -1.0})
    with pytest.raises(ValueError):
        fuse_trust("d", l1, l2, weights={"layer1": 0.0, "layer2": 0.0})
    with pytest.raises(ValueError):
        fuse_trust("d", l1, l2, timestamp_s=-1.0)
