"""Unit tests for trustguard_model.monitors."""

import numpy as np
import pytest

from trustguard_model.monitors import (
    DEFAULT_CONFIG,
    Layer2Result,
    ModelSignal,
    ModelTrustMonitor,
)


def ref_proba_1d(seed=0, n=500):
    return np.random.default_rng(seed).beta(2, 5, n)


def same_dist_1d(seed=1, n=300):
    return np.random.default_rng(seed).beta(2, 5, n)


def shifted_1d(seed=2, n=300):
    return np.random.default_rng(seed).beta(8, 2, n)


def test_healthy_on_same_distribution():
    mon = ModelTrustMonitor("stress-v1", ref_proba_1d())
    res = mon.update(same_dist_1d())
    assert isinstance(res, Layer2Result)
    assert res.model_id == "stress-v1"
    assert res.n_windows == 300
    assert res.degraded is False
    assert res.health == pytest.approx(100.0)
    assert set(res.signals) == {"psi_proba", "ks_proba", "mean_confidence_drop"}
    for sig in res.signals.values():
        assert isinstance(sig, ModelSignal)
        assert sig.breached is False


def test_shift_flags_degradation():
    mon = ModelTrustMonitor("stress-v1", ref_proba_1d())
    res = mon.update(shifted_1d())
    assert res.degraded is True
    assert res.health < 100.0
    assert res.signals["psi_proba"].breached is True
    assert res.signals["psi_proba"].value > DEFAULT_CONFIG["psi_threshold"]
    assert res.signals["ks_proba"].value > 0.0
    assert "pvalue" in res.signals["ks_proba"].details


def test_hysteresis_holds_breach_then_heals():
    ref = ref_proba_1d()
    mon = ModelTrustMonitor("m", ref)  # K = 3 by default
    bad = mon.update(shifted_1d())
    assert bad.degraded is True
    # one clean update (the reference itself: PSI = KS = 0) is not enough
    assert mon.update(ref).degraded is True
    assert mon.update(ref).degraded is True
    # third consecutive clean update heals
    healed = mon.update(ref)
    assert healed.degraded is False
    assert healed.health == pytest.approx(100.0)


def test_n_windows_accumulates():
    mon = ModelTrustMonitor("m", ref_proba_1d())
    mon.update(same_dist_1d(n=50))
    mon.update(same_dist_1d(n=70))
    assert mon.n_windows == 120


def test_confidence_drop_signal():
    ref = np.full(200, 0.95)  # very confident reference
    mon = ModelTrustMonitor("m", ref)
    # current batch: confident too -> no breach
    ok = mon.update(np.full(50, 0.93))
    assert ok.signals["mean_confidence_drop"].breached is False
    # current batch: uncertain -> drop breaches
    bad = mon.update(np.full(50, 0.55))
    sig = bad.signals["mean_confidence_drop"]
    assert sig.value == pytest.approx(0.95 - 0.55, abs=1e-9)
    assert sig.breached is True


def test_explicit_confidences_used():
    mon = ModelTrustMonitor("m", np.full(200, 0.9))
    res = mon.update(np.full(50, 0.9), confidences=np.full(50, 0.4))
    sig = res.signals["mean_confidence_drop"]
    assert sig.details["cur_mean_confidence"] == pytest.approx(0.4)
    assert sig.breached is True  # 0.9 - 0.4 = 0.5 drop


def test_label_gated_signals_bootstrapping_and_accuracy_drop():
    ref = ref_proba_1d()
    mon = ModelTrustMonitor(
        "m", ref, config={"accuracy_window": 3, "hysteresis_streak": 1}
    )
    proba = same_dist_1d(seed=5, n=100)
    y_good = (proba >= 0.5).astype(int)  # perfect labels
    r1 = mon.update(proba, y_true=y_good)
    assert "ece_drift" in r1.signals and "rolling_accuracy" in r1.signals
    assert r1.signals["ece_drift"].value == pytest.approx(0.0)  # bootstrapped
    assert r1.signals["ece_drift"].details["note"].startswith("reference ECE boot")
    assert r1.signals["rolling_accuracy"].value == pytest.approx(1.0)
    assert r1.degraded is False
    # flipped labels: accuracy collapses -> rolling accuracy breaches
    y_bad = 1 - y_good
    r2 = mon.update(proba, y_true=y_bad)
    assert r2.signals["rolling_accuracy"].value == pytest.approx(0.5)
    assert r2.signals["rolling_accuracy"].breached is True
    assert r2.degraded is True


def test_label_gated_ece_drift_with_config_ref():
    mon = ModelTrustMonitor("m", ref_proba_1d(), config={"ece_ref": 0.02})
    proba = same_dist_1d(seed=5, n=100)
    y = (proba >= 0.5).astype(int)
    res = mon.update(proba, y_true=y)
    assert res.signals["ece_drift"].details["ref_ece"] == pytest.approx(0.02)


def test_2d_proba_path():
    rng = np.random.default_rng(0)
    ref = rng.dirichlet([9, 1, 1], 400)  # confident reference
    cur_same = np.random.default_rng(1).dirichlet([9, 1, 1], 200)
    cur_shift = np.random.default_rng(2).dirichlet([2, 2, 2], 200)  # uncertain
    mon = ModelTrustMonitor("multi", ref)
    ok = mon.update(cur_same)
    assert ok.degraded is False
    bad = mon.update(cur_shift)
    assert bad.degraded is True
    assert bad.signals["psi_proba"].breached is True


def test_2d_binary_accepts_y_true():
    rng = np.random.default_rng(0)
    ref = rng.dirichlet([4, 1], 300)
    mon = ModelTrustMonitor("bin2d", ref, config={"hysteresis_streak": 1})
    cur = np.random.default_rng(1).dirichlet([4, 1], 100)
    y = (cur[:, 1] >= 0.5).astype(int)
    res = mon.update(cur, y_true=y)
    assert "ece_drift" in res.signals
    assert "rolling_accuracy" in res.signals


def test_multiclass_rejects_label_gated_signals():
    rng = np.random.default_rng(0)
    ref = rng.dirichlet([5, 2, 1], 200)
    mon = ModelTrustMonitor("multi", ref)
    cur = np.random.default_rng(1).dirichlet([5, 2, 1], 50)
    with pytest.raises(ValueError, match="binary model"):
        mon.update(cur, y_true=np.zeros(50, dtype=int))


def test_reset_clears_state():
    mon = ModelTrustMonitor("m", ref_proba_1d())
    mon.update(shifted_1d())
    assert mon.n_windows == 300
    mon.reset()
    assert mon.n_windows == 0
    res = mon.update(ref_proba_1d())  # healthy again immediately
    assert res.degraded is False


def test_validation_errors():
    ref = ref_proba_1d()
    with pytest.raises(ValueError):
        ModelTrustMonitor("", ref)
    with pytest.raises(ValueError):
        ModelTrustMonitor("m", np.array([]))
    with pytest.raises(ValueError):
        ModelTrustMonitor("m", np.array([0.2, 1.5, -0.1]))
    with pytest.raises(ValueError):
        ModelTrustMonitor("m", np.full((10, 2), 0.5) * 0.9)  # rows != 1
    with pytest.raises(ValueError):
        ModelTrustMonitor("m", ref, config={"nope": 1})
    with pytest.raises(ValueError):
        ModelTrustMonitor("m", ref, config={"psi_threshold": -1})
    with pytest.raises(ValueError):
        ModelTrustMonitor("m", ref, config={"hysteresis_streak": 0})

    mon = ModelTrustMonitor("m", ref)
    with pytest.raises(ValueError):
        mon.update(np.array([]))
    with pytest.raises(ValueError):
        mon.update(np.array([0.1, 2.0]))
    with pytest.raises(ValueError):
        mon.update(np.full((10, 2), 0.5))  # ndim mismatch vs 1D reference
    with pytest.raises(ValueError):
        mon.update(same_dist_1d(n=10), y_true=np.zeros(9, dtype=int))
    with pytest.raises(ValueError):
        mon.update(same_dist_1d(n=10), y_true=np.full(10, 7))
    with pytest.raises(ValueError):
        mon.update(same_dist_1d(n=10), confidences=np.full(9, 0.5))
    with pytest.raises(ValueError):
        mon.update(same_dist_1d(n=10), confidences=np.full(10, 1.5))
    with pytest.raises(ValueError):
        ModelTrustMonitor("m", ref, config={"signal_weights": {"nope": 1.0}})
