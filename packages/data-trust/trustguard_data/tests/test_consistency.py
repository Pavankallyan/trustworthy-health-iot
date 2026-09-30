"""Unit tests for trustguard_data.consistency."""

import numpy as np
import pytest

from trustguard_data.consistency import acc_consistency, hr_agreement


def _acc_pair(seed=0, delay_s=0.0, fs=100.0, n=2000):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    a = (
        np.sin(2 * np.pi * 0.5 * t)
        + 0.6 * np.sin(2 * np.pi * 1.7 * t + 1.3)
        + 0.05 * rng.standard_normal(n)
    )
    b = (
        np.sin(2 * np.pi * 0.5 * (t - delay_s))
        + 0.6 * np.sin(2 * np.pi * 1.7 * (t - delay_s) + 1.3)
        + 0.05 * rng.standard_normal(n)
    )
    axes_a = {"x": a, "y": 0.5 * a, "z": 0.2 * a}
    axes_b = {"x": b, "y": 0.5 * b, "z": 0.2 * b}
    return axes_a, axes_b


def test_acc_consistency_synchronized():
    acc_a, acc_b = _acc_pair(delay_s=0.0)
    r = acc_consistency(acc_a, 100.0, acc_b, 100.0)
    assert r["correlation"] > 0.9
    assert r["lag_s"] == pytest.approx(0.0, abs=0.05)
    assert r["desync"] is False


def test_acc_consistency_desync_detected():
    acc_a, acc_b = _acc_pair(delay_s=0.7)
    r = acc_consistency(acc_a, 100.0, acc_b, 100.0, desync_lag_s=0.5)
    assert r["lag_s"] == pytest.approx(0.7, abs=0.05)
    assert r["desync"] is True


def test_acc_consistency_small_lag_not_desync():
    acc_a, acc_b = _acc_pair(delay_s=0.2)
    r = acc_consistency(acc_a, 100.0, acc_b, 100.0, desync_lag_s=0.5)
    assert r["lag_s"] == pytest.approx(0.2, abs=0.05)
    assert r["desync"] is False


def test_acc_consistency_different_rates():
    acc_a, _ = _acc_pair(delay_s=0.0, fs=100.0, n=2000)
    _, acc_b = _acc_pair(delay_s=0.0, fs=50.0, n=1000)
    r = acc_consistency(acc_a, 100.0, acc_b, 50.0)
    assert r["common_fs"] == 50.0
    assert r["correlation"] > 0.9


def test_acc_consistency_bad_input():
    acc_a, acc_b = _acc_pair()
    with pytest.raises(ValueError):
        acc_consistency({}, 100.0, acc_b, 100.0)
    bad = {"x": np.zeros(100), "y": np.zeros(50)}
    with pytest.raises(ValueError):
        acc_consistency(bad, 100.0, acc_b, 100.0)
    with pytest.raises(ValueError):
        acc_consistency(acc_a, 0.0, acc_b, 100.0)


def _pulse_train(bpm, seconds=30, fs=100.0, seed=0, noise=0.02):
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    x = np.zeros(n)
    period = 60.0 / bpm
    k = 0
    while True:
        idx = int(round(k * period * fs)) + 5
        if idx >= n:
            break
        x[idx] = 1.0
        k += 1
    return x + noise * rng.standard_normal(n)


def test_hr_agreement_matching():
    ecg = _pulse_train(60, seed=0)
    bvp = _pulse_train(60, seed=1)
    r = hr_agreement(ecg, 100.0, bvp, 100.0)
    assert r["agreement"] is True
    assert r["hr_ecg_bpm"] == pytest.approx(60.0, abs=1.0)
    assert r["hr_bvp_bpm"] == pytest.approx(60.0, abs=1.0)
    assert r["abs_diff_bpm"] == pytest.approx(0.0, abs=1.0)


def test_hr_agreement_mismatch():
    ecg = _pulse_train(60, seed=0)
    bvp = _pulse_train(90, seed=1)
    r = hr_agreement(ecg, 100.0, bvp, 100.0)
    assert r["agreement"] is False
    assert r["abs_diff_bpm"] == pytest.approx(30.0, abs=2.0)


def test_hr_agreement_flat_signal():
    r = hr_agreement(np.zeros(3000), 100.0, _pulse_train(60), 100.0)
    assert r["hr_ecg_bpm"] is None
    assert r["agreement"] is False
    assert r["abs_diff_bpm"] is None


def test_hr_agreement_bad_tol():
    with pytest.raises(ValueError):
        hr_agreement(np.zeros(100), 100.0, np.zeros(100), 100.0, tol_bpm=-1.0)
