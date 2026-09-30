"""Unit tests for trustguard_data.drift."""

import numpy as np
import pytest

from trustguard_data.drift import ks_drift, psi, trend_slope


def test_ks_drift_same_distribution():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(1000)
    b = rng.standard_normal(1000)
    r = ks_drift(a, b)
    assert r["pvalue"] > 0.05
    assert 0.0 <= r["statistic"] <= 1.0
    assert r["n"] == 1000 and r["n_baseline"] == 1000


def test_ks_drift_shifted_distribution():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(1000)
    b = rng.standard_normal(1000) + 1.0
    r = ks_drift(a, b)
    assert r["pvalue"] < 1e-6
    assert r["statistic"] > 0.3


def test_ks_drift_drops_nan():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(500)
    a[::10] = np.nan
    b = rng.standard_normal(500)
    r = ks_drift(a, b)
    assert r["n"] == 450
    assert r["pvalue"] > 0.01


def test_ks_drift_needs_data():
    with pytest.raises(ValueError):
        ks_drift(np.array([1.0, np.nan]), np.zeros(10))


def test_psi_same_distribution():
    rng = np.random.default_rng(3)
    e = rng.standard_normal(2000)
    a = rng.standard_normal(2000)
    assert psi(e, a) < 0.1


def test_psi_shifted_distribution():
    rng = np.random.default_rng(3)
    e = rng.standard_normal(2000)
    a = rng.standard_normal(2000) + 0.8
    assert psi(e, a) > 0.25


def test_psi_degenerate_expected():
    e = np.full(100, 2.0)
    assert psi(e, np.full(100, 2.0)) == 0.0
    assert psi(e, np.full(100, 5.0)) == float("inf")


def test_psi_bad_bins():
    with pytest.raises(ValueError):
        psi(np.zeros(10), np.zeros(10), bins=1)


def test_trend_slope_ramp():
    t = np.arange(500) / 50.0  # 10 s @ 50 Hz
    x = 1.0 * t  # exactly 1.0 unit/s
    r = trend_slope(x, 50.0)
    assert r["slope_per_s"] == pytest.approx(1.0, rel=1e-9)
    assert r["pvalue"] < 1e-10
    assert r["n"] == 500


def test_trend_slope_flat():
    r = trend_slope(np.full(200, 4.0), 100.0)
    assert r["slope_per_s"] == 0.0
    assert r["pvalue"] == 1.0


def test_trend_slope_no_trend():
    rng = np.random.default_rng(5)
    r = trend_slope(rng.standard_normal(2000), 100.0)
    assert abs(r["slope_per_s"]) < 0.05
    assert r["pvalue"] > 0.05


def test_trend_slope_drops_nan():
    t = np.arange(250) / 50.0
    x = 2.0 * t  # exactly 2.0 units/s
    x[::5] = np.nan
    r = trend_slope(x, 50.0)
    assert r["slope_per_s"] == pytest.approx(2.0, rel=1e-9)
    assert r["n"] == 200


def test_trend_slope_bad_input():
    with pytest.raises(ValueError):
        trend_slope(np.zeros(2), 100.0)
    with pytest.raises(ValueError):
        trend_slope(np.zeros(100), 0.0)
