"""Unit tests for trustguard_data.stuck."""

import numpy as np
import pytest

from trustguard_data.stuck import detect_stuck


def test_constant_signal_is_stuck():
    x = np.full(1000, 3.25)
    r = detect_stuck(x, 100.0)
    assert r["is_stuck"] is True
    assert r["longest_stuck_s"] == pytest.approx(10.0)
    assert r["longest_stuck_samples"] == 1000
    assert r["stuck_fraction"] == pytest.approx(1.0)
    assert r["value"] == pytest.approx(3.25)


def test_noisy_signal_not_stuck():
    rng = np.random.default_rng(0)
    r = detect_stuck(rng.standard_normal(2000), 100.0)
    assert r["is_stuck"] is False
    assert r["n_stuck_runs"] == 0
    assert r["value"] is None


def test_short_flat_blip_not_stuck():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(1000)
    x[100:150] = 0.5  # 0.5 s flat < 5 s minimum
    r = detect_stuck(x, 100.0)
    assert r["is_stuck"] is False


def test_nan_breaks_stuck_runs():
    # Two 4 s constant halves separated by NaN; min_duration 5 s -> not stuck.
    x = np.full(1000, 2.0)
    x[400:600] = np.nan
    r = detect_stuck(x, 100.0, min_duration_s=5.0)
    assert r["is_stuck"] is False


def test_partial_stuck_fraction():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(1000)
    x[0:600] = -1.5  # 6 s stuck of 10 s
    r = detect_stuck(x, 100.0)
    assert r["is_stuck"] is True
    assert r["longest_stuck_s"] == pytest.approx(6.0)
    assert r["stuck_fraction"] == pytest.approx(0.6)
    assert r["value"] == pytest.approx(-1.5)


def test_custom_min_duration():
    x = np.full(1000, 1.0)
    r = detect_stuck(x, 100.0, min_duration_s=20.0)
    assert r["is_stuck"] is False  # only 10 s available


def test_bad_input():
    with pytest.raises(ValueError):
        detect_stuck(np.array([]), 100.0)
    with pytest.raises(ValueError):
        detect_stuck(np.zeros(10), -1.0)
    with pytest.raises(ValueError):
        detect_stuck(np.zeros(10), 100.0, tol=-0.1)
    with pytest.raises(ValueError):
        detect_stuck(np.zeros(10), 100.0, min_duration_s=0.0)
