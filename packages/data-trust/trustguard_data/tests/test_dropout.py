"""Unit tests for trustguard_data.dropout."""

import numpy as np
import pytest

from trustguard_data.dropout import detect_dropout


def test_no_dropout():
    rng = np.random.default_rng(0)
    r = detect_dropout(rng.standard_normal(1000), 100.0)
    assert r["n_missing"] == 0
    assert r["missing_fraction"] == 0.0
    assert r["n_runs"] == 0
    assert r["longest_nan_run_samples"] == 0
    assert r["longest_nan_run_s"] == 0.0
    assert r["n_total"] == 1000


def test_single_nan_run():
    x = np.ones(1000)
    x[100:300] = np.nan  # 200 samples = 2 s @ 100 Hz
    r = detect_dropout(x, 100.0)
    assert r["n_missing"] == 200
    assert r["missing_fraction"] == pytest.approx(0.2)
    assert r["n_runs"] == 1
    assert r["longest_nan_run_samples"] == 200
    assert r["longest_nan_run_s"] == pytest.approx(2.0)


def test_multiple_runs():
    x = np.ones(1000)
    x[0:50] = np.nan
    x[200:210] = np.nan
    x[990:1000] = np.nan
    r = detect_dropout(x, 50.0)
    assert r["n_runs"] == 3
    assert r["longest_nan_run_samples"] == 50
    assert r["longest_nan_run_s"] == pytest.approx(1.0)


def test_all_nan():
    r = detect_dropout(np.full(100, np.nan), 10.0)
    assert r["missing_fraction"] == 1.0
    assert r["n_runs"] == 1


def test_bad_input():
    with pytest.raises(ValueError):
        detect_dropout(np.array([]), 100.0)
    with pytest.raises(ValueError):
        detect_dropout(np.zeros((10, 2)), 100.0)
    with pytest.raises(ValueError):
        detect_dropout(np.zeros(10), 0.0)
    with pytest.raises(TypeError):
        detect_dropout([1.0, 2.0], 100.0)
