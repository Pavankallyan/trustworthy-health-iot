"""End-to-end integration test: synthetic windows -> features -> model -> monitor.

Trains a tiny stress classifier on synthetic multi-channel windows, then
feeds the ModelTrustMonitor clean vs distribution-shifted predicted
probabilities and asserts the monitor flags the shift as degradation.
All data is synthetic with fixed seeds; no datasets, no network.
"""

import numpy as np
import pytest

from trustguard_model._compat import Window
from trustguard_model.baseline import (
    evaluate,
    load_model,
    save_model,
    train_stress_model,
)
from trustguard_model.features import extract_features
from trustguard_model.monitors import ModelTrustMonitor

CHANNELS = ["ecg", "bvp"]
N_PER_CLASS = 100  # 200 windows total: PSI needs reasonable batch sizes
N_SAMPLES = 120
FS = 50.0


def synth_windows(seed=11, ecg_shift=0.0):
    """40 windows; class 1 has higher ecg/bvp means. ecg_shift simulates drift."""
    rng = np.random.default_rng(seed)
    windows, labels = [], []
    for cls in (0, 1):
        for i in range(N_PER_CLASS):
            ecg = rng.normal(1.5 * cls + ecg_shift, 1.0, N_SAMPLES)
            bvp = rng.normal(5.0 + 1.0 * cls, 0.5, N_SAMPLES)
            start = float((cls * N_PER_CLASS + i) * N_SAMPLES / FS)
            windows.append(
                Window(
                    device_id="dev-int",
                    start_s=start,
                    end_s=start + N_SAMPLES / FS,
                    channels={"ecg": ecg, "bvp": bvp},
                    fs={"ecg": FS, "bvp": FS},
                )
            )
            labels.append(cls)
    return windows, np.array(labels)


def test_end_to_end_train_then_detect_shift(tmp_path):
    windows, y = synth_windows(seed=11)
    X = extract_features(windows, CHANNELS)
    assert X.shape == (2 * N_PER_CLASS, 14)
    assert not X.isna().any().any()

    # every 4th window held out -> 150 train / 50 test, both classes in each
    idx = np.arange(2 * N_PER_CLASS)
    train_idx = idx[idx % 4 != 3]
    test_idx = idx[idx % 4 == 3]
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]
    assert set(np.unique(y_train)) == {0, 1}
    assert set(np.unique(y_test)) == {0, 1}

    model = train_stress_model(X_train, y_train, seed=11)
    metrics = evaluate(model, X_test, y_test)
    assert metrics["accuracy"] >= 0.8, metrics

    # persistence round-trip keeps the deployment story intact
    path = tmp_path / "stress.joblib"
    save_model(model, path)
    model = load_model(path)

    reference_proba = model.predict_proba(X_train)[:, 1]
    monitor = ModelTrustMonitor("stress-v1", reference_proba)

    # clean deployment batch: no degradation
    clean_proba = model.predict_proba(X_test)[:, 1]
    clean_res = monitor.update(clean_proba)
    assert clean_res.degraded is False
    assert clean_res.health == pytest.approx(100.0)

    # shifted deployment: sensor drift pushes ecg up -> model saturates at 1
    shifted_windows, _ = synth_windows(seed=77, ecg_shift=3.0)
    X_shift = extract_features(shifted_windows, CHANNELS)
    shift_proba = model.predict_proba(X_shift)[:, 1]
    shift_res = monitor.update(shift_proba)
    assert shift_res.degraded is True
    assert shift_res.signals["psi_proba"].breached is True
    assert shift_res.signals["psi_proba"].value > 0.25
    assert shift_res.health < 100.0
    assert shift_res.n_windows == len(X_test) + len(X_shift)
