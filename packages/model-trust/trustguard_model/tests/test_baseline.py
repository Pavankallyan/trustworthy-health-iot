"""Unit tests for trustguard_model.baseline."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from trustguard_model.baseline import (
    evaluate,
    expected_calibration_error,
    load_model,
    save_model,
    train_stress_model,
)


def make_xy(seed=7, n_per_class=30):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(0.0, 1.0, (n_per_class, 4))
    x1 = rng.normal(2.0, 1.0, (n_per_class, 4))
    X = pd.DataFrame(
        np.vstack([x0, x1]), columns=[f"f{i}" for i in range(4)]
    )
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return X, y


def test_train_returns_pipeline_and_scores_well():
    X, y = make_xy()
    model = train_stress_model(X, y, seed=11)
    assert isinstance(model, Pipeline)
    assert [name for name, _ in model.steps] == ["imputer", "scaler", "clf"]
    metrics = evaluate(model, X, y)
    assert set(metrics.keys()) == {"accuracy", "f1", "auc", "ece"}
    assert metrics["accuracy"] > 0.95
    assert metrics["auc"] > 0.95
    assert 0.0 <= metrics["ece"] <= 1.0


def test_train_is_deterministic_for_fixed_seed():
    X, y = make_xy()
    m1 = train_stress_model(X, y, seed=11)
    m2 = train_stress_model(X, y, seed=11)
    p1 = m1.predict_proba(X)
    p2 = m2.predict_proba(X)
    assert np.allclose(p1, p2)


def test_train_handles_nan_features_via_imputer():
    X, y = make_xy()
    X = X.copy()
    X.iloc[0, 0] = np.nan
    X.iloc[5, 2] = np.nan
    model = train_stress_model(X, y, seed=11)
    preds = model.predict(X)  # must not raise / produce NaN
    assert not np.isnan(np.asarray(model.predict_proba(X))).any()
    assert set(np.unique(preds)) <= {0, 1}


def test_ece_known_values():
    # Perfectly confident and correct -> ECE ~ 0
    y = np.array([0, 0, 1, 1])
    assert expected_calibration_error(y, np.array([0.0, 0.0, 1.0, 1.0])) == pytest.approx(0.0)
    # Confident but always wrong -> ECE ~ 1
    assert expected_calibration_error(y, np.array([1.0, 1.0, 0.0, 0.0])) == pytest.approx(1.0)
    # Constant 0.5 with balanced labels -> ECE ~ 0
    assert expected_calibration_error(y, np.full(4, 0.5)) == pytest.approx(0.0)


def test_save_load_roundtrip(tmp_path):
    X, y = make_xy()
    model = train_stress_model(X, y, seed=11)
    path = tmp_path / "stress.joblib"
    saved = save_model(model, path)
    assert saved.exists()
    loaded = load_model(path)
    assert np.allclose(model.predict_proba(X), loaded.predict_proba(X))


def test_save_load_pkl_suffix(tmp_path):
    X, y = make_xy()
    model = train_stress_model(X, y, seed=11)
    path = tmp_path / "stress.pkl"
    save_model(model, path)
    assert np.allclose(model.predict_proba(X), load_model(path).predict_proba(X))


def test_validation_errors(tmp_path):
    X, y = make_xy()
    with pytest.raises(TypeError):
        train_stress_model(X, y, seed="11")  # seed must be int
    with pytest.raises(TypeError):
        train_stress_model(X, y, seed=1.5)
    with pytest.raises(ValueError):
        train_stress_model(X, np.array([0, 0, 0]), seed=1)  # length mismatch
    with pytest.raises(ValueError):
        train_stress_model(X, np.zeros(len(X), dtype=int), seed=1)  # single class
    with pytest.raises(ValueError):
        train_stress_model(X, np.array([0, 1, 2] * 20), seed=1)  # not binary
    with pytest.raises(ValueError):
        train_stress_model(pd.DataFrame(), y[:0], seed=1)  # empty X
    with pytest.raises(TypeError):
        train_stress_model(X.to_numpy(), y, seed=1)  # X must be DataFrame
    model = train_stress_model(X, y, seed=1)
    with pytest.raises(ValueError):
        save_model(model, tmp_path / "m.txt")  # bad suffix
    with pytest.raises(ValueError):
        save_model(model, tmp_path / "nope" / "m.joblib")  # missing parent
    with pytest.raises(ValueError):
        load_model(tmp_path / "missing.joblib")
    with pytest.raises(TypeError):
        load_model(123)
