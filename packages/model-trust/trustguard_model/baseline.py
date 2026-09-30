"""Baseline stress classifier: training, evaluation, persistence.

The reference model is intentionally small and fast — a median imputer (so
the model never sees NaNs from :func:`extract_features`), a standard scaler,
and a logistic regression classifier.  ``seed`` is required and threaded
into every randomized step, so training is fully deterministic.

Evaluation reports accuracy, F1, ROC-AUC and expected calibration error
(ECE, implemented here with 10 equal-width confidence bins).  Binary
classification (labels 0/1) only; anything else raises ``ValueError``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

__all__ = [
    "train_stress_model",
    "evaluate",
    "save_model",
    "load_model",
    "N_ECE_BINS",
]

#: Number of equal-width bins used by the hand-rolled expected calibration error.
N_ECE_BINS = 10

PathLike = Union[str, Path]


def _validate_xy(X: pd.DataFrame, y: np.ndarray, where: str) -> np.ndarray:
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"{where}: X must be a pandas DataFrame")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"{where}: X must be non-empty (got shape {X.shape})")
    y = np.asarray(y).ravel()
    if y.shape[0] != X.shape[0]:
        raise ValueError(
            f"{where}: X has {X.shape[0]} rows but y has {y.shape[0]} labels"
        )
    if y.shape[0] < 2:
        raise ValueError(f"{where}: need at least 2 samples, got {y.shape[0]}")
    uniq = np.unique(y)
    if uniq.size != 2 or not set(uniq.tolist()) <= {0, 1}:
        raise ValueError(
            f"{where}: y must be binary with labels in {{0, 1}}; "
            f"got unique values {uniq.tolist()}"
        )
    return y.astype(int)


def _validate_seed(seed: int, where: str) -> int:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"{where}: seed must be an int, got {type(seed).__name__}")
    return seed


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int = N_ECE_BINS) -> float:
    """Hand-rolled expected calibration error (equal-width confidence bins)."""
    y_true = np.asarray(y_true, dtype=int).ravel()
    proba = np.asarray(proba, dtype=float).ravel()
    if proba.shape[0] != y_true.shape[0]:
        raise ValueError("y_true and proba must have the same length")
    if np.any((proba < 0.0) | (proba > 1.0)):
        raise ValueError("proba values must lie in [0, 1]")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = proba.shape[0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (proba >= lo) & (proba <= hi)
        else:
            mask = (proba >= lo) & (proba < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        acc = float(y_true[mask].mean())
        conf = float(proba[mask].mean())
        ece += (count / n) * abs(acc - conf)
    return float(ece)


def train_stress_model(X: pd.DataFrame, y: np.ndarray, seed: int) -> Pipeline:
    """Train the baseline binary stress classifier.

    Pipeline: ``SimpleImputer(strategy="median")`` -> ``StandardScaler`` ->
    ``LogisticRegression``.  Deterministic for a fixed ``seed``.

    Parameters
    ----------
    X:
        Feature DataFrame from :func:`extract_features` (may contain NaNs;
        the imputer handles them).
    y:
        Binary labels (0/1), one per row of ``X``.
    seed:
        Required int; passed to the classifier's ``random_state``.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fitted pipeline with ``predict`` / ``predict_proba``.
    """
    y = _validate_xy(X, y, "train_stress_model")
    seed = _validate_seed(seed, "train_stress_model")
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=2000, random_state=seed),
            ),
        ]
    )
    model.fit(X, y)
    return model


def evaluate(model: Pipeline, X: pd.DataFrame, y: np.ndarray) -> dict:
    """Evaluate a fitted binary classifier.

    Returns a dict of floats with exactly these keys: ``accuracy``, ``f1``,
    ``auc`` (ROC-AUC on the positive-class probability), ``ece`` (expected
    calibration error, 10 bins).
    """
    y = _validate_xy(X, y, "evaluate")
    if not hasattr(model, "predict_proba"):
        raise TypeError("evaluate: model must expose predict_proba")
    try:
        proba = np.asarray(model.predict_proba(X), dtype=float)
    except Exception as exc:
        raise ValueError(f"evaluate: model.predict_proba(X) failed: {exc}") from exc
    if proba.ndim != 2 or proba.shape[0] != X.shape[0] or proba.shape[1] != 2:
        raise ValueError(
            "evaluate: predict_proba must return shape (n_samples, 2) for the "
            f"binary model; got {proba.shape}"
        )
    p1 = proba[:, 1]
    y_pred = (p1 >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1": float(f1_score(y, y_pred)),
        "auc": float(roc_auc_score(y, p1)),
        "ece": float(expected_calibration_error(y, p1)),
    }


def save_model(model: Pipeline, path: PathLike) -> Path:
    """Persist a fitted pipeline with joblib.  Returns the resolved path."""
    if not isinstance(path, (str, Path)):
        raise TypeError("save_model: path must be a str or pathlib.Path")
    path = Path(path)
    if path.suffix not in (".joblib", ".pkl", ".pickle"):
        raise ValueError(
            "save_model: path must end in .joblib, .pkl, or .pickle; "
            f"got '{path.name}'"
        )
    if path.parent != Path(".") and not path.parent.exists():
        raise ValueError(
            f"save_model: parent directory does not exist: {path.parent}"
        )
    joblib.dump(model, path)
    return path


def load_model(path: PathLike) -> Pipeline:
    """Load a pipeline persisted with :func:`save_model`."""
    if not isinstance(path, (str, Path)):
        raise TypeError("load_model: path must be a str or pathlib.Path")
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"load_model: file not found: {path}")
    model = joblib.load(path)
    if not isinstance(model, Pipeline):
        raise TypeError(
            f"load_model: expected a sklearn Pipeline, got {type(model).__name__}"
        )
    return model
