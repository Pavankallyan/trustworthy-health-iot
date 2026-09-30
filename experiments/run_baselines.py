"""Baselines: IsolationForest, KS-only, PCA-reconstruction.

Pre-registered in experiment_plan.md (2026-09-30). Each baseline replays the
EXACT injection runs from run_layer1 (same FAULTS order, same TEST_SUBJECTS,
same seeds 42000 + run_index, so the corrupted windows are identical) and is
scored with the same detection rule (target channel flagged). Thresholds are
tuned on each subject's validation windows to match the pipeline's validation
false-alarm rate, then frozen for test.

The PCA baseline is the honest linear proxy for the "autoencoder
reconstruction" baseline — no deep-learning framework is used in this build,
and the substitution is stated in the plan, not hidden.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import run_layer1 as R  # noqa: E402
from trustguard_data.stream import Window  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)
SEED = 42


def window_features(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.full(8, np.nan)
    q10, q50, q90 = np.percentile(a, [10, 50, 90])
    return np.array([a.mean(), a.std(), a.min(), a.max(),
                     np.sqrt(np.mean(a ** 2)), q10, q50, q90])


class IsolationForestBaseline:
    name = "isolation_forest"

    def fit(self, train_windows, target: str):
        X = np.array([window_features(w.channels[target])
                      for w in train_windows])
        self._med = np.nanmedian(X, axis=0)
        X = np.where(np.isnan(X), self._med, X)
        self.model = IsolationForest(n_estimators=100, random_state=SEED,
                                     n_jobs=2)
        self.model.fit(X)

    def scores(self, windows, target: str) -> np.ndarray:
        X = np.array([window_features(w.channels[target]) for w in windows])
        X = np.where(np.isnan(X), self._med, X)
        return -self.model.score_samples(X)  # higher = more anomalous


class KSOnlyBaseline:
    name = "ks_only"

    def fit(self, train_windows, target: str):
        base = np.concatenate([w.channels[target].ravel()
                               for w in train_windows])
        self._base = base[np.isfinite(base)]

    def scores(self, windows, target: str) -> np.ndarray:
        out = []
        for w in windows:
            a = np.asarray(w.channels[target]).ravel()
            a = a[np.isfinite(a)]
            if len(a) < 10:
                out.append(0.0)
                continue
            p = stats.ks_2samp(self._base, a).pvalue
            out.append(-np.log10(max(p, 1e-300)))  # higher = more drift
        return np.array(out)


class PCAReconstructionBaseline:
    name = "pca_reconstruction"

    def fit(self, train_windows, target: str):
        X = np.array([window_features(w.channels[target])
                      for w in train_windows])
        self._med = np.nanmedian(X, axis=0)
        X = np.where(np.isnan(X), self._med, X)
        mu, sd = X.mean(0), X.std(0) + 1e-12
        self._mu, self._sd = mu, sd
        Xs = (X - mu) / sd
        self.pca = PCA(n_components=0.95, random_state=SEED)
        self.pca.fit(Xs)

    def scores(self, windows, target: str) -> np.ndarray:
        X = np.array([window_features(w.channels[target]) for w in windows])
        X = np.where(np.isnan(X), self._med, X)
        Xs = (X - self._mu) / self._sd
        rec = self.pca.inverse_transform(self.pca.transform(Xs))
        return np.mean((Xs - rec) ** 2, axis=1)  # higher = more anomalous


BASELINES = [IsolationForestBaseline, KSOnlyBaseline, PCAReconstructionBaseline]


def tune_threshold(scores_val: np.ndarray, target_far: float) -> float:
    """Threshold giving val flag rate ~= target FAR (floor 1e-4)."""
    rate = min(max(target_far, 1e-4), 0.5)
    return float(np.quantile(scores_val, 1.0 - rate))


def main():
    layer1 = json.load(open(os.path.join(RESULTS, "layer1_wesad.json")))
    val_far = {s: layer1["false_alarm"][s]["far"] for s in R.TEST_SUBJECTS}

    # per-subject windows/ranges, identical to run_layer1
    subj = {}
    for s in R.TEST_SUBJECTS:
        train_w, val_w, test_w = R.subject_windows(s)
        ranges = R.common.empirical_ranges(
            train_w, list(train_w[0].channels.keys()))
        subj[s] = {"train": train_w, "val": val_w, "test": test_w,
                   "ranges": ranges}

    summary = {"dataset": "wesad", "val_far_matched": val_far,
               "baselines": {}}
    for cls in BASELINES:
        name = cls.name
        rec = {"n": 0, "detected": 0, "by_fault": {}}
        # fit + tune per (baseline, subject, target)
        fitted: dict[tuple[str, str], tuple] = {}
        for s in R.TEST_SUBJECTS:
            for tgt in R.TARGETS:
                det = cls()
                det.fit(subj[s]["train"], tgt)
                s_val = det.scores(subj[s]["val"], tgt)
                thr = tune_threshold(s_val, val_far[s])
                fitted[(s, tgt)] = (det, thr)
        for i, (ftype, params) in enumerate(R.FAULTS):
            s = R.TEST_SUBJECTS[i % len(R.TEST_SUBJECTS)]
            w = subj[s]["test"][(i // len(R.TEST_SUBJECTS))
                                % len(subj[s]["test"])]
            target = R.TARGETS[i % len(R.TARGETS)]
            win = Window(device_id=w.device_id, start_s=0.0,
                         end_s=R.WINDOW_S, channels=dict(w.channels),
                         fs=dict(w.fs))
            injected, _truth = R.inject_fault(win, ftype, params, target,
                                              subj[s]["ranges"], seed=42000 + i)
            det, thr = fitted[(s, target)]
            score = det.scores([injected], target)[0]
            detected = bool(score >= thr)
            rec["n"] += 1
            rec["detected"] += int(detected)
            d = rec["by_fault"].setdefault(ftype, {"n": 0, "det": 0})
            d["n"] += 1
            d["det"] += int(detected)
            print(f"[wesad/{name}] run {i}: {ftype} {params} on {target}: "
                  f"detected={detected} score={score:.3f} thr={thr:.3f}",
                  flush=True)
        rec["recall"] = rec["detected"] / rec["n"] if rec["n"] else 0.0
        for f, d in rec["by_fault"].items():
            d["recall"] = d["det"] / d["n"] if d["n"] else 0.0
        summary["baselines"][name] = rec
        print(f"[wesad/{name}] recall={rec['recall']:.3f}", flush=True)

    path = os.path.join(RESULTS, "baselines_wesad.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"-> {path}", flush=True)
    return summary


if __name__ == "__main__":
    main()
