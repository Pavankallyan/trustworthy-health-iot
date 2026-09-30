"""Layer 2 evaluation: WESAD stress model + degradation monitoring.

Pre-registered in experiment_plan.md (2026-09-30).
1. Train a binary stress classifier (stress vs non-stress) on WESAD.
2. Measure test accuracy/F1/AUC/ECE.
3. Replay test streams with injected corruption into the model's inputs and
   check the ModelTrustMonitor flags degradation (label-free), while also
   recording the TRUE accuracy drop (labels exist) to validate the proxy.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (  # noqa: E402
    SEED,
    WESAD_NONSTRESS,
    WESAD_STRESS,
    slice_window,
    split_stream,
    wesad_labels,
    wesad_stream,
    window_label,
)

from trustguard_data.injection import FaultInjector  # noqa: E402
from trustguard_data.stream import Window  # noqa: E402
from trustguard_model.baseline import (  # noqa: E402
    evaluate,
    load_model,
    save_model,
    train_stress_model,
)
from trustguard_model.features import extract_features  # noqa: E402
from trustguard_model.monitors import ModelTrustMonitor  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)

FEATURE_CHANNELS = ["ecg", "eda_c", "emg", "resp", "temp_c", "bvp", "eda_w"]
WIN_S, HOP_S = 60.0, 30.0
# stress periods differ per subject (S2: 2273-2888s, S3: 2244-2884s,
# S4: 3614s+, S5: 3602s+); durations chosen to include stress + non-stress.
SUBJ_DURATION = {"S2": 3200.0, "S3": 3200.0, "S4": 5000.0, "S5": 5000.0}
TRAIN_SUBJ, VAL_SUBJ, TEST_SUBJS = "S2", "S3", ["S4", "S5"]
# degradation segment: S4 [3400, 4600) covers transient -> stress -> rest.
DEG_SUBJ, DEG_START, DEG_END = "S4", 3400.0, 4600.0
DEG_FAULT_START, DEG_FAULT_END = 60.0, 660.0


def build_labeled(subject: str):
    """Windows + binary labels for one subject (30 s hop, splits by subject)."""
    dur = SUBJ_DURATION[subject]
    stream = wesad_stream(subject, FEATURE_CHANNELS, duration_s=dur)
    windows = split_stream(stream, WIN_S, HOP_S)
    labels = wesad_labels(subject, 0.0, dur)
    fs_lab = 700.0
    Xw, y = [], []
    for w in windows:
        i0 = int(round((w.start_s - stream.start_s) * fs_lab))
        i1 = int(round((w.end_s - stream.start_s) * fs_lab))
        lab = window_label(labels[i0:i1])
        if lab == 0:
            continue
        Xw.append(w)
        y.append(1 if lab == WESAD_STRESS else 0)
    X = extract_features(Xw, FEATURE_CHANNELS)
    return Xw, X, np.asarray(y, dtype=int)


def degradation_run(model, monitor_cfg: dict, corrupt: list[dict],
                    seed: int, subj: str = "S4") -> dict:
    """Inject corruption into model inputs; watch the monitor + true accuracy."""
    stream = wesad_stream(subj, FEATURE_CHANNELS, start_s=DEG_START,
                          duration_s=DEG_END - DEG_START)
    # empirical ranges for plausible fault magnitudes
    ranges = {}
    for ch in FEATURE_CHANNELS:
        x = stream.channels[ch]
        ranges[ch] = (float(np.nanmin(x)), float(np.nanmax(x)))
    faults = []
    for c in corrupt:
        f = dict(c)
        f["start"], f["end"] = DEG_FAULT_START, DEG_FAULT_END
        p = dict(f.get("params", {}))
        if f["type"] == "gradual_drift":
            lo, hi = ranges[f["channel"]]
            p = {"magnitude": p.get("rate", 0.05) * (hi - lo)
                 * ((DEG_FAULT_END - DEG_FAULT_START) / 60.0)}
        elif f["type"] == "random_dropout":
            p = {"fraction": p.get("p", 0.3)}
        f["params"] = p
        faults.append(f)
    injector = FaultInjector(seed=seed, plausible_ranges=ranges)
    result = injector.inject(stream, faults)
    corrupted, truth = result.window, result.ground_truth

    clean_w = split_stream(stream, WIN_S, WIN_S)
    corr_w = split_stream(corrupted, WIN_S, WIN_S)
    labels = wesad_labels(subj, DEG_START, DEG_END)
    fs_lab = 700.0

    def predict(ws):
        X = extract_features(ws, FEATURE_CHANNELS)
        return model.predict_proba(X)[:, 1], X

    clean_proba, _ = predict(clean_w)
    corr_proba, _ = predict(corr_w)

    # true labels per window for honest accuracy measurement
    y_true = []
    for w in corr_w:
        i0 = int(round((w.start_s - stream.start_s) * fs_lab))
        i1 = int(round((w.end_s - stream.start_s) * fs_lab))
        lab = window_label(labels[i0:i1])
        y_true.append(-1 if lab == 0 else (1 if lab == WESAD_STRESS else 0))
    y_true = np.asarray(y_true)
    mask = y_true >= 0
    acc_clean = float(np.mean((clean_proba[mask] >= 0.5) == y_true[mask]))
    acc_corr = float(np.mean((corr_proba[mask] >= 0.5) == y_true[mask]))

    monitor = ModelTrustMonitor(**monitor_cfg)
    batches_until_flag, flagged = None, False
    batch = 20
    for i in range(0, len(corr_proba), batch):
        res = monitor.update(corr_proba[i:i + batch])
        if res.degraded and not flagged:
            flagged = True
            batches_until_flag = i // batch + 1
    return {
        "seed": seed, "subject": subj, "faults": corrupt,
        "n_windows": len(corr_proba),
        "true_acc_clean": acc_clean, "true_acc_corrupted": acc_corr,
        "true_acc_drop": acc_clean - acc_corr,
        "monitor_flagged": flagged,
        "batches_until_flag": batches_until_flag,
        "truth": truth,
    }


def main():
    print("[layer2] building labeled windows...", flush=True)
    Xw_tr, X_tr, y_tr = build_labeled(TRAIN_SUBJ)
    Xw_va, X_va, y_va = build_labeled(VAL_SUBJ)
    Xw_te, X_te, y_te = build_labeled(TEST_SUBJS[0])
    for s in TEST_SUBJS[1:]:
        _, Xi, yi = build_labeled(s)
        X_te = pd.concat([X_te, Xi], ignore_index=True)
        y_te = np.concatenate([y_te, yi])
    print(f"[layer2] train={len(y_tr)} val={len(y_va)} test={len(y_te)} "
          f"pos_rate={y_tr.mean():.2f}/{y_va.mean():.2f}/{y_te.mean():.2f}",
          flush=True)

    print("[layer2] training stress model...", flush=True)
    model = train_stress_model(X_tr, y_tr, seed=SEED)
    save_model(model, os.path.join(RESULTS, "stress_model.pkl"))
    metrics = evaluate(model, X_te, y_te)
    print("[layer2] test metrics: " + json.dumps(metrics), flush=True)

    val_proba = model.predict_proba(X_va)[:, 1]
    monitor_cfg = {"model_id": "wesad-stress-rf",
                   "reference_proba": val_proba}

    print("[layer2] degradation runs...", flush=True)
    deg = []
    scenarios = [
        [{"type": "gradual_drift", "channel": "ecg", "params": {"rate": 0.05}},
         {"type": "gradual_drift", "channel": "eda_c", "params": {"rate": 0.05}}],
        [{"type": "random_dropout", "channel": "ecg", "params": {"p": 0.3}},
         {"type": "random_dropout", "channel": "eda_c", "params": {"p": 0.3}}],
        [{"type": "stuck_at", "channel": "bvp", "params": {}}],
    ]
    for i, corrupt in enumerate(scenarios):
        r = degradation_run(model, monitor_cfg, corrupt, seed=42000 + i)
        deg.append(r)
        print(f"[layer2] scenario {i}: flagged={r['monitor_flagged']} "
              f"batches={r['batches_until_flag']} "
              f"acc {r['true_acc_clean']:.3f}->{r['true_acc_corrupted']:.3f}",
              flush=True)

    out = {"test_metrics": metrics, "degradation_runs": deg,
           "n_train": len(y_tr), "n_val": len(y_va), "n_test": len(y_te)}
    with open(os.path.join(RESULTS, "layer2.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    return out


if __name__ == "__main__":
    main()
