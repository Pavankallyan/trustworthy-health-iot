"""Ablation grid (WESAD): is each layer / fusion actually needed?

Pre-registered in experiment_plan.md (2026-09-30), adapted by addenda C/E:
1. Layer 1 only — fault coverage without model monitoring.
2. Layer 2 only — stress-model monitor on corrupted inputs, no Layer 1.
   The monitor is label-free: it can only see corruptions that move the
   model's outputs. Sensor faults that leave predictions unchanged are
   (correctly) invisible to it — the honest, measured result.
3. Full stack — union of the layers.
4. Layer 1 without distribution drift detectors (KS/PSI thresholds set to
   +inf) — do the distribution tests carry the drift/quant detections?
   (Replaces the pre-registered "no cross-sensor consistency" cell, which is
   moot: consistency is disabled in the real experiments per addendum C.)
5. Fused score vs independent alerts — per-run fused trust AUC (12 corrupted
   + 12 clean windows) vs the binary OR of the layers' alerts.

Replays the EXACT 12 injection runs from run_layer1 (same order, subjects,
windows, targets, seeds). The Layer 2 reference is each subject's own recent
clean val windows (a fixed early reference goes stale as EDA/temperature
drift — the deployment-realistic setup).
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import replace

import numpy as np
from sklearn.metrics import auc, roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_layer1 as R  # noqa: E402
from run_layer2 import FEATURE_CHANNELS, build_labeled  # noqa: E402
from trustguard_data.pipeline import DataTrustPipeline  # noqa: E402
from trustguard_data.stream import Window  # noqa: E402
from trustguard_data.taxonomy import HealthState  # noqa: E402
from trustguard_fusion.fusion import fuse_trust  # noqa: E402
from trustguard_model.baseline import train_stress_model  # noqa: E402
from trustguard_model.features import extract_features  # noqa: E402
from trustguard_model.monitors import ModelTrustMonitor  # noqa: E402
from common import SEED  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)


def main():
    # ---- Layer 1 pipelines (per subject, as in run_layer1) ----
    pipes, pipes_no_kspsi, subj = {}, {}, {}
    for s in R.TEST_SUBJECTS:
        train_w, val_w, test_w = R.subject_windows(s)
        pipe, ranges, thresholds, _cal = R.build_pipeline(
            train_w, val_w, R.TARGETS, seed=SEED)
        pipes[s] = pipe
        subj[s] = {"train": train_w, "val": val_w, "test": test_w,
                   "ranges": ranges, "thresholds": thresholds}
        t2 = dict(thresholds)
        t2.update({"ks_min_d": float("inf"), "psi_drift": float("inf"),
                   "psi_watch": float("inf")})
        p2 = DataTrustPipeline(config={"ranges": ranges, "thresholds": t2})
        p2.fit_baseline(train_w, seed=SEED)
        pipes_no_kspsi[s] = p2

    # ---- Layer 2: stress model + per-subject recent reference ----
    print("[ablation] training stress model...", flush=True)
    _, X_tr, y_tr = build_labeled("S2")
    model = train_stress_model(X_tr, y_tr, seed=SEED)
    ref_proba_map = {}
    for s in R.TEST_SUBJECTS:
        pv = model.predict_proba(
            extract_features(subj[s]["val"], FEATURE_CHANNELS))[:, 1]
        ref_proba_map[s] = pv
        print(f"[ablation] reference proba {s}: n={len(pv)} "
              f"mean={pv.mean():.4f}", flush=True)

    def l2_batch(s, windows):
        """Monitor (flag, health, result) for a batch vs subject reference."""
        p = model.predict_proba(
            extract_features(windows, FEATURE_CHANNELS))[:, 1]
        mon = ModelTrustMonitor(model_id="wesad-stress-rf",
                                reference_proba=ref_proba_map[s])
        res = mon.update(p)
        return res.degraded, res.health, res

    def fused_trust(s, target, window, l2_res):
        rr1 = pipes[s].score_window(window)
        tch = rr1.channels[target]
        r1t = replace(rr1, channels={target: tch}, layer_score=tch.score)
        rep = fuse_trust(f"wesad-{s}", r1t, l2_res, timestamp_s=0.0)
        return rep.trust_score, tch.state != HealthState.HEALTHY

    rows = []
    for i, (ftype, params) in enumerate(R.FAULTS):
        s = R.TEST_SUBJECTS[i % len(R.TEST_SUBJECTS)]
        w = subj[s]["test"][(i // len(R.TEST_SUBJECTS)) % len(subj[s]["test"])]
        target = R.TARGETS[i % len(R.TARGETS)]
        win = Window(device_id=w.device_id, start_s=0.0, end_s=R.WINDOW_S,
                     channels=dict(w.channels), fs=dict(w.fs))
        injected, _truth = R.inject_fault(win, ftype, params, target,
                                          subj[s]["ranges"], seed=42000 + i)

        # Cell 1: Layer 1 only (target channel)
        r1 = pipes[s].score_window(injected)
        l1_det = r1.channels[target].state != HealthState.HEALTHY
        # Cell 4: Layer 1 without KS/PSI distribution detectors
        r4 = pipes_no_kspsi[s].score_window(injected)
        noks_det = r4.channels[target].state != HealthState.HEALTHY

        # Cell 2: Layer 2 only — batch of 5 clean val + this window
        l2_det_corr, _, l2_res_corr = l2_batch(
            s, list(subj[s]["val"]) + [injected])
        l2_det_clean, _, l2_res_clean = l2_batch(
            s, list(subj[s]["val"]) + [win])
        l2_det = bool(l2_det_corr)
        l2_clean_fp = int(l2_det_clean)

        # Cell 3: full stack
        full = bool(l1_det) or l2_det

        # Cell 5 items: fused trust + independent OR, corrupted and clean
        trust_corr, l1f_corr = fused_trust(s, target, injected, l2_res_corr)
        trust_clean, l1f_clean = fused_trust(s, target, win, l2_res_clean)

        rows.append({
            "run": i, "subject": s, "fault": ftype,
            "params": params, "target": target, "seed": 42000 + i,
            "l1_only": bool(l1_det), "l2_only": l2_det,
            "full_stack": full, "no_kspsi": bool(noks_det),
            "l2_clean_false_alarm": l2_clean_fp,
            "trust_corrupted": trust_corr, "trust_clean": trust_clean,
            "indep_corrupted": bool(l1f_corr or l2_det_corr),
            "indep_clean": bool(l1f_clean or l2_det_clean),
        })
        print(f"[ablation] run {i} ({s} {ftype} {params} on {target}): "
              f"L1={bool(l1_det)} L2={l2_det} full={full} "
              f"noKSPSI={bool(noks_det)}", flush=True)

    def cov(key):
        vals = [r[key] for r in rows]
        return sum(vals) / len(vals) if vals else 0.0

    # Cell 5: 24 items (12 corrupted y=1, 12 clean y=0)
    y_true = [1] * len(rows) + [0] * len(rows)
    fused_scores = ([-r["trust_corrupted"] for r in rows]
                    + [-r["trust_clean"] for r in rows])
    indep_scores = ([int(r["indep_corrupted"]) for r in rows]
                    + [int(r["indep_clean"]) for r in rows])
    fpr, tpr, _ = roc_curve(y_true, fused_scores)
    fused_auc = float(auc(fpr, tpr))
    fpr2, tpr2, _ = roc_curve(y_true, indep_scores)
    indep_auc = float(auc(fpr2, tpr2))
    fp_indep = sum(1 for y, f in zip(y_true, indep_scores) if f and not y)

    out = {
        "n_runs": len(rows),
        "coverage": {
            "layer1_only": cov("l1_only"),
            "layer2_only": cov("l2_only"),
            "full_stack": cov("full_stack"),
            "no_kspsi": cov("no_kspsi"),
        },
        "layer2_clean_false_alarms": sum(r["l2_clean_false_alarm"]
                                         for r in rows),
        "cell5_n_items": len(y_true),
        "cell5_note": ("fused AUC ranks 12 corrupted vs 12 clean windows by "
                       "trust score; independent AUC uses the binary OR of "
                       "the layers' alerts"),
        "fused_auc": fused_auc,
        "independent_auc": indep_auc,
        "independent_or_false_positives": fp_indep,
    }
    with open(os.path.join(RESULTS, "ablations.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    with open(os.path.join(RESULTS, "ablations_runs.json"), "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print("[ablation] " + json.dumps(out, default=str), flush=True)
    return out


if __name__ == "__main__":
    main()
