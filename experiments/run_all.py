"""Orchestrator: run the full experiment suite and build the summary.

Order: layer1 -> baselines -> layer2 -> ablations -> summary + dashboard.
Every step writes JSON under experiments/results/. Fixed seeds everywhere
(see experiment_plan.md, pre-registered 2026-09-30 with addenda A-F).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)


def main():
    import run_layer1
    import run_baselines
    import run_layer2
    import run_ablations

    print("=" * 60 + "\n[1/4] Layer 1 fault-injection evaluation\n" + "=" * 60,
          flush=True)
    l1 = run_layer1.main()

    print("=" * 60 + "\n[2/4] Baselines\n" + "=" * 60, flush=True)
    bl = run_baselines.main()

    print("=" * 60 + "\n[3/4] Layer 2 stress model + degradation\n" + "=" * 60,
          flush=True)
    l2 = run_layer2.main()

    print("=" * 60 + "\n[4/4] Ablation grid\n" + "=" * 60, flush=True)
    ab = run_ablations.main()

    summary = build_summary(l1, bl, l2, ab)
    with open(os.path.join(RESULTS, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print("\n" + "=" * 60 + "\nHEADLINE RESULTS\n" + "=" * 60)
    print(json.dumps(summary["headline"], indent=1, default=str))

    print("\n[dashboard] rendering from real scored windows...", flush=True)
    render_dashboard()
    print("[done] results in", RESULTS, flush=True)
    return summary


def build_summary(l1, bl, l2, ab) -> dict:
    headline = {}
    rec = l1["recall_by_fault"]
    headline["layer1"] = {
        "dataset": "wesad",
        "subjects": l1["subjects"],
        "n_runs": len(l1["runs"]),
        "overall_recall": round(l1["overall_recall"], 4),
        "recall_by_fault": {
            k: round(v["recall"], 4) for k, v in rec.items()},
        "val_false_alarm_rate": {
            s: round(f["far"], 4) for s, f in l1["false_alarm"].items()},
    }
    base_rec = {k: round(v["recall"], 4)
                for k, v in bl["baselines"].items()}
    headline["baselines"] = {
        "trustguard_recall": round(l1["overall_recall"], 4),
        "baseline_recall": base_rec,
    }
    tm = l2["test_metrics"]
    headline["layer2_stress_model"] = {
        "n_train": l2["n_train"], "n_val": l2["n_val"], "n_test": l2["n_test"],
        "accuracy": round(tm["accuracy"], 4),
        "f1": round(tm["f1"], 4),
        "auc": round(tm["auc"], 4),
        "ece": round(tm["ece"], 4),
    }
    headline["layer2_degradation"] = [
        {"faults": [f["type"] + ":" + f["channel"] for f in r["faults"]],
         "flagged": r["monitor_flagged"],
         "batches_until_flag": r["batches_until_flag"],
         "true_acc_clean": round(r["true_acc_clean"], 4),
         "true_acc_corrupted": round(r["true_acc_corrupted"], 4),
         "true_acc_drop": round(r["true_acc_drop"], 4)}
        for r in l2["degradation_runs"]
    ]
    headline["ablations"] = {
        "coverage": {k: round(v, 4) for k, v in ab["coverage"].items()},
        "layer2_clean_false_alarms": ab["layer2_clean_false_alarms"],
        "fused_auc": round(ab["fused_auc"], 4),
        "independent_auc": round(ab["independent_auc"], 4),
    }
    return {"headline": headline,
            "layer1": l1, "baselines": bl,
            "layer2": l2, "ablations": ab}


def render_dashboard():
    """Build the dashboard from real scored windows (not synthetic).

    Two test devices (wesad-S4, wesad-S6): their 6 test windows each, with
    one injected fault per device at a known position. Every window is
    scored by the real Layer 1 pipeline and fused with a real Layer 2
    monitor batch result; alerts are routed from those real reports.
    """
    import run_layer1 as R
    from run_layer2 import FEATURE_CHANNELS, build_labeled
    from trustguard_data.stream import Window
    from trustguard_fusion.alerts import route_alerts
    from trustguard_fusion.dashboard import render_html
    from trustguard_fusion.fusion import fuse_trust
    from trustguard_model.baseline import train_stress_model
    from trustguard_model.features import extract_features
    from trustguard_model.monitors import ModelTrustMonitor
    from common import SEED

    # stress model + per-subject references (as in run_ablations)
    _, X_tr, y_tr = build_labeled("S2")
    model = train_stress_model(X_tr, y_tr, seed=SEED)

    demos = [
        ("S4", "gradual_drift", {"rate": 0.10}, "ecg", 2),
        ("S6", "stuck_at", {"value": "median"}, "bvp", 3),
    ]
    reports, alerts, trend = [], [], {}
    for subj, ftype, params, target, fault_pos in demos:
        train_w, val_w, test_w = R.subject_windows(subj)
        pipe, ranges, _thresholds, _cal = R.build_pipeline(
            train_w, val_w, R.TARGETS, seed=SEED)
        ref_proba = model.predict_proba(
            extract_features(val_w, FEATURE_CHANNELS))[:, 1]

        windows = []
        for k, w in enumerate(test_w):
            win = Window(device_id=w.device_id, start_s=k * R.WINDOW_S,
                         end_s=(k + 1) * R.WINDOW_S,
                         channels=dict(w.channels), fs=dict(w.fs))
            if k == fault_pos:
                win, _truth = R.inject_fault(win, ftype, params, target,
                                             ranges, seed=9000 + k)
            windows.append(win)

        from dataclasses import replace
        scores = []
        for k, win in enumerate(windows):
            l1 = pipe.score_window(win)
            # dashboard focuses on the monitored vital-sign channels
            # (ECG/BVP), as do the reported metrics; slow channels
            # (EDA/temperature) drift naturally at rest.
            tch = {t: l1.channels[t] for t in R.TARGETS}
            l1t = replace(
                l1, channels=tch,
                layer_score=float(np.mean([c.score for c in tch.values()])))
            p = model.predict_proba(
                extract_features(list(val_w) + [win],
                                 FEATURE_CHANNELS))[:, 1]
            mon = ModelTrustMonitor(model_id="wesad-stress-rf",
                                    reference_proba=ref_proba)
            l2 = mon.update(p)
            rep = fuse_trust(f"wesad-{subj}", l1t, l2,
                             timestamp_s=win.start_s)
            scores.append(rep.trust_score)
            alerts.extend(route_alerts(rep, l1=l1t, l2=l2))
            if k == len(windows) - 1:
                reports.append(rep)
        trend[f"wesad-{subj}"] = scores
        print(f"[dashboard] wesad-{subj}: trust trend="
              f"{[round(s, 1) for s in scores]}", flush=True)

    html_out = render_html(
        reports, alerts, trend=trend,
        title="TrustGuard-IoT — Live Experiment Dashboard "
              "(real WESAD windows, injected faults)")
    with open(os.path.join(RESULTS, "dashboard.html"), "w") as f:
        f.write(html_out)


if __name__ == "__main__":
    main()
