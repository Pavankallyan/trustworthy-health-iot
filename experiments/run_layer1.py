"""Layer 1 real-data experiments: fault injection + detection on WESAD.

Pre-registered plan: experiments/experiment_plan.md (incl. addenda A-E).
Final scope (addendum E): WESAD only, condition-locked (resting baseline,
minutes 4-23), per-device splits, test subjects S4/S6. Faults injected on
ECG and wrist BVP; metrics count target-channel states only.
Master seed 42; per-run seeds 42000 + run_index.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "data-trust"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

import common
from trustguard_data.injection import FaultInjector
from trustguard_data.pipeline import DataTrustPipeline
from trustguard_data.stream import Window
from trustguard_data.taxonomy import HealthState

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)

WESAD_STREAM_CHANNELS = [
    "ecg", "eda_c", "emg", "resp", "temp_c",
    "bvp", "eda_w",
    "acc_c_x", "acc_c_y", "acc_c_z",
    "acc_w_x", "acc_w_y", "acc_w_z",
]

TARGETS = ["ecg", "bvp"]
WINDOW_S = 60.0
# Resting-baseline condition: seconds [240, 1380) -> 19 windows of 60 s.
BASE_START_S = 240.0
BASE_END_S = 1380.0
TRAIN_WIN = (0, 8)
VAL_WIN = (8, 13)
TEST_WIN = (13, 19)
TEST_SUBJECTS = ["S4", "S6"]

# 4 fault types x 3 levels = 12 runs (desync dropped: addendum C).
FAULTS: list[tuple[str, dict]] = [
    ("gradual_drift", {"rate": 0.02}),
    ("gradual_drift", {"rate": 0.05}),
    ("gradual_drift", {"rate": 0.10}),
    ("stuck_at", {"value": "onset"}),
    ("stuck_at", {"value": "median"}),
    ("stuck_at", {"value": "midrange"}),
    ("random_dropout", {"p": 0.10}),
    ("random_dropout", {"p": 0.30}),
    ("random_dropout", {"p": 0.50}),
    ("quantization_noise", {"n_levels": 4}),
    ("quantization_noise", {"n_levels": 8}),
    ("quantization_noise", {"n_levels": 16}),
]


def subject_windows(subject):
    """(train, val, test) windows for one subject, chronological."""
    stream = common.wesad_stream(subject, channels=WESAD_STREAM_CHANNELS,
                                 start_s=BASE_START_S,
                                 duration_s=BASE_END_S - BASE_START_S)
    wins = common.split_stream(stream, WINDOW_S, WINDOW_S)
    a, b = TRAIN_WIN
    c, d = VAL_WIN
    e, f = TEST_WIN
    return wins[a:b], wins[c:d], wins[e:f]


def build_pipeline(train_windows, val_windows, target_channels, seed=42):
    """Fit baseline on train; calibrate thresholds on clean val TARGETS.

    Returns (pipeline, ranges, thresholds, calibration_info). Calibration
    follows addendum B, restricted to the channels under test (addendum E):
    slow channels (EDA, temperature) drift naturally at rest and would
    otherwise force meaninglessly loose global thresholds.
    """
    channels = list(train_windows[0].channels.keys())
    ranges = common.empirical_ranges(train_windows, channels)

    pipe0 = DataTrustPipeline(config={"ranges": ranges})
    pipe0.fit_baseline(train_windows, seed=seed)

    max_d, max_psi, max_flat = 0.0, 0.0, 0.0
    n_val = 0
    for w in val_windows:
        res = pipe0.score_window(w)
        n_val += 1
        for name in target_channels:
            ch = res.channels[name]
            d = ch.details
            ks = d.get("ks") or {}
            if ks.get("statistic") is not None:
                max_d = max(max_d, float(ks["statistic"]))
            psiv = d.get("psi")
            if isinstance(psiv, (int, float)) and np.isfinite(psiv):
                max_psi = max(max_psi, float(psiv))
            stuck = d.get("stuck") or {}
            max_flat = max(max_flat, float(stuck.get("longest_stuck_s", 0.0) or 0.0))
    thresholds = {
        "ks_min_d": max(0.15, 1.25 * max_d),
        "psi_drift": max(0.30, 1.25 * max_psi),
        "psi_watch": max(0.15, 1.10 * max_psi),
        "stuck_min_duration_s": max(10.0, 2.0 * max_flat),
    }
    pipe = DataTrustPipeline(config={"ranges": ranges, "thresholds": thresholds})
    pipe.fit_baseline(train_windows, seed=seed)
    cal = {"n_val_windows": n_val, "max_clean_ks_d": max_d,
           "max_clean_psi": max_psi, "max_clean_flat_s": max_flat}
    return pipe, ranges, thresholds, cal


def _resolve_params(ftype, params, channel, stream, ranges):
    """Map plan-level fault params to the injector's param schema."""
    if ftype == "gradual_drift":
        lo, hi = ranges[channel]
        span_min = params["_span_s"] / 60.0
        return {"magnitude": params["rate"] * (hi - lo) * span_min}
    if ftype == "random_dropout":
        return {"fraction": params["p"]}
    if ftype == "quantization_noise":
        return {"n_levels": params["n_levels"]}
    if ftype == "stuck_at":
        v = params["value"]
        if v == "onset":
            return {}
        if v == "median":
            return {"value": float(np.nanmedian(stream.channels[channel]))}
        if v == "midrange":
            lo, hi = ranges[channel]
            return {"value": float((lo + hi) / 2.0)}
        raise ValueError(f"unknown stuck value {v!r}")
    raise ValueError(f"unknown fault {ftype!r}")


def inject_fault(window, ftype, params, target, ranges, seed):
    """Inject one fault into a single window; returns (window, ground_truth)."""
    span_s = window.end_s - window.start_s - 10.0
    params = dict(params)
    params["_span_s"] = span_s
    injector = FaultInjector(seed=seed, plausible_ranges=ranges)
    fault = {
        "type": ftype,
        "channel": target,
        "start": 5.0,
        "end": 5.0 + span_s,
        "params": _resolve_params(ftype, params, target, window, ranges),
    }
    res = injector.inject(window, [fault])
    return res.window, res.ground_truth


def score_detection(pipe, window, target):
    """Score one injected window; detection = target channel non-HEALTHY."""
    res = pipe.score_window(window)
    ch = res.channels[target]
    return {
        "detected": bool(ch.state != HealthState.HEALTHY),
        "state": ch.state.value,
        "channel_score": ch.score,
        "layer_score": res.layer_score,
    }


def false_alarm_rate(pipe, val_windows, target_channels):
    """Fraction of clean val windows with a non-HEALTHY TARGET channel."""
    flagged = 0
    per_channel: dict[str, int] = {}
    for w in val_windows:
        res = pipe.score_window(w)
        bad = [n for n in target_channels
               if res.channels[n].state != HealthState.HEALTHY]
        if bad:
            flagged += 1
        for n in bad:
            per_channel[n] = per_channel.get(n, 0) + 1
    n = len(val_windows)
    return {"far": flagged / n if n else 0.0, "n_windows": n,
            "per_channel_counts": per_channel}


def run(seed=42):
    pipes, ranges_map, thresholds_map, far_map, cal_map = {}, {}, {}, {}, {}
    test_windows = {}
    for s in TEST_SUBJECTS:
        train_w, val_w, test_w = subject_windows(s)
        print(f"[wesad/{s}] windows: train={len(train_w)} val={len(val_w)} "
              f"test={len(test_w)}", flush=True)
        pipe, ranges, thresholds, cal = build_pipeline(train_w, val_w, TARGETS,
                                                       seed=seed)
        pipes[s] = pipe
        test_windows[s] = test_w
        ranges_map[s] = ranges
        thresholds_map[s] = thresholds
        cal_map[s] = cal
        far = false_alarm_rate(pipe, val_w, TARGETS)
        far_map[s] = far
        print(f"[wesad/{s}] thresholds={thresholds}", flush=True)
        print(f"[wesad/{s}] val FAR (targets): {far['far']:.4f}", flush=True)

    runs = []
    for i, (ftype, params) in enumerate(FAULTS):
        s = TEST_SUBJECTS[i % len(TEST_SUBJECTS)]
        w = test_windows[s][(i // len(TEST_SUBJECTS)) % len(test_windows[s])]
        target = TARGETS[i % len(TARGETS)]
        win = Window(device_id=w.device_id, start_s=0.0, end_s=WINDOW_S,
                     channels=dict(w.channels), fs=dict(w.fs))
        injected, truth = inject_fault(win, ftype, params, target,
                                       ranges_map[s], seed=42000 + i)
        det = score_detection(pipes[s], injected, target)
        runs.append({
            "run_index": i, "seed": 42000 + i, "subject": s,
            "fault": ftype, "params": params, "target": target,
            "window_start_s": w.start_s, "device": w.device_id,
            "ground_truth": truth,
            **det,
        })
        print(f"[wesad] run {i} ({s}): {ftype} {params} on {target}: "
              f"detected={det['detected']} state={det['state']}", flush=True)

    by_type: dict[str, dict] = {}
    for r in runs:
        b = by_type.setdefault(r["fault"], {"n": 0, "detected": 0})
        b["n"] += 1
        b["detected"] += int(r["detected"])
    recall = {k: {"recall": v["detected"] / v["n"], **v} for k, v in by_type.items()}

    result = {
        "dataset": "wesad", "seed": seed,
        "subjects": TEST_SUBJECTS,
        "split": {"train_windows": list(TRAIN_WIN), "val_windows": list(VAL_WIN),
                   "test_windows": list(TEST_WIN),
                   "interval_s": [BASE_START_S, BASE_END_S],
                   "window_s": WINDOW_S, "targets": TARGETS},
        "thresholds": thresholds_map,
        "calibration": cal_map,
        "false_alarm": far_map,
        "recall_by_fault": recall,
        "overall_recall": sum(r["detected"] for r in runs) / len(runs),
        "runs": runs,
    }
    path = os.path.join(RESULTS, "layer1_wesad.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[wesad] overall recall: {result['overall_recall']:.3f} -> {path}",
          flush=True)
    with open(os.path.join(RESULTS, "layer1_summary.json"), "w") as f:
        json.dump({"overall_recall": result["overall_recall"],
                   "recall_by_fault": result["recall_by_fault"],
                   "false_alarm": result["false_alarm"],
                   "thresholds": result["thresholds"]}, f, indent=2, default=str)
    return result


def main():
    return run()


if __name__ == "__main__":
    main()
