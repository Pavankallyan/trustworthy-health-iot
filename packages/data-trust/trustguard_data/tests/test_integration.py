"""Integration test: full Layer 1 pipeline on a synthetic 3-channel stream
with an injected stuck-at fault. The fault must be detected and classified.
"""

import numpy as np

from trustguard_data.injection import FaultInjector
from trustguard_data.pipeline import DataTrustPipeline
from trustguard_data.stream import split_stream
from trustguard_data.taxonomy import HealthState


def _synthetic_stream(seed=42):
    rng = np.random.default_rng(seed)
    fs = 50.0
    seconds = 60
    n = int(seconds * fs)
    t = np.arange(n) / fs

    # ECG-like: 1 Hz QRS spikes on a small baseline wander + noise (60 bpm).
    ecg = 0.15 * np.sin(2 * np.pi * 1.0 * t) + 0.02 * rng.standard_normal(n)
    for k in range(seconds):
        ecg[k * int(fs) + 10] += 1.0

    # BVP-like: 1 Hz pulsatile wave (60 bpm), phase-shifted.
    bvp = 0.4 * np.sin(2 * np.pi * 1.0 * t - 0.5) + 0.02 * rng.standard_normal(n)

    # ACC magnitude-like: slow movement + noise.
    acc = 0.5 * np.sin(2 * np.pi * 0.3 * t) + 0.1 * rng.standard_normal(n)

    return {"ecg": ecg, "bvp": bvp, "acc": acc}, {"ecg": fs, "bvp": fs, "acc": fs}


def test_full_pipeline_detects_injected_stuck_fault():
    channels, fs = _synthetic_stream()
    windows = split_stream("watch-01", channels, fs, window_s=10.0)
    assert len(windows) == 6

    config = {
        "ranges": {"ecg": (-1.5, 1.5), "bvp": (-1.0, 1.0), "acc": (-1.5, 1.5)},
        "schema": {"ecg": ("f", 100), "bvp": ("f", 100), "acc": ("f", 100)},
        "hr_pair": ("ecg", "bvp"),
    }
    pipe = DataTrustPipeline(config)
    pipe.fit_baseline(windows[:3], seed=0)  # clean windows only

    # Inject a full-window stuck-at fault on ecg in window 3 (20-30 s).
    injector = FaultInjector(seed=7)
    faulty, truth = injector.inject(
        windows[3],
        [{"type": "stuck_at", "channel": "ecg",
          "start": 0.0, "end": 10.0, "params": {}}],
    )
    assert truth[0]["channel"] == "ecg"
    assert truth[0]["start_sample"] == 0 and truth[0]["end_sample"] == 500

    scored = windows[:3] + [faulty] + windows[4:]
    results = pipe.score_stream(scored)
    assert len(results) == 6

    # The injected fault is detected and classified as STUCK.
    faulty_result = results[3]
    assert faulty_result.channels["ecg"].state == HealthState.STUCK
    assert faulty_result.channels["ecg"].score < 50.0

    # Cross-channel check: ecg flat -> HR disagreement flags bvp too.
    assert faulty_result.channels["bvp"].state == HealthState.DEGRADED

    # Clean windows stay healthy and score higher than the faulty one.
    for r in results[:3] + results[4:]:
        assert all(ch.state == HealthState.HEALTHY
                   for ch in r.channels.values()), \
            f"window {r.start_s}s not healthy"
    assert faulty_result.layer_score < min(
        r.layer_score for r in results[:3] + results[4:])
