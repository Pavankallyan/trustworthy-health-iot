"""Unit tests for trustguard_data.injection."""

import numpy as np
import pytest

from trustguard_data.injection import (
    FAULT_TYPES,
    FaultInjector,
    InjectionResult,
)
from trustguard_data.stream import Window
from trustguard_data.validators import check_range


def _window(seed=0):
    rng = np.random.default_rng(seed)
    return Window(
        device_id="dev-1",
        start_s=0.0,
        end_s=10.0,
        channels={
            "ecg": rng.standard_normal(1000),
            "bvp": rng.standard_normal(500) + 1.0,
        },
        fs={"ecg": 100.0, "bvp": 50.0},
    )


def test_fault_types_complete():
    assert set(FAULT_TYPES) == {
        "gradual_drift", "stuck_at", "random_dropout",
        "quantization_noise", "desync",
    }


def _inject(**fault_kwargs):
    injector = FaultInjector(seed=7)
    fault = {"type": "stuck_at", "channel": "ecg",
             "start": 2.0, "end": 8.0, "params": {}}
    fault.update(fault_kwargs)
    return injector.inject(_window(), [fault])


def test_stuck_at_applies_and_logs():
    res = _inject()
    assert isinstance(res, InjectionResult)
    new_window, truth = res  # tuple unpacking per SPEC
    assert len(truth) == 1
    g = truth[0]
    assert g["type"] == "stuck_at" and g["channel"] == "ecg"
    assert g["start_sample"] == 200 and g["end_sample"] == 800
    seg = new_window.channels["ecg"][200:800]
    assert np.all(seg == seg[0])


def test_gradual_drift_ramp():
    injector = FaultInjector(seed=7)
    w = _window()
    before = w.channels["ecg"].copy()
    res = injector.inject(w, [{"type": "gradual_drift", "channel": "ecg",
                               "start": 0.0, "end": 10.0,
                               "params": {"magnitude": 2.0}}])
    after = res.window.channels["ecg"]
    diff = after - before
    assert diff[0] == pytest.approx(0.0, abs=1e-9)
    # magnitude is clipped to observed max; ramp must still be increasing
    assert diff[-1] > diff[len(diff) // 2] > 0


def test_random_dropout_fraction():
    injector = FaultInjector(seed=7)
    res = injector.inject(_window(), [{"type": "random_dropout",
                                       "channel": "ecg", "start": 0.0,
                                       "end": 10.0,
                                       "params": {"fraction": 0.3}}])
    x = res.window.channels["ecg"]
    assert np.mean(np.isnan(x)) == pytest.approx(0.3, abs=0.02)
    assert res.ground_truth[0]["start_sample"] == 0
    assert res.ground_truth[0]["end_sample"] == 1000


def test_quantization_noise_levels():
    injector = FaultInjector(seed=7)
    res = injector.inject(_window(), [{"type": "quantization_noise",
                                       "channel": "ecg", "start": 0.0,
                                       "end": 10.0,
                                       "params": {"n_levels": 4}}])
    seg = res.window.channels["ecg"]
    assert len(np.unique(np.round(seg, 10))) <= 5  # <= n_levels + rounding


def test_desync_shift():
    injector = FaultInjector(seed=7)
    w = _window()
    before = w.channels["ecg"].copy()
    res = injector.inject(w, [{"type": "desync", "channel": "ecg",
                               "start": 1.0, "end": 9.0,
                               "params": {"shift_s": 0.5}}])
    after = res.window.channels["ecg"]
    # 50-sample shift inside [100, 900)
    assert np.allclose(after[150:900], before[100:850])
    assert res.ground_truth[0]["params"] == {"shift_s": 0.5}


def test_deterministic_with_seed():
    faults = [{"type": "random_dropout", "channel": "ecg",
               "start": 0.0, "end": 10.0, "params": {"fraction": 0.3}}]
    a = FaultInjector(seed=11).inject(_window(), faults).window
    b = FaultInjector(seed=11).inject(_window(), faults).window
    assert np.array_equal(a.channels["ecg"], b.channels["ecg"],
                          equal_nan=True)


def test_original_window_untouched():
    w = _window()
    before = w.channels["ecg"].copy()
    FaultInjector(seed=7).inject(w, [{"type": "stuck_at", "channel": "ecg",
                                     "start": 0.0, "end": 10.0,
                                     "params": {}}])
    assert np.array_equal(w.channels["ecg"], before)


def test_injections_pass_range_checks():
    # Plausible faults must not trip plain range checks built from the
    # channel's own observed min/max.
    w = _window()
    ranges = {}
    for ch, arr in w.channels.items():
        valid = arr[~np.isnan(arr)]
        ranges[ch] = (float(np.min(valid)), float(np.max(valid)))
    injector = FaultInjector(seed=7)
    faults = [
        {"type": t, "channel": "ecg", "start": 1.0, "end": 9.0, "params": {}}
        for t in FAULT_TYPES
    ]
    res = injector.inject(w, faults)
    for r in check_range(res.window, ranges):
        assert r["ok"] is True, f"{r['channel']} failed range check"


def test_bad_faults_rejected():
    injector = FaultInjector(seed=7)
    w = _window()
    with pytest.raises(ValueError):
        injector.inject(w, [{"type": "nope", "channel": "ecg",
                             "start": 0.0, "end": 1.0, "params": {}}])
    with pytest.raises(ValueError):
        injector.inject(w, [{"type": "stuck_at", "channel": "ghost",
                             "start": 0.0, "end": 1.0, "params": {}}])
    with pytest.raises(ValueError):
        injector.inject(w, [{"type": "stuck_at", "channel": "ecg",
                             "start": 5.0, "end": 5.0, "params": {}}])
    with pytest.raises(ValueError):
        injector.inject(w, [{"type": "stuck_at", "channel": "ecg",
                             "start": -1.0, "end": 5.0, "params": {}}])
    with pytest.raises(ValueError):
        injector.inject(w, [{"type": "stuck_at", "channel": "ecg",
                             "start": 0.0, "end": 11.0, "params": {}}])
    with pytest.raises(ValueError):
        injector.inject(w, [])
    with pytest.raises(ValueError):
        injector.inject(w, [{"type": "random_dropout", "channel": "ecg",
                             "start": 0.0, "end": 1.0,
                             "params": {"fraction": 1.5}}])


def test_bad_seed_rejected():
    with pytest.raises(TypeError):
        FaultInjector(seed="seven")
