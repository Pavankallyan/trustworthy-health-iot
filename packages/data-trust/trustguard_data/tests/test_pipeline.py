"""Unit tests for trustguard_data.pipeline.DataTrustPipeline."""

import numpy as np
import pytest

from trustguard_data.pipeline import DataTrustPipeline
from trustguard_data.stream import Window, split_stream
from trustguard_data.taxonomy import HealthState


def _clean_windows(seed=0, n_windows=4):
    rng = np.random.default_rng(seed)
    channels = {
        "ecg": rng.standard_normal(4000),
        "bvp": rng.standard_normal(2000) * 0.5,
    }
    return split_stream("dev-1", channels,
                        {"ecg": 100.0, "bvp": 50.0}, window_s=10.0)[:n_windows]


def _config():
    return {
        "ranges": {"ecg": (-5.0, 5.0), "bvp": (-5.0, 5.0)},
        "schema": {"ecg": ("f", 100), "bvp": ("f", 100)},
    }


def test_fit_baseline_and_state():
    pipe = DataTrustPipeline(_config())
    assert pipe.is_fitted is False
    pipe.fit_baseline(_clean_windows(), seed=0)
    assert pipe.is_fitted is True
    assert pipe.baseline_channels == ["bvp", "ecg"]


def test_fit_baseline_deterministic_subsample():
    wins = _clean_windows()
    p1 = DataTrustPipeline(_config())
    p1.fit_baseline(wins, seed=3)
    p2 = DataTrustPipeline(_config())
    p2.fit_baseline(wins, seed=3)
    for ch in p1.baseline_channels:
        assert np.array_equal(p1._baselines[ch], p2._baselines[ch])


def test_fit_baseline_rejects_bad_input():
    pipe = DataTrustPipeline(_config())
    with pytest.raises(ValueError):
        pipe.fit_baseline([])
    with pytest.raises(ValueError):
        # all-NaN channels: nothing usable
        w = Window(device_id="d", start_s=0.0, end_s=10.0,
                   channels={"ecg": np.full(100, np.nan)},
                   fs={"ecg": 10.0})
        pipe.fit_baseline([w])


def test_score_healthy_window():
    pipe = DataTrustPipeline(_config())
    pipe.fit_baseline(_clean_windows(), seed=0)
    res = pipe.score_window(_clean_windows(seed=99)[0])
    assert res.device_id == "dev-1"
    assert res.start_s == 0.0 and res.end_s == 10.0
    for ch in res.channels.values():
        assert ch.state == HealthState.HEALTHY
    assert res.layer_score == pytest.approx(100.0)


def test_score_without_baseline_still_works():
    pipe = DataTrustPipeline(_config())
    res = pipe.score_window(_clean_windows()[0])
    assert all(ch.state == HealthState.HEALTHY
               for ch in res.channels.values())


def test_missing_expected_channel():
    pipe = DataTrustPipeline({**_config(),
                              "schema": {"ecg": ("f", 100),
                                         "spo2": ("f", 10)}})
    res = pipe.score_window(_clean_windows()[0])
    assert res.channels["spo2"].state == HealthState.MISSING
    assert res.channels["spo2"].score == 0.0
    assert res.layer_score < 100.0


def test_score_stream_returns_list():
    pipe = DataTrustPipeline(_config())
    results = pipe.score_stream(_clean_windows())
    assert len(results) == 4
    assert [r.start_s for r in results] == [0.0, 10.0, 20.0, 30.0]


def test_score_stream_rejects_empty():
    with pytest.raises(ValueError):
        DataTrustPipeline(_config()).score_stream([])


def test_bad_config_rejected():
    with pytest.raises(ValueError):
        DataTrustPipeline({"ranges": {"ecg": (5.0, -5.0)}})
    with pytest.raises(ValueError):
        DataTrustPipeline({"thresholds": {"nope": 1.0}})
    with pytest.raises(TypeError):
        DataTrustPipeline(config="nope")
    with pytest.raises(ValueError):
        DataTrustPipeline({"hr_pair": ("only-one",)})
