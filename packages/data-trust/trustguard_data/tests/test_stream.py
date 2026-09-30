"""Unit tests for trustguard_data.stream."""

import numpy as np
import pytest

from trustguard_data.stream import Window, split_stream, validate_window


def _window(**kwargs):
    base = {
        "device_id": "dev-1",
        "start_s": 0.0,
        "end_s": 10.0,
        "channels": {"ecg": np.zeros(500), "bvp": np.ones(250)},
        "fs": {"ecg": 50.0, "bvp": 25.0},
    }
    base.update(kwargs)
    return Window(**base)


def test_window_valid_and_duration():
    w = _window()
    validate_window(w)
    assert w.duration_s == 10.0
    assert w.channels["ecg"].dtype == np.float64


def test_window_rejects_bad_bounds():
    with pytest.raises(ValueError):
        _window(end_s=0.0)
    with pytest.raises(ValueError):
        _window(device_id="")


def test_window_rejects_fs_mismatch():
    with pytest.raises(ValueError):
        _window(fs={"ecg": 50.0})
    with pytest.raises(ValueError):
        _window(fs={"ecg": 0.0, "bvp": 25.0})
    with pytest.raises(ValueError):
        _window(fs={"ecg": -5.0, "bvp": 25.0})


def test_window_rejects_empty_channels():
    with pytest.raises(ValueError):
        _window(channels={})
    with pytest.raises(ValueError):
        _window(channels={"ecg": np.array([]), "bvp": np.ones(250)})


def test_window_rejects_non_1d():
    with pytest.raises(ValueError):
        _window(channels={"ecg": np.zeros((500, 2)), "bvp": np.ones(250)})


def test_validate_window_type_error():
    with pytest.raises(TypeError):
        validate_window({"not": "a window"})


def test_split_stream_multi_rate():
    rng = np.random.default_rng(0)
    channels = {
        "ecg": rng.standard_normal(1000),  # 10 s @ 100 Hz
        "bvp": rng.standard_normal(500),  # 10 s @ 50 Hz
    }
    windows = split_stream("dev-9", channels, {"ecg": 100.0, "bvp": 50.0},
                           window_s=2.0)
    assert len(windows) == 5
    assert windows[0].start_s == 0.0 and windows[0].end_s == 2.0
    assert windows[-1].start_s == 8.0 and windows[-1].end_s == 10.0
    for w in windows:
        assert w.channels["ecg"].shape == (200,)
        assert w.channels["bvp"].shape == (100,)
        assert w.device_id == "dev-9"


def test_split_stream_drops_trailing_samples():
    channels = {"a": np.zeros(105), "b": np.zeros(105)}
    windows = split_stream("d", channels, {"a": 10.0, "b": 10.0}, window_s=5.0)
    assert len(windows) == 2  # 5 leftover samples dropped


def test_split_stream_start_offset():
    channels = {"a": np.zeros(100)}
    windows = split_stream("d", channels, {"a": 10.0}, window_s=5.0, start_s=100.0)
    assert windows[0].start_s == 100.0
    assert windows[1].end_s == 110.0


def test_split_stream_errors():
    with pytest.raises(ValueError):
        split_stream("d", {"a": np.zeros(5)}, {"a": 10.0}, window_s=5.0)
    with pytest.raises(ValueError):
        split_stream("d", {"a": np.zeros(100)}, {"a": 10.0}, window_s=0.0)
    with pytest.raises(ValueError):
        split_stream("", {"a": np.zeros(100)}, {"a": 10.0}, window_s=1.0)
    with pytest.raises(ValueError):
        split_stream("d", {"a": np.zeros(100)}, {"b": 10.0}, window_s=1.0)
