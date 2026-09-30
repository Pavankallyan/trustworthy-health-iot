"""Unit tests for trustguard_data.validators."""

import numpy as np
import pytest

from trustguard_data.stream import Window
from trustguard_data.validators import check_range, check_schema


def _window():
    return Window(
        device_id="dev-1",
        start_s=0.0,
        end_s=10.0,
        channels={"ecg": np.zeros(500), "bvp": np.ones(250)},
        fs={"ecg": 50.0, "bvp": 25.0},
    )


def test_check_schema_ok():
    res = check_schema(_window(), {"ecg": ("f", 100), "bvp": ("f", 250)})
    by_ch = {r["channel"]: r for r in res}
    assert by_ch["ecg"]["ok"] is True
    assert by_ch["bvp"]["ok"] is True
    assert by_ch["ecg"]["dtype"] == "f"


def test_check_schema_missing_channel():
    res = check_schema(_window(), {"ecg": ("f", 100), "spo2": ("f", 10)})
    by_ch = {r["channel"]: r for r in res}
    assert by_ch["spo2"]["ok"] is False
    assert "missing" in by_ch["spo2"]["message"]


def test_check_schema_short_and_wrong_kind():
    w = _window()
    res = check_schema(w, {"ecg": ("f", 10_000)})
    assert res[0]["ok"] is False
    res = check_schema(w, {"ecg": ("i", 10)})
    assert res[0]["ok"] is False  # float64 kind is 'f', not 'i'


def test_check_schema_bad_spec():
    with pytest.raises(ValueError):
        check_schema(_window(), {})
    with pytest.raises(ValueError):
        check_schema(_window(), {"ecg": ("f",)})


def test_check_range_ok():
    res = check_range(_window(), {"ecg": (-1.0, 1.0), "bvp": (0.0, 2.0)})
    by_ch = {r["channel"]: r for r in res}
    assert by_ch["ecg"]["ok"] is True
    assert by_ch["ecg"]["fraction_out_of_range"] == 0.0


def test_check_range_violations():
    w = _window()
    w.channels["ecg"][0:50] = 99.0  # 10% out of range
    res = check_range(w, {"ecg": (-1.0, 1.0)})
    r = res[0]
    assert r["ok"] is False
    assert r["n_out"] == 50
    assert r["fraction_out_of_range"] == pytest.approx(0.1)


def test_check_range_ignores_nan():
    w = _window()
    w.channels["ecg"][0:100] = np.nan
    res = check_range(w, {"ecg": (-1.0, 1.0)})
    r = res[0]
    assert r["ok"] is True
    assert r["n_nan"] == 100
    assert r["n_total"] == 400


def test_check_range_missing_channel():
    res = check_range(_window(), {"spo2": (0.0, 100.0)})
    assert res[0]["ok"] is False
    assert res[0]["fraction_out_of_range"] is None


def test_check_range_bad_bounds():
    with pytest.raises(ValueError):
        check_range(_window(), {"ecg": (1.0, -1.0)})
    with pytest.raises(ValueError):
        check_range(_window(), {"ecg": (0.0, np.inf)})
