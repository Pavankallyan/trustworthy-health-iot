"""Unit tests for trustguard_model.features."""

import numpy as np
import pytest

from trustguard_model._compat import Window
from trustguard_model.features import FEATURE_STATS, extract_features


def make_window(device_id="dev-1", start_s=0.0, seed=0, n=100, nan_frac=0.0,
                nan_channel=None):
    rng = np.random.default_rng(seed)
    ecg = rng.normal(0.0, 1.0, n)
    bvp = rng.normal(5.0, 0.5, n)
    if nan_frac > 0.0:
        mask = rng.random(n) < nan_frac
        ecg = ecg.copy()
        ecg[mask] = np.nan
    channels = {"ecg": ecg, "bvp": bvp}
    if nan_channel is not None:
        channels[nan_channel] = np.full(n, np.nan)
    return Window(
        device_id=device_id,
        start_s=start_s,
        end_s=start_s + n / 50.0,
        channels=channels,
        fs={"ecg": 50.0, "bvp": 50.0},
    )


def test_shape_columns_and_determinism():
    windows = [make_window(start_s=float(i * 2), seed=i) for i in range(3)]
    df1 = extract_features(windows, ["ecg", "bvp"])
    df2 = extract_features(windows, ["ecg", "bvp"])
    assert df1.shape == (3, 14)
    expected_cols = [f"{ch}__{s}" for ch in ["ecg", "bvp"] for s in FEATURE_STATS]
    assert list(df1.columns) == expected_cols
    assert df1.equals(df2)  # deterministic


def test_stat_values_match_numpy():
    w = make_window(seed=42, n=64)
    df = extract_features([w], ["ecg"])
    x = w.channels["ecg"]
    row = df.iloc[0]
    assert row["ecg__mean"] == pytest.approx(float(x.mean()))
    assert row["ecg__std"] == pytest.approx(float(x.std()))
    assert row["ecg__min"] == pytest.approx(float(x.min()))
    assert row["ecg__max"] == pytest.approx(float(x.max()))
    assert row["ecg__rms"] == pytest.approx(float(np.sqrt(np.mean(x**2))))
    assert row["ecg__peak_to_peak"] == pytest.approx(float(x.max() - x.min()))
    # zero crossings of demeaned signal
    xc = x - x.mean()
    signs = np.sign(xc)
    signs = signs[signs != 0]
    expected_zc = float(np.sum(signs[1:] != signs[:-1])) if signs.size >= 2 else 0.0
    assert row["ecg__zero_crossings"] == pytest.approx(expected_zc)


def test_partial_nans_excluded_from_stats():
    w = make_window(seed=3, n=100, nan_frac=0.2)
    df = extract_features([w], ["ecg"])
    x = w.channels["ecg"]
    valid = x[~np.isnan(x)]
    assert len(valid) < 100 and len(valid) > 0
    assert df.iloc[0]["ecg__mean"] == pytest.approx(float(valid.mean()))
    assert df.iloc[0]["ecg__std"] == pytest.approx(float(valid.std()))
    assert not df.isna().any().any()  # no NaNs emitted when some samples remain


def test_all_nan_channel_yields_nan_features():
    w = make_window(seed=3, nan_channel="ecg")
    df = extract_features([w], ["ecg", "bvp"])
    ecg_cols = [c for c in df.columns if c.startswith("ecg__")]
    bvp_cols = [c for c in df.columns if c.startswith("bvp__")]
    assert df[ecg_cols].isna().all().all()  # documented: imputed downstream
    assert not df[bvp_cols].isna().any().any()


def test_zero_crossings_constant_signal():
    w = Window(device_id="d", start_s=0.0, end_s=1.0,
               channels={"flat": np.full(50, 2.0)}, fs={"flat": 50.0})
    df = extract_features([w], ["flat"])
    assert df.iloc[0]["flat__zero_crossings"] == 0.0
    assert df.iloc[0]["flat__std"] == pytest.approx(0.0)
    assert df.iloc[0]["flat__peak_to_peak"] == pytest.approx(0.0)


def test_channel_subset_and_order():
    windows = [make_window(seed=i) for i in range(2)]
    df = extract_features(windows, ["bvp", "ecg"])  # reversed order
    assert list(df.columns)[:7] == [f"bvp__{s}" for s in FEATURE_STATS]
    assert list(df.columns)[7:] == [f"ecg__{s}" for s in FEATURE_STATS]


def test_validation_errors():
    w = make_window()
    with pytest.raises(ValueError):
        extract_features([], ["ecg"])
    with pytest.raises(ValueError):
        extract_features([w], [])
    with pytest.raises(ValueError):
        extract_features([w], ["nope"])  # unknown channel
    with pytest.raises(ValueError):
        extract_features("not-a-list", ["ecg"])
    with pytest.raises(ValueError):
        extract_features([w], [""])
    # empty channel array: the real Layer 1 Window rejects these at
    # construction, so exercise this path with a Window-like duck type
    from types import SimpleNamespace

    bad_like = SimpleNamespace(channels={"ecg": np.array([])})
    with pytest.raises(ValueError, match="empty"):
        extract_features([bad_like], ["ecg"])
    with pytest.raises(TypeError):
        extract_features([object()], ["ecg"])  # not Window-like
