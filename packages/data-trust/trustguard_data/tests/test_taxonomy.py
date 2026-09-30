"""Unit tests for trustguard_data.taxonomy."""

import pytest

from trustguard_data.taxonomy import (
    ChannelHealth,
    HealthState,
    Layer1Result,
    classify_channel,
)


def _outputs(**kwargs):
    base = {
        "missing": False,
        "schema_ok": True,
        "dropout": {
            "missing_fraction": 0.0,
            "longest_nan_run_s": 0.0,
        },
        "range": {"fraction_out_of_range": 0.0},
        "stuck": {"is_stuck": False, "stuck_fraction": 0.0,
                  "longest_stuck_s": 0.0},
        "ks": {"pvalue": 0.5, "statistic": 0.05},
        "psi": 0.02,
        "trend": {"slope_per_s": 0.0, "pvalue": 0.8},
        "window_s": 10.0,
        "std": 1.0,
        "consistency_issue": False,
    }
    base.update(kwargs)
    return base


def test_healthy():
    ch = classify_channel("ecg", _outputs())
    assert ch.state == HealthState.HEALTHY
    assert ch.score == 100.0
    assert ch.channel == "ecg"
    assert isinstance(ch, ChannelHealth)


def test_missing():
    ch = classify_channel("ecg", _outputs(missing=True))
    assert ch.state == HealthState.MISSING
    assert ch.score == 0.0


def test_dropout_by_fraction():
    o = _outputs()
    o["dropout"] = {"missing_fraction": 0.4, "longest_nan_run_s": 1.0}
    ch = classify_channel("ecg", o)
    assert ch.state == HealthState.DROPOUT
    assert ch.score == pytest.approx(100 - 50 - 50 * 0.4)


def test_dropout_by_long_run():
    o = _outputs()
    o["dropout"] = {"missing_fraction": 0.01, "longest_nan_run_s": 15.0}
    ch = classify_channel("ecg", o)
    assert ch.state == HealthState.DROPOUT


def test_stuck():
    o = _outputs()
    o["stuck"] = {"is_stuck": True, "stuck_fraction": 0.5,
                  "longest_stuck_s": 8.0}
    ch = classify_channel("ecg", o)
    assert ch.state == HealthState.STUCK
    assert ch.score == pytest.approx(100 - 60 - 20 * 0.5)


def test_drifting_by_ks():
    o = _outputs(ks={"pvalue": 1e-5, "statistic": 0.4})
    ch = classify_channel("ecg", o)
    assert ch.state == HealthState.DRIFTING
    assert ch.score < 100.0


def test_drifting_by_psi():
    ch = classify_channel("ecg", _outputs(psi=0.4))
    assert ch.state == HealthState.DRIFTING


def test_drifting_by_trend():
    o = _outputs(trend={"slope_per_s": 0.5, "pvalue": 1e-4})
    ch = classify_channel("ecg", o)
    assert ch.state == HealthState.DRIFTING  # swing 5.0 > 1.0 * std


def test_trend_without_swing_not_drifting():
    o = _outputs(trend={"slope_per_s": 0.01, "pvalue": 1e-4})
    ch = classify_channel("ecg", o)
    assert ch.state == HealthState.HEALTHY  # swing 0.1 < 1.0 * std


def test_degraded_by_range():
    o = _outputs()
    o["range"] = {"fraction_out_of_range": 0.1}
    ch = classify_channel("ecg", o)
    assert ch.state == HealthState.DEGRADED
    assert ch.score == 80.0


def test_degraded_by_psi_watch_band():
    ch = classify_channel("ecg", _outputs(psi=0.15))
    assert ch.state == HealthState.DEGRADED


def test_degraded_by_consistency():
    ch = classify_channel("ecg", _outputs(consistency_issue=True))
    assert ch.state == HealthState.DEGRADED


def test_priority_dropout_over_stuck():
    o = _outputs()
    o["dropout"] = {"missing_fraction": 0.5, "longest_nan_run_s": 1.0}
    o["stuck"] = {"is_stuck": True, "stuck_fraction": 0.5,
                  "longest_stuck_s": 8.0}
    ch = classify_channel("ecg", o)
    assert ch.state == HealthState.DROPOUT


def test_priority_stuck_over_drift():
    o = _outputs(psi=0.9)
    o["stuck"] = {"is_stuck": True, "stuck_fraction": 0.5,
                  "longest_stuck_s": 8.0}
    ch = classify_channel("ecg", o)
    assert ch.state == HealthState.STUCK


def test_threshold_override():
    # psi=0.4 no longer reaches the overridden drift bar, but still lands in
    # the default watch band -> DEGRADED rather than DRIFTING.
    ch = classify_channel("ecg", _outputs(psi=0.4),
                          thresholds={"psi_drift": 0.9})
    assert ch.state == HealthState.DEGRADED


def test_threshold_override_full():
    ch = classify_channel(
        "ecg", _outputs(psi=0.4),
        thresholds={"psi_drift": 0.9, "psi_watch": 0.5},
    )
    assert ch.state == HealthState.HEALTHY


def test_bad_input():
    with pytest.raises(ValueError):
        classify_channel("", _outputs())
    with pytest.raises(TypeError):
        classify_channel("ecg", ["not", "a", "dict"])
    with pytest.raises(TypeError):
        classify_channel("ecg", _outputs(), thresholds="nope")


def test_layer1_result_shape():
    ch = classify_channel("ecg", _outputs())
    res = Layer1Result(device_id="d", start_s=0.0, end_s=10.0,
                       channels={"ecg": ch}, layer_score=ch.score)
    assert res.layer_score == 100.0
    assert res.channels["ecg"].state == HealthState.HEALTHY
