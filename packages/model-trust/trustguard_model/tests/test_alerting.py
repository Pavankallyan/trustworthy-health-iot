"""Unit tests for trustguard_model.alerting (threshold + hysteresis)."""

import math

import pytest

from trustguard_model.alerting import DEFAULT_CLEAR_STREAK, DegradationAlert


def test_breach_above_threshold_immediately():
    alert = DegradationAlert("psi", threshold=0.25)
    assert alert.breached is False
    assert alert.update(0.30) is True
    assert alert.breached is True


def test_no_breach_at_or_below_threshold():
    alert = DegradationAlert("psi", threshold=0.25)
    assert alert.update(0.25) is False  # strictly greater breaches
    assert alert.update(0.10) is False
    assert alert.breached is False


def test_hysteresis_requires_k_consecutive_clears():
    k = 3
    alert = DegradationAlert("psi", threshold=0.25, clear_streak_required=k)
    alert.update(0.9)  # breach
    assert alert.breached is True
    # fewer than K clears: still breached
    assert alert.update(0.05) is True
    assert alert.clear_streak == 1
    assert alert.update(0.05) is True
    assert alert.clear_streak == 2
    # Kth consecutive clear: heals
    assert alert.update(0.05) is False
    assert alert.breached is False
    assert alert.clear_streak == 0


def test_rebreach_resets_clear_streak():
    alert = DegradationAlert("psi", threshold=0.25, clear_streak_required=3)
    alert.update(0.9)
    alert.update(0.05)
    alert.update(0.05)
    assert alert.clear_streak == 2
    alert.update(0.9)  # re-breach resets the streak
    assert alert.breached is True
    assert alert.clear_streak == 0


def test_direction_below():
    alert = DegradationAlert("acc", threshold=0.8, direction="below")
    assert alert.update(0.85) is False
    assert alert.update(0.79) is True  # strictly less breaches
    # heal needs K clears at/above threshold
    assert alert.update(0.9) is True
    assert alert.update(0.9) is True
    assert alert.update(0.9) is False


def test_margin_holds_band_values():
    # margin 0.05, direction above: clear only at <= 0.20; (0.20, 0.25] holds
    alert = DegradationAlert("psi", threshold=0.25, margin=0.05,
                            clear_streak_required=2)
    alert.update(0.9)
    assert alert.breached is True
    alert.update(0.22)  # inside the band: holds, streak resets
    assert alert.breached is True
    assert alert.clear_streak == 0
    alert.update(0.20)  # clear zone
    assert alert.clear_streak == 1
    assert alert.update(0.10) is False  # second consecutive clear heals


def test_reset():
    alert = DegradationAlert("psi", threshold=0.25)
    alert.update(0.9)
    alert.reset()
    assert alert.breached is False
    assert alert.clear_streak == 0


def test_snapshot():
    alert = DegradationAlert("psi", threshold=0.25)
    snap = alert.snapshot()
    assert snap.name == "psi"
    assert snap.breached is False
    assert snap.clear_streak_required == DEFAULT_CLEAR_STREAK


def test_invalid_construction():
    with pytest.raises(ValueError):
        DegradationAlert("", threshold=0.25)
    with pytest.raises(ValueError):
        DegradationAlert("x", threshold=0.25, direction="sideways")
    with pytest.raises(ValueError):
        DegradationAlert("x", threshold=0.25, clear_streak_required=0)
    with pytest.raises(ValueError):
        DegradationAlert("x", threshold=float("nan"))
    with pytest.raises(ValueError):
        DegradationAlert("x", threshold=0.25, margin=-0.1)
    with pytest.raises(TypeError):
        DegradationAlert("x", threshold="high")


def test_invalid_update_values():
    alert = DegradationAlert("psi", threshold=0.25)
    with pytest.raises(ValueError):
        alert.update(float("nan"))
    with pytest.raises(ValueError):
        alert.update(float("inf"))
    with pytest.raises(TypeError):
        alert.update("0.3")
    with pytest.raises(TypeError):
        alert.update(None)


def test_default_k_is_three():
    assert DEFAULT_CLEAR_STREAK == 3
