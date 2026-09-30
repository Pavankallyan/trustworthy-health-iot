"""Unit tests for trustguard_model.proxies (label-free proxies)."""

import numpy as np
import pytest

from trustguard_model.monitors import ModelSignal
from trustguard_model.proxies import EntropyMonitor, SurrogateAgreementMonitor


def confident_proba(seed=0, n=200):
    # confident predictions -> low entropy (mean ~0.29 nats)
    return np.random.default_rng(seed).beta(30, 3, n)


def uncertain_proba(seed=1, n=200):
    # near-coin-flip predictions -> high entropy (mean ~0.59 nats)
    return np.random.default_rng(seed).beta(2, 2, n)


def test_entropy_rise_breaches_and_heals():
    mon = EntropyMonitor(confident_proba(), threshold=0.15)
    assert mon.reference_mean_entropy < 0.5  # confident reference is low-entropy
    sig = mon.update(confident_proba(seed=9))
    assert isinstance(sig, ModelSignal)
    assert sig.name == "prediction_entropy"
    assert sig.breached is False
    bad = mon.update(uncertain_proba())
    assert bad.value > 0.15
    assert bad.breached is True
    # hysteresis: K=3 confident updates to heal
    assert mon.update(confident_proba(seed=9)).breached is True
    assert mon.update(confident_proba(seed=9)).breached is True
    assert mon.update(confident_proba(seed=9)).breached is False


def test_entropy_2d_proba():
    rng = np.random.default_rng(0)
    ref = rng.dirichlet([9, 1], 200)  # confident 2-class
    mon = EntropyMonitor(ref)
    cur = np.random.default_rng(1).dirichlet([1, 1], 200)  # uniform: max entropy
    assert mon.update(cur).breached is True


def test_entropy_validation():
    with pytest.raises(ValueError):
        EntropyMonitor(np.array([]))
    with pytest.raises(ValueError):
        EntropyMonitor(np.array([0.5, 1.2]))
    with pytest.raises(ValueError):
        EntropyMonitor(np.array([0.5, -0.1]))
    mon = EntropyMonitor(confident_proba())
    with pytest.raises(ValueError):
        mon.update(np.array([]))
    with pytest.raises(ValueError):
        EntropyMonitor(confident_proba(), threshold=-0.1)


def test_surrogate_agreement_stable_then_drops():
    # surrogate: simple threshold rule on the first feature
    surrogate = lambda X: (X[:, 0] > 0.0).astype(int)
    mon = SurrogateAgreementMonitor(surrogate, threshold=0.8)
    rng = np.random.default_rng(0)
    X = rng.normal(0.0, 1.0, (100, 3))
    labels = (X[:, 0] > 0.0).astype(int)
    proba_agree = np.where(labels == 1, 0.9, 0.1)  # model matches surrogate
    ok = mon.update(proba_agree, X)
    assert ok.name == "surrogate_agreement"
    assert ok.value == pytest.approx(1.0)
    assert ok.breached is False
    assert ok.details["n_agree"] == 100
    # model flips its decisions -> agreement collapses
    proba_flip = np.where(labels == 1, 0.1, 0.9)
    bad = mon.update(proba_flip, X)
    assert bad.value == pytest.approx(0.0)
    assert bad.breached is True
    # hysteresis holds the breach through one agreeing batch
    assert mon.update(proba_agree, X).breached is True


def test_surrogate_2d_proba_argmax():
    surrogate = lambda X: (X[:, 0] > 0.0).astype(int)
    mon = SurrogateAgreementMonitor(surrogate)
    X = np.array([[1.0, 0.0], [-1.0, 0.0], [2.0, 0.0]])
    proba = np.array([[0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])  # argmax = [1,0,1]
    sig = mon.update(proba, X)
    assert sig.value == pytest.approx(1.0)
    assert sig.breached is False


def test_surrogate_validation():
    with pytest.raises(TypeError):
        SurrogateAgreementMonitor("not-callable")
    with pytest.raises(ValueError):
        SurrogateAgreementMonitor(lambda X: X[:, 0], threshold=1.5)
    mon = SurrogateAgreementMonitor(lambda X: (X[:, 0] > 0).astype(int))
    X = np.zeros((10, 2))
    with pytest.raises(ValueError):
        mon.update(np.full(9, 0.5), X)  # length mismatch
    with pytest.raises(ValueError):
        mon.update(np.full(10, 0.5), np.zeros((10,)))  # X not 2D
    with pytest.raises(ValueError):
        bad = SurrogateAgreementMonitor(lambda X: np.zeros(5, dtype=int))
        bad.update(np.full(10, 0.5), X)  # surrogate returns wrong length
    with pytest.raises(ValueError):
        bad = SurrogateAgreementMonitor(lambda X: np.full(10, 0.5))  # floats
        bad.update(np.full(10, 0.5), X)
    with pytest.raises(ValueError):
        bad = SurrogateAgreementMonitor(lambda X: 1 / 0)
        bad.update(np.full(10, 0.5), X)  # surrogate raises


def test_reset():
    mon = EntropyMonitor(confident_proba())
    mon.update(uncertain_proba())
    mon.reset()
    sig = mon.update(confident_proba(seed=9))
    assert sig.breached is False
