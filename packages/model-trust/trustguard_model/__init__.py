"""trustguard_model — Layer 2 of TrustGuard-IoT: deployed-model trust.

Monitors a deployed health-IoT classifier for degradation without requiring
ground-truth labels for most signals (prediction PSI, KS test, confidence
drop), adding label-gated signals (ECE drift, rolling accuracy) when labels
arrive, plus label-free proxies (prediction entropy, cheap-surrogate
agreement).  All alerting is threshold-plus-hysteresis.
"""

from ._compat import Window
from .alerting import DEFAULT_CLEAR_STREAK, DegradationAlert
from .baseline import (
    N_ECE_BINS,
    evaluate,
    expected_calibration_error,
    load_model,
    save_model,
    train_stress_model,
)
from .features import FEATURE_STATS, extract_features
from .monitors import DEFAULT_CONFIG, Layer2Result, ModelSignal, ModelTrustMonitor
from .proxies import EntropyMonitor, SurrogateAgreementMonitor

__all__ = [
    # features
    "extract_features",
    "FEATURE_STATS",
    # baseline
    "train_stress_model",
    "evaluate",
    "expected_calibration_error",
    "save_model",
    "load_model",
    "N_ECE_BINS",
    # monitors
    "ModelTrustMonitor",
    "ModelSignal",
    "Layer2Result",
    "DEFAULT_CONFIG",
    # proxies
    "EntropyMonitor",
    "SurrogateAgreementMonitor",
    # alerting
    "DegradationAlert",
    "DEFAULT_CLEAR_STREAK",
    # Layer 1 Window type (canonical import when trustguard_data is present)
    "Window",
]

__version__ = "0.1.0"
