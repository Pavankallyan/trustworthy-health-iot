"""TrustGuard-IoT Layer 1: silent data-corruption detection for health IoT.

Public API:

- :class:`~trustguard_data.stream.Window` — the core window type.
- :class:`~trustguard_data.pipeline.DataTrustPipeline` — fit baselines, score windows.
- :class:`~trustguard_data.injection.FaultInjector` — plausible fault injection
  with logged ground truth (for evaluation).
- :class:`~trustguard_data.taxonomy.HealthState`,
  :class:`~trustguard_data.taxonomy.ChannelHealth`,
  :class:`~trustguard_data.taxonomy.Layer1Result`,
  :func:`~trustguard_data.taxonomy.classify_channel` — the health taxonomy.
- Detector functions: ``check_schema`` / ``check_range`` (validators),
  ``detect_dropout``, ``detect_stuck``, ``ks_drift`` / ``psi`` /
  ``trend_slope`` (drift), ``acc_consistency`` / ``hr_agreement``
  (consistency).
"""

from .consistency import acc_consistency, hr_agreement
from .drift import ks_drift, psi, trend_slope
from .dropout import detect_dropout
from .injection import FAULT_TYPES, FaultInjector, InjectionResult
from .pipeline import DataTrustPipeline
from .stream import Window, split_stream, validate_window
from .stuck import detect_stuck
from .taxonomy import (
    DEFAULT_THRESHOLDS,
    ChannelHealth,
    HealthState,
    Layer1Result,
    classify_channel,
)
from .validators import check_range, check_schema

__all__ = [
    "Window",
    "validate_window",
    "split_stream",
    "check_schema",
    "check_range",
    "detect_dropout",
    "detect_stuck",
    "ks_drift",
    "psi",
    "trend_slope",
    "acc_consistency",
    "hr_agreement",
    "HealthState",
    "ChannelHealth",
    "Layer1Result",
    "classify_channel",
    "DEFAULT_THRESHOLDS",
    "FAULT_TYPES",
    "FaultInjector",
    "InjectionResult",
    "DataTrustPipeline",
]
