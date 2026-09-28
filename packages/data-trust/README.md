# data-trust (Layer 1)

Streaming validators for connected health-IoT sensor data.

Planned components (Weeks 3-4):
- Schema and range checks per sensor channel
- Cross-sensor consistency rules (e.g. physiological plausibility)
- Statistical drift detection: Kolmogorov–Smirnov test, Population Stability Index
- Unsupervised anomaly detection baselines
- Per-sensor health scoring: `HEALTHY / DEGRADED / DRIFTING / STUCK / DROPOUT`

Importable as `trustguard_data` (Python 3.10+).
