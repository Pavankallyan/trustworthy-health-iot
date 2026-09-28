"""TrustGuard-IoT Layer 1: Data Trust.

Streaming validators for connected health-IoT sensor data:
schema/range checks, cross-sensor consistency rules, statistical drift
detection (KS test, PSI), unsupervised anomaly detection, and per-sensor
health scoring (HEALTHY / DEGRADED / DRIFTING / STUCK / DROPOUT).

Status: scaffold — validators land in Weeks 3-4.
"""

__version__ = "0.1.0"
