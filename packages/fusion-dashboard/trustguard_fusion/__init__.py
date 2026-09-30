"""TrustGuard-IoT fusion layer: one trust score per device.

I fuse Layer 1 (data trust: is the sensor stream telling the truth?) with
Layer 2 (model trust: is the deployed model still performing?) into a single
0-100 score, with every penalty term named so the score stays explainable.
"""

from trustguard_fusion.alerts import (
    Alert,
    AlertDispatcher,
    Severity,
    route_alerts,
)
from trustguard_fusion.dashboard import render_html
from trustguard_fusion.fusion import TrustReport, fuse_trust

__all__ = [
    "Alert",
    "AlertDispatcher",
    "Severity",
    "TrustReport",
    "fuse_trust",
    "render_html",
    "route_alerts",
]
