"""Static HTML trust dashboard: one self-contained file, no external assets.

I render the dashboard as a plain HTML string (inline CSS, inline SVG
sparklines, no JavaScript framework) so the report can be opened anywhere,
attached to an email, or archived as audit evidence.
"""

from __future__ import annotations

import html

from trustguard_fusion.alerts import Alert, Severity
from trustguard_fusion.fusion import TrustReport

_CSS = """
body{font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
margin:0;background:#0f172a;color:#e2e8f0}
header{background:#111c33;padding:20px 28px;border-bottom:1px solid #1e293b}
header h1{margin:0;font-size:22px}header p{margin:6px 0 0;color:#94a3b8;font-size:13px}
main{padding:20px 28px;max-width:1100px;margin:0 auto}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}
.card{background:#16213a;border:1px solid #243355;border-radius:12px;padding:16px}
.card h2{margin:0 0 4px;font-size:16px}
.score{font-size:40px;font-weight:700;margin:6px 0}
.bar{height:10px;background:#243355;border-radius:6px;overflow:hidden;margin:6px 0}
.bar>div{height:100%;border-radius:6px}
.meta{font-size:12px;color:#94a3b8;margin:2px 0}
.pen{font-size:12px;color:#fbbf24}
table{width:100%;border-collapse:collapse;margin-top:8px;font-size:13px}
th,td{text-align:left;padding:8px;border-bottom:1px solid #243355}
th{color:#94a3b8;font-weight:600}
.badge{display:inline-block;padding:2px 10px;border-radius:999px;font-size:11px;font-weight:700}
.crit{background:#7f1d1d;color:#fecaca}.warn{background:#78350f;color:#fde68a}
.info{background:#1e3a8a;color:#bfdbff}
footer{padding:16px 28px;color:#64748b;font-size:12px;text-align:center}
"""


def _color(score: float) -> str:
    if score >= 80:
        return "#22c55e"
    if score >= 60:
        return "#eab308"
    if score >= 40:
        return "#f97316"
    return "#ef4444"


def _sparkline(values: list[float], width: int = 220, height: int = 48) -> str:
    if not values:
        return ""
    vmin, vmax = min(values), max(values)
    span = (vmax - vmin) or 1.0
    n = len(values)
    pts = []
    for i, v in enumerate(values):
        x = 8 + i * (width - 16) / max(n - 1, 1)
        y = height - 8 - (v - vmin) / span * (height - 16)
        pts.append(f"{x:.1f},{y:.1f}")
    color = _color(values[-1])
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" '
        f'stroke-width="2"/></svg>'
    )


def _sev_badge(sev: Severity) -> str:
    cls = {Severity.CRITICAL: "crit", Severity.WARNING: "warn",
           Severity.INFO: "info"}[sev]
    return f'<span class="badge {cls}">{html.escape(sev.value.upper())}</span>'


def render_html(
    reports: list[TrustReport],
    alerts: list[Alert] | None = None,
    trend: dict[str, list[float]] | None = None,
    title: str = "TrustGuard-IoT — Device Trust Dashboard",
) -> str:
    """Render a self-contained HTML dashboard.

    reports: one TrustReport per device (latest).
    alerts: alerts to list in the alert table.
    trend: optional device_id -> list of historical trust scores for sparklines.
    """
    if not isinstance(reports, list) or not reports:
        raise ValueError("reports must be a non-empty list")
    for r in reports:
        if not isinstance(r, TrustReport):
            raise TypeError("reports must contain TrustReport objects")
    alerts = alerts or []
    for a in alerts:
        if not isinstance(a, Alert):
            raise TypeError("alerts must contain Alert objects")
    trend = trend or {}
    if not isinstance(trend, dict):
        raise TypeError("trend must be a dict or None")

    cards = []
    for r in sorted(reports, key=lambda x: x.trust_score):
        c = _color(r.trust_score)
        pens = "".join(
            f'<div class="pen">{html.escape(k)}: {v:+.2f}</div>'
            for k, v in sorted(r.contributions.items(), key=lambda kv: kv[1])[:6]
        )
        l2 = (f"{r.layer2_score:.1f}" if r.layer2_score is not None else "n/a")
        spark = _sparkline(trend.get(r.device_id, []))
        worst = ", ".join(html.escape(wc) for wc in r.worst_channels) or "none"
        cards.append(f"""
        <div class="card">
          <h2>{html.escape(r.device_id)}</h2>
          <div class="score" style="color:{c}">{r.trust_score:.0f}</div>
          <div class="meta">trust score / 100</div>
          {spark}
          <div class="meta">Layer 1 (data): {r.layer1_score:.1f}</div>
          <div class="bar"><div style="width:{r.layer1_score:.0f}%;background:{_color(r.layer1_score)}"></div></div>
          <div class="meta">Layer 2 (model): {l2}</div>
          <div class="meta">Watch: {worst}</div>
          {pens}
        </div>""")

    rows = []
    for a in alerts:
        rows.append(
            f"<tr><td>{_sev_badge(a.severity)}</td>"
            f"<td>{html.escape(a.device_id)}</td>"
            f"<td>{html.escape(a.title)}</td>"
            f"<td>{html.escape(a.detail)}</td>"
            f"<td>{html.escape(a.source)}</td></tr>")
    alert_table = ("<table><tr><th>Severity</th><th>Device</th><th>Alert</th>"
                   "<th>Detail</th><th>Source</th></tr>" + "".join(rows) + "</table>"
                   if rows else "<p class='meta'>No active alerts.</p>")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{html.escape(title)}</title><style>{_CSS}</style></head>
<body><header><h1>{html.escape(title)}</h1>
<p>Unified data-trust + model-trust monitoring for connected health IoT.
Scores are 0&ndash;100; higher is healthier.</p></header>
<main><h2>Devices</h2><div class="grid">{"".join(cards)}</div>
<h2 style="margin-top:28px">Alerts</h2>{alert_table}</main>
<footer>TrustGuard-IoT &mdash; generated report. Scores fuse Layer 1 sensor
health with Layer 2 model-degradation signals; every penalty term is
explainable in the per-device breakdown.</footer></body></html>
"""
