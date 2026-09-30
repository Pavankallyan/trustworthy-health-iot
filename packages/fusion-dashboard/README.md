# fusion-dashboard

The fusion layer of TrustGuard-IoT. I take the per-sensor health scores from
Layer 1 (`data-trust`) and the degradation signals from Layer 2
(`model-trust`) and fuse them into **one per-device trust score (0–100)** —
the number an operator watches, with a full explainable breakdown underneath.

## How the score works

`trust = 100 + sum(named penalties)`, clamped to [0, 100]:

- Each Layer 1 channel contributes `-(100 - channel_score) * w1 / n_channels`,
  named like `layer1:ecg:drifting`.
- Each Layer 2 signal contributes `-(100 - model_health) * w2 / n_signals`,
  named like `layer2:prediction_psi`.
- Default weights: Layer 1 = 0.6, Layer 2 = 0.4 (renormalized if you pass
  custom weights; Layer 1 alone gets weight 1.0 when no model is monitored).

Because every penalty is named, the dashboard can always answer "why is this
device at 62?" — no black-box scoring.

## Alerts

`route_alerts(report, l1, l2)` maps reports to severity tiers deterministically:

- trust < 40 → CRITICAL, trust < 70 → WARNING
- stuck / dropout / missing channels → WARNING; drifting / degraded → INFO
- breached model-degradation signals → WARNING

`AlertDispatcher` adds cooldown dedup and a JSONL audit log.

## Dashboard

`render_html(reports, alerts, trend)` produces one self-contained HTML file —
inline CSS, inline SVG sparklines, zero external assets. Open it anywhere,
attach it to an email, or archive it as audit evidence.
