# trustguard_model — Layer 2: Deployed-Model Trust for Health IoT

I'm Pavan, an MS Data Science student at Wentworth building **TrustGuard-IoT**,
a two-layer framework that watches both the *data* coming off connected health
devices (Layer 1) and the *deployed ML models* consuming that data (Layer 2).
This package is Layer 2.

## Why this exists

In my SDET role I learned a painful lesson: models don't fail loudly. A
classifier silently degrading under sensor drift looks exactly like a model
that's working — until someone checks the outcomes. With 3+ years across QA
and data work behind me, I wanted a monitor that catches degradation *before*
labels arrive to confirm it, because in production health IoT, labels are
late, sparse, or never come.

## What it does

Given a stream of predicted probabilities from a deployed binary/multiclass
classifier, `trustguard_model` answers one question continuously: **"is this
model still behaving like the model we validated?"**

- **Label-free signals** (always on): prediction PSI vs a trusted reference
  distribution, a two-sample KS test on predicted probabilities, and
  mean-confidence drop. No labels needed.
- **Label-gated signals** (when `y_true` arrives): ECE drift and rolling
  accuracy — because when you *do* get labels, you should squeeze them.
- **Label-free proxies** (`proxies.py`): prediction-entropy monitoring
  (rising indecision is a classic pre-degradation tell) and agreement with a
  cheap heuristic surrogate you supply (a threshold rule is enough — what
  matters is that agreement is *stable* in-distribution).
- **Threshold + hysteresis alerting** (`alerting.py`): a breached signal must
  clear for K consecutive updates (default 3) before it un-breaches, so one
  noisy batch can't flap the `degraded` flag.

## Layout

```
trustguard_model/
  features.py   # Window list -> DataFrame (mean, std, min, max, rms,
                #   peak-to-peak, zero-crossings per channel; NaNs excluded
                #   from stats, all-NaN channels emit NaN -> median-imputed
                #   by the baseline pipeline)
  baseline.py   # train_stress_model (imputer + scaler + logistic regression),
                #   evaluate (accuracy, f1, auc, hand-rolled ECE),
                #   save_model / load_model (joblib)
  monitors.py   # ModelTrustMonitor -> Layer2Result (signals, degraded, health)
  proxies.py    # EntropyMonitor, SurrogateAgreementMonitor
  alerting.py   # DegradationAlert: reusable threshold+hysteresis primitive
  tests/        # pytest suite, synthetic data only, fixed seeds
```

## Quick start

```python
import numpy as np
from trustguard_model import (
    extract_features, train_stress_model, evaluate, ModelTrustMonitor,
)

# 1. features from Layer 1 windows
X = extract_features(windows, channels=["ecg", "bvp"])

# 2. train the reference stress model
model = train_stress_model(X, y, seed=11)
print(evaluate(model, X, y))   # {'accuracy': ..., 'f1': ..., 'auc': ..., 'ece': ...}

# 3. monitor the deployed model against its validated behavior
reference_proba = model.predict_proba(X)[:, 1]
monitor = ModelTrustMonitor("stress-v1", reference_proba)

result = monitor.update(deployment_proba)            # label-free
result = monitor.update(deployment_proba, y_true)    # + ECE drift, rolling accuracy
print(result.degraded, result.health)
for name, sig in result.signals.items():
    print(name, round(sig.value, 4), "breached:" , sig.breached)
```

## Design notes

- **Deterministic**: every randomized step takes an explicit `seed`; no
  hidden global RNG. Tests use fixed seeds throughout.
- **Small returns**: dataclasses (`ModelSignal`, `Layer2Result`) and plain
  dicts of floats — never ambiguous bare tuples.
- **Strict validation**: every public function validates inputs and raises
  `ValueError`/`TypeError` with clear messages (empty arrays, mismatched
  lengths, non-binary labels, bad config keys, proba outside [0, 1]).
- **Dependencies**: numpy, pandas, scipy, scikit-learn only. Python ≥ 3.10.

## Tests

```bash
PYTHONPATH=packages/model-trust:packages/data-trust \
  python -m pytest packages/model-trust/trustguard_model/tests/ -q
```

The suite covers every module plus one end-to-end integration test: train on
synthetic windows, then detect a simulated sensor-drift shift via the
monitor. Everything is synthetic with fixed seeds — no datasets, no network.
