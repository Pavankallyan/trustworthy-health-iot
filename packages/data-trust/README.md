# trustguard-data — Layer 1: Silent Data-Corruption Detection

I'm Pavan, an MS Data Science student building **TrustGuard-IoT**, a research
project on trustworthy health IoT. This package is **Layer 1: data trust** —
it watches raw sensor streams (ECG, BVP/PPG, accelerometer, …) and flags
*silent* corruption: the kind where the device keeps streaming and nothing
raises an alarm, but the numbers are wrong.

## Why I built this

In my SDET role I spent 3+ years chasing bugs that never threw an exception —
flaky sensors, frozen values, drifting calibrations. Health wearables have the
same failure modes, except the "test oracle" is a person's wellbeing. Layer 1
is my attempt to give every window of sensor data a health verdict of its own,
before any ML model ever sees it.

## What it does

- **Windows** (`stream.py`): the core `Window` type — one fixed-duration slice
  of synchronized channels (`NaN` = missing sample) — plus validation and a
  helper to slice streams into windows.
- **Detectors** (each unit-tested directly):
  - `validators.py` — `check_schema` (channel presence, dtype, length) and
    `check_range` (plausible value ranges; NaNs don't count as violations).
  - `dropout.py` — `detect_dropout`: missing fraction, longest NaN run, run
    count.
  - `stuck.py` — `detect_stuck`: frozen-value runs within a tolerance, with a
    minimum duration so brief plateaus don't false-positive.
  - `drift.py` — `ks_drift` (two-sample KS vs a fitted baseline), `psi`
    (Population Stability Index), `trend_slope` (least-squares slope per
    second with a p-value significance proxy).
  - `consistency.py` — `acc_consistency` (two accelerometers compared by
    signal magnitude: Pearson correlation + cross-correlation lag with a
    desync flag) and `hr_agreement` (peak-based heart rate from two pulsatile
    signals, e.g. ECG vs BVP, with an agreement flag). Channel names are
    always parameters — nothing is hardcoded.
- **Taxonomy** (`taxonomy.py`): `HealthState` (`HEALTHY / DEGRADED / DRIFTING /
  STUCK / DROPOUT / MISSING`), `ChannelHealth`, `Layer1Result`, and
  `classify_channel`, which maps detector outputs to a state with a 0–100
  score. The decision order is documented in the docstring: missing →
  dropout → stuck → drifting → degraded → healthy. A trend only counts as
  drift if the fitted line moves the signal by at least one standard
  deviation across the window — tiny-but-"significant" slopes on long noisy
  windows are usually noise, and I tuned that bar after watching it
  false-positive on clean accelerometer data.
- **Fault injection** (`injection.py`): `FaultInjector` applies five plausible
  fault types (`gradual_drift`, `stuck_at`, `random_dropout`,
  `quantization_noise`, `desync`) and returns the corrupted window plus
  ground-truth sample indices. Injections are clipped to plausible ranges so
  they pass plain range checks — only the statistical detectors should fire.
  Seeded and deterministic.
- **Pipeline** (`pipeline.py`): `DataTrustPipeline` fits per-channel baselines
  from *clean* windows only (`fit_baseline`), then scores windows
  (`score_window`) or streams (`score_stream`), folding cross-channel
  consistency results into the per-channel verdicts.

## Quickstart

```python
from trustguard_data import (
    DataTrustPipeline, FaultInjector, split_stream,
)

# Slice a 3-channel stream into 10 s windows.
windows = split_stream("watch-01", channels, fs, window_s=10.0)

pipeline = DataTrustPipeline(config={
    "ranges": {"ecg": (-1.5, 1.5), "bvp": (-1.0, 1.0), "acc": (-1.5, 1.5)},
    "schema": {"ecg": ("f", 100), "bvp": ("f", 100), "acc": ("f", 100)},
    "hr_pair": ("ecg", "bvp"),          # cross-check these two channels' HR
})
pipeline.fit_baseline(windows[:3], seed=0)   # clean windows only

# Evaluate a fault the way the paper's experiments will:
injector = FaultInjector(seed=7)
faulty, truth = injector.inject(windows[3], [
    {"type": "stuck_at", "channel": "ecg",
     "start": 0.0, "end": 10.0, "params": {}},
])

result = pipeline.score_window(faulty)
print(result.channels["ecg"].state)   # HealthState.STUCK
print(result.layer_score)              # mean of channel scores, 0-100
```

## Design rules I held myself to

- **Validate everything**: every public function raises `ValueError`/`TypeError`
  with a clear message on bad input — never an obscure traceback.
- **Deterministic when seeded**: randomness always takes an explicit `seed`;
  there is no hidden global RNG. Tests use fixed seeds and tiny synthetic
  signals — no datasets, no network.
- **Small, named returns**: detectors return plain dicts and small dataclasses,
  never bare tuples. (`inject` returns an `InjectionResult` that also supports
  `new_window, truth = ...` unpacking.)
- **Dependencies**: numpy, pandas, scipy only. Python ≥ 3.10.

## Tests

92 pytest tests live inside the package (`trustguard_data/tests/`), one module
per source file plus an end-to-end integration test: a synthetic 3-channel
stream, baseline fit on clean windows, an injected stuck-at fault, asserting
the fault is detected, classified `STUCK`, and that clean windows stay
`HEALTHY` with a higher layer score.

```bash
cd <repo-root>
PYTHONPATH=packages/data-trust <venv>/bin/python -m pytest \
    packages/data-trust/trustguard_data/tests/ -q
```

Layer 2 (`trustguard_model`) consumes per-window features built on top of these
verdicts — but that's another package's story.
