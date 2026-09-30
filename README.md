# TrustGuard-IoT

I built TrustGuard-IoT because silent sensor corruption and silent model degradation are the two failures I kept running into in health IoT — and no single tool watched for both. It's a unified two-layer framework: **Layer 1** watches the raw sensor data for corruption (drift, stuck sensors, dropouts, quantization), **Layer 2** watches a deployed model for performance degradation without needing labels, and a **fusion engine** combines both into one explainable trust score per device with a live dashboard.

## What I measured (all real runs, fixed seeds, 2026-09-30)

**Layer 1 — fault injection on real WESAD wearable data** (chest + wrist sensors, 2 test subjects, 12 injected faults across 4 fault types × 3 severities, per-device calibrated thresholds):

| Fault type | Recall |
|---|---|
| Gradual drift | 3/3 |
| Stuck-at | 3/3 |
| Random dropout | 2/3 |
| Quantization noise | 3/3 |
| **Overall** | **11/12 (0.917)** |

Validation false-alarm rate: **0.000** on both subjects. The one miss was 10% random dropout on ECG — below the detection floor I calibrated.

**Against baselines** (same 12 runs, thresholds matched to my validation FAR): TrustGuard **0.917** vs IsolationForest **0.500**, KS-only **0.750**, PCA-reconstruction **0.833**.

**Layer 2 — stress-model degradation monitoring** (trained on subject S2, validated on S3, tested on S4+S5): test accuracy **0.780**, F1 **0.629**, AUC **0.884**, ECE **0.191**. When I corrupted the model's inputs, the label-free monitor flagged **3/3** degradation scenarios, with true accuracy drops of 0.461, 0.000, and 0.154 (the middle one the model genuinely tolerated — the monitor is conservative and I report that).

**Ablations**: Layer 1 alone catches 11/12 sensor faults; Layer 2 alone catches 1/12 (it can only see corruptions that move model outputs — which is exactly why Layer 1 is necessary); removing the KS/PSI distribution detectors drops Layer 1 to 5/12. Fused trust AUC **0.951** vs independent alert OR **0.875**.

**Tests**: 156/156 pass (`pytest packages/`). CI runs on every push.

## Honest limitations

- Layer 1 is evaluated on WESAD only. I tried PAMAP2 (activities change every ~4 min — no stable baseline possible) and PTB (acute-MI records with severe intrinsic non-stationarity, KS D up to 0.99 between adjacent clean windows) and excluded both with measured reasons, documented in `experiments/experiment_plan.md`.
- Slow signals (EDA, skin temperature) drift naturally as a resting subject settles; my metrics focus on ECG/BVP where the evaluation is clean.
- The stress model leans on EDA/temperature, which are confounded with session time — cross-subject generalization is a known weakness.
- One WESAD subject (S5) was excluded from testing: its wrist BVP shows intrinsic contact-loss artifact.

## Layout

- `packages/data-trust/` — Layer 1: stream/window handling, validators (range, schema, dropout, stuck-at, drift via KS/PSI/trend, cross-sensor consistency), fault injection, pipeline.
- `packages/model-trust/` — Layer 2: feature extraction, stress-model training/evaluation, `ModelTrustMonitor` (PSI/KS/confidence/ECE/accuracy signals).
- `packages/fusion-dashboard/` — fusion (deterministic trust score with per-penalty explanations), alert routing with cooldown, JSONL audit log, static HTML dashboard.
- `experiments/` — pre-registered plan (`experiment_plan.md` + addenda), `run_layer1.py`, `run_baselines.py`, `run_layer2.py`, `run_ablations.py`, `run_all.py`, and `results/` with every measured JSON + the dashboard.
- `requirements.txt` — exact pinned versions (numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1; Python 3.12).

## Reproduce

```bash
pip install -r requirements.txt
PYTHONPATH=packages/data-trust:packages/model-trust:packages/fusion-dashboard \
  python -m pytest packages/ -q          # 156 tests
PYTHONPATH=packages/data-trust:packages/model-trust:packages/fusion-dashboard:experiments \
  python experiments/run_all.py          # full suite -> experiments/results/
```

Datasets (WESAD) are public and downloaded separately; see `experiments/README.md`. Every run uses fixed seeds (master 42, per-run 42000 + index).

## Background

I'm Pavan Kalyan, an MS Data Science student at Wentworth (Dec 2026) with 3+ years in software testing/QA. I build health-IoT side projects and got tired of data pipelines failing silently — this is my attempt at a principled fix. Preprint planned for arXiv, then 2027 workshops.
