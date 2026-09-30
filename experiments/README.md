# experiments

How I prove TrustGuard-IoT works — no hand-waving, every number measured.

## Pre-registration

[`experiment_plan.md`](experiment_plan.md) locks the full protocol **before**
any experiment runs: datasets and fixed subsets, the five fault types with
locked parameters, train/validation/test splits by subject (no leakage), fixed
seeds (master 42, per-run `42000 + run_index`), the three baselines, the
five-cell ablation grid, and the success bar. Results are reported per
dataset, pass or fail — no cherry-picking.

## Scripts

| Script | What it does |
|---|---|
| `common.py` | Dataset loaders (WESAD / PAMAP2 / PTB ECG), windowing, empirical ranges |
| `run_layer1.py` | Fault-injection detection on WESAD (4 fault types × 3 severities, per-device splits), recall/false-alarms |
| `run_baselines.py` | IsolationForest, KS-only, PCA-reconstruction on the identical injections |
| `run_layer2.py` | WESAD stress classifier + degradation monitoring under corruption |
| `run_ablations.py` | L1-only / L2-only / full / no-KS-PSI / fused-vs-independent |
| `run_all.py` | Runs everything in order, writes `results/results_summary.json` + `dashboard.html` |

Run the suite with:

```bash
PYTHONPATH=packages/data-trust:packages/model-trust:packages/fusion-dashboard:experiments \
  python experiments/run_all.py
```

## Results

`results/` holds every run's JSON (per-dataset details), `results_summary.json`
(headline numbers), the trained stress model, and `dashboard.html` — a
self-contained report rendered from real scored streams.

## Honest limits (also in the plan)

- Injected faults stand in for real-world sensor failures.
- Each dataset runs on a fixed, stated subset (2 CPUs / ~2 GB RAM here).
- The PCA baseline is a linear proxy for an autoencoder baseline.
- Label-free proxies are validated against true labels in these experiments;
  label-free deployment stays an open problem.
