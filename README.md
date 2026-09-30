# TrustGuard-IoT

A unified two-layer framework for detecting **silent data corruption** and **ML model degradation** in connected health IoT systems.

## The problem I'm tackling

Health IoT devices fail in two ways that crash-based monitoring never catches:

1. **Silent data corruption** — sensors keep streaming, but the readings are wrong: gradual drift, stuck-at values, dropouts, or sensors falling out of sync with each other. Everything looks "up" while the data quietly poisons every downstream decision.
2. **Model degradation** — the ML models consuming those streams degrade after deployment as data distributions shift, with no labels around to tell you.

Today these two problems are handled (if at all) by separate tooling. I'm building one framework that watches both layers and fuses them into a single, interpretable **trust score per device** — so an operator can see *why* a device shouldn't be trusted right now, not just that something is off.

## Why this project exists

This is my flagship research build. It pulls together everything I've worked on:

- SDET/QA instincts (3+ years testing, 40+ IoT products) → the corruption-injection evaluation method
- Health-app experience → the connected-health domain
- My `data-quality-monitor` and `ml-monitoring-dashboard` repos → the two layers
- My `sla-breach-forecaster` → the alerting engine
- My `iot-predictive-maintenance` → the streaming backbone

The goal is a working system **and** a research paper (working title: *TrustGuard-IoT: A Unified Two-Layer Framework for Detecting Silent Data Corruption and Model Degradation in Connected Health IoT Systems*). The full research blueprint — architecture, datasets, experiments, paper outline, venues — lives in the project blueprint document.

## Current status

🚧 **Week 1 — scaffold + dataset curation.** Right now this repo contains the monorepo skeleton and the curated open datasets under `data/raw/` (not committed — see `data/README.md`). The validators, model monitors, and fusion dashboard land over the next few weeks per the blueprint's 10-week plan.

No results yet, and I'm not claiming any — metrics will appear here only after real experiments run.

## Repository layout

```
trustworthy-health-iot/
├── packages/
│   ├── data-trust/          # Layer 1: streaming data validators + sensor health scoring
│   ├── model-trust/         # Layer 2: deployed-model degradation monitors
│   └── fusion-dashboard/    # Trust fusion: unified per-device trust score + dashboard
├── experiments/             # Reproducible experiment scripts (corruption injection, baselines)
├── docs/                    # Design notes, paper drafts
└── data/
    ├── README.md            # Dataset inventory (sources, licenses, checksums)
    └── raw/                 # Downloaded datasets (gitignored, never committed)
```

## Datasets

Open datasets only — no PHI. See [`data/README.md`](data/README.md) for sources, licenses, and verification details.

## License

MIT
