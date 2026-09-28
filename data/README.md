# Datasets

Open health-IoT datasets used by TrustGuard-IoT. **Nothing under `data/raw/` is
committed to git** — download the datasets locally following the sources below,
or reproduce the verification steps. All three were downloaded and verified on
2026-09-28.

## Inventory

| Dataset | Source | Size (download) | Contents (verified) | License |
|---|---|---|---|---|
| WESAD | Uni Siegen sciebo share (linked from [UCI ML Repo #465](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection)) | 2,249,444,501 bytes (`wesad.zip`) | 92 files, ~17.6 GB uncompressed; 15 subjects (S2–S17); chest device (ACC, ECG, EDA, EMG, Resp, Temp @700 Hz) + wrist device (ACC, BVP, EDA, TEMP @64 Hz); affect labels (baseline / stress / amusement / meditation) | Publicly available for research (per dataset authors; see `WESAD/wesad_readme.pdf` in the archive) |
| PAMAP2 | [UCI ML Repo #231](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring) (legacy mirror: `archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip`) | 688,167,640 bytes (`pamap2.zip`) | 22 files, ~1.73 GB uncompressed; 3,850,505 rows × 54 columns; 9 protocol + 3 optional subjects; heart rate + 3 IMUs (accel/gyro/magnetometer) @100 Hz; 18 activity labels | CC-BY 4.0 (UCI ML Repository default) |
| PTB Diagnostic ECG Database (v1.0.0) | [PhysioNet](https://physionet.org/content/ptbdb/1.0.0/) — **Open Access** | ~1.7 GB, 549 records (no single zip; fetched per-record from `https://physionet.org/files/ptbdb/1.0.0/`) | 549 records from 290 subjects; 15-lead ECG (12 standard + 3 Frank leads) @1000 Hz, 16-bit; `.hea` headers with clinical annotations | Open Data Commons Attribution License v1.0; cite Bousseljot R. et al. |

### Note on PTB-XL

The blueprint originally named PTB-XL, but PTB-XL on PhysioNet requires
credentialed access. The **PTB Diagnostic ECG Database** (same source
institution, open access) is used instead — same 15-lead, 1000 Hz ECG format,
suitable for the Layer-1 corruption-injection experiments (drift, dropout,
desync across leads).

## Verification performed (2026-09-28)

- **WESAD**: `unzip -t` over the full 2.25 GB archive — no errors. Extracted
  subject S2's pickle: keys `signal`/`label`/`subject`; chest ACC shape
  (4,255,300, 3); wrist+chest channel lists confirmed.
- **PAMAP2**: `unzip -t` — no errors. 3,850,505 total data rows across the 12
  subject `.dat` files; 54 columns per row confirmed on `subject101.dat`.
- **PTB-DB**: all 549/549 records fetched (`.dat` + `.hea`); SHA-256 checksums
  spot-checked against PhysioNet's `SHA256SUMS.txt` — all matched.

## Layout

```
data/
├── README.md   # this file
└── raw/        # local copies (gitignored)
    ├── wesad.zip
    ├── pamap2.zip
    └── ptbdb/  # patient001/ ... patient294/
```

Corruption-injection harnesses (Weeks 8–9) read from `data/raw/` and write
derived experiment artifacts to `experiments/` — never back into `data/`.
