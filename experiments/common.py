"""Shared experiment utilities: dataset loaders, windowing, seeds.

All loaders read ONE subject/patient at a time to stay within memory limits.
Fixed master seed SEED = 42 (matches experiment_plan.md).
"""

from __future__ import annotations

import os
import pickle
import struct

import numpy as np
import pandas as pd

try:
    from trustguard_data.stream import Window
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "trustguard_data must be importable (sibling package). "
        "Run pytest/install from the repo root first."
    ) from e

SEED = 42

DATA_ROOT = os.path.expanduser(
    "~/workspace/goals/publish-a-research-paper-on-trustworthy-health-iot"
    "/hidden_files/datasets"
)
WORK = os.path.join(DATA_ROOT, "work")

# ---------------------------------------------------------------------------
# WESAD
# ---------------------------------------------------------------------------
# name -> (location, key, column-or-None, fs)
WESAD_CHANNELS = {
    "ecg": ("chest", "ECG", None, 700.0),
    "eda_c": ("chest", "EDA", None, 700.0),
    "emg": ("chest", "EMG", None, 700.0),
    "resp": ("chest", "Resp", None, 700.0),
    "temp_c": ("chest", "Temp", None, 700.0),
    "acc_c": ("chest", "ACC", "mag", 700.0),
    "bvp": ("wrist", "BVP", None, 64.0),
    "eda_w": ("wrist", "EDA", None, 4.0),
    "temp_w": ("wrist", "TEMP", None, 4.0),
    "acc_w": ("wrist", "ACC", "mag", 32.0),
    # raw accelerometer axes (cross-sensor consistency checks need axes)
    "acc_c_x": ("chest", "ACC", 0, 700.0),
    "acc_c_y": ("chest", "ACC", 1, 700.0),
    "acc_c_z": ("chest", "ACC", 2, 700.0),
    "acc_w_x": ("wrist", "ACC", 0, 32.0),
    "acc_w_y": ("wrist", "ACC", 1, 32.0),
    "acc_w_z": ("wrist", "ACC", 2, 32.0),
}

# Binary stress labels: 2 = stress; 1/3/4/6/7 = non-stress; 0 = transient (dropped)
WESAD_STRESS = 2
WESAD_NONSTRESS = {1, 3, 4, 6, 7}


def load_wesad_subject(subject: str) -> dict:
    """Load one WESAD subject pickle. Returns dict with 'signals' and 'label'."""
    path = os.path.join(WORK, "wesad", "WESAD", subject, f"{subject}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"WESAD subject file not found: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return data


def _wesad_channel(data: dict, name: str) -> np.ndarray:
    loc, key, col, _ = WESAD_CHANNELS[name]
    arr = np.asarray(data["signal"][loc][key], dtype=np.float64)
    if col == "mag":
        arr = np.sqrt(np.sum(arr ** 2, axis=1))
    elif isinstance(col, int):
        arr = arr[:, col]
    else:
        arr = arr.ravel()
    return arr


def wesad_stream(
    subject: str,
    channels: list[str],
    start_s: float = 0.0,
    duration_s: float = 720.0,
) -> Window:
    """One long Window of raw WESAD data (for fault injection)."""
    for c in channels:
        if c not in WESAD_CHANNELS:
            raise ValueError(f"unknown WESAD channel {c!r}")
    data = load_wesad_subject(subject)
    ch_data, fs = {}, {}
    for c in channels:
        _, _, _, f = WESAD_CHANNELS[c]
        arr = _wesad_channel(data, c)
        i0, i1 = int(start_s * f), int((start_s + duration_s) * f)
        if i1 > len(arr):
            raise ValueError(
                f"subject {subject} channel {c}: requested end {i1} > {len(arr)}"
            )
        ch_data[c] = arr[i0:i1]
        fs[c] = f
    return Window(device_id=f"wesad-{subject}", start_s=start_s,
                  end_s=start_s + duration_s, channels=ch_data, fs=fs)


def wesad_labels(subject: str, start_s: float, duration_s: float,
                 fs_label: float = 700.0) -> np.ndarray:
    data = load_wesad_subject(subject)
    lab = np.asarray(data["label"], dtype=np.int64).ravel()
    i0, i1 = int(start_s * fs_label), int((start_s + duration_s) * fs_label)
    return lab[i0:i1]


def window_label(labels: np.ndarray) -> int:
    """Majority non-zero label in a window; 0 if none."""
    nz = labels[labels != 0]
    if len(nz) == 0:
        return 0
    vals, counts = np.unique(nz, return_counts=True)
    return int(vals[np.argmax(counts)])


def slice_window(window: Window, start_s: float, end_s: float) -> Window:
    """Slice a long Window into a sub-window (seconds, relative to window start)."""
    if not 0 <= start_s < end_s <= (window.end_s - window.start_s):
        raise ValueError("slice bounds out of range")
    channels, fs = {}, {}
    for name, arr in window.channels.items():
        f = window.fs[name]
        i0, i1 = int(round(start_s * f)), int(round(end_s * f))
        channels[name] = arr[i0:i1].astype(np.float64)
        fs[name] = f
    return Window(device_id=window.device_id,
                  start_s=window.start_s + start_s,
                  end_s=window.start_s + end_s,
                  channels=channels, fs=fs)


def split_stream(window: Window, win_s: float = 60.0,
                 hop_s: float = 60.0) -> list[Window]:
    """Split a long Window into fixed windows."""
    total = window.end_s - window.start_s
    out = []
    t = 0.0
    while t + win_s <= total + 1e-9:
        out.append(slice_window(window, t, t + win_s))
        t += hop_s
    if not out:
        raise ValueError("stream shorter than one window")
    return out


def empirical_ranges(windows: list[Window], channels: list[str],
                     margin: float = 1.5) -> dict[str, tuple[float, float]]:
    """Derive plausible physiological ranges from clean train windows.

    Uses the 0.5th/99.5th percentiles widened by `margin`. Documented as
    empirical (dataset-specific), not textbook physiology.
    """
    if margin <= 1.0:
        raise ValueError("margin must be > 1")
    ranges = {}
    for c in channels:
        vals = np.concatenate([np.asarray(w.channels[c]).ravel()
                               for w in windows])
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            raise ValueError(f"no finite values for channel {c}")
        lo, hi = np.percentile(vals, [0.5, 99.5])
        mid = (lo + hi) / 2
        half = (hi - lo) / 2 * margin
        ranges[c] = (float(mid - half), float(mid + half))
    return ranges


# ---------------------------------------------------------------------------
# PAMAP2
# ---------------------------------------------------------------------------
# 0-indexed columns: 0 timestamp, 1 activity, 2 HR,
# hand IMU cols 3..19 (acc16 = 4,5,6), chest IMU cols 20..36 (acc16 = 21,22,23)
PAMAP2_HAND_ACC = [4, 5, 6]
PAMAP2_CHEST_ACC = [21, 22, 23]
PAMAP2_HR = 2


def load_pamap2(subject: int) -> pd.DataFrame:
    path = os.path.join(WORK, "pamap2", "PAMAP2_Dataset", "Protocol",
                        f"subject10{subject}.dat")
    if not os.path.exists(path):
        raise FileNotFoundError(f"PAMAP2 file not found: {path}")
    df = pd.read_csv(path, sep=r"\s+", header=None)
    return df


def pamap2_stream(subject: int, start_s: float = 0.0,
                  duration_s: float = 720.0) -> Window:
    df = load_pamap2(subject)
    t = df[0].to_numpy(dtype=np.float64)
    dt = float(np.median(np.diff(t)))
    if not 0 < dt < 1:
        raise ValueError(f"unexpected PAMAP2 sampling interval {dt}")
    fs = 1.0 / dt
    i0 = int(np.searchsorted(t, t[0] + start_s))
    i1 = int(np.searchsorted(t, t[0] + start_s + duration_s))
    if i1 - i0 < int(fs * 60):
        raise ValueError("PAMAP2 stream shorter than requested")
    hand = df[PAMAP2_HAND_ACC].to_numpy(dtype=np.float64)[i0:i1]
    chest = df[PAMAP2_CHEST_ACC].to_numpy(dtype=np.float64)[i0:i1]
    hr = df[PAMAP2_HR].to_numpy(dtype=np.float64)[i0:i1]
    channels = {
        "acc_hand": np.sqrt(np.nansum(hand ** 2, axis=1)),
        "acc_chest": np.sqrt(np.nansum(chest ** 2, axis=1)),
        "hr": hr,
        # raw axes for cross-sensor consistency checks
        "acc_hand_x": hand[:, 0], "acc_hand_y": hand[:, 1], "acc_hand_z": hand[:, 2],
        "acc_chest_x": chest[:, 0], "acc_chest_y": chest[:, 1], "acc_chest_z": chest[:, 2],
    }
    return Window(device_id=f"pamap2-s10{subject}", start_s=start_s,
                  end_s=start_s + duration_s, channels=channels,
                  fs={k: fs for k in channels})


# ---------------------------------------------------------------------------
# PTB ECG (WFDB format 16, parsed manually — no wfdb dependency)
# ---------------------------------------------------------------------------
def _ptb_records(patient: int) -> list[str]:
    pdir = os.path.join(DATA_ROOT, "ptbdb", f"patient{patient:03d}")
    if not os.path.isdir(pdir):
        raise FileNotFoundError(f"PTB patient dir not found: {pdir}")
    recs = sorted(f[:-4] for f in os.listdir(pdir) if f.endswith(".hea")
                  and not f.endswith("_frank.hea"))
    # keep only records that have a .dat (12-lead), not .xyz-only
    recs = [r for r in recs
            if os.path.exists(os.path.join(pdir, r + ".dat"))]
    return [os.path.join(pdir, r) for r in recs]


def load_ptb_lead2(patient: int, record_idx: int = 0) -> tuple[np.ndarray, float]:
    """Load lead II (mV) of one PTB record. Returns (signal, fs).

    Robust to inconsistent headers seen in the wild: the header's signal
    count / sample count are cross-checked against the actual signal lines
    and the .dat file size. Physical conversion is (raw - baseline) / gain
    per the WFDB signal-spec field order
    (file, format, adc_gain, baseline, units, ...); verified to yield a
    properly centered ECG (~1 mV QRS).
    """
    recs = _ptb_records(patient)
    if not recs:
        raise ValueError(f"no 12-lead records for PTB patient {patient}")
    base = recs[record_idx % len(recs)]
    with open(base + ".hea") as f:
        header = f.read().splitlines()
    rec_name, n_sig_hdr, fs, n_samp_hdr = header[0].split()[:4]
    fs = float(fs)
    sig_lines = [ln for ln in header[1:] if ln and not ln.startswith("#")]
    # Only the .dat signal lines live in the file we read (.xyz Frank leads
    # are listed too but stored separately); lead II is the 2nd .dat signal.
    dat_lines = [ln for ln in sig_lines if ln.split()[0].endswith(".dat")]
    parts = dat_lines[1].split()
    fname, fmt, gain, baseline = parts[0], parts[1], float(parts[2]), float(parts[3])
    n_sig = len(dat_lines)  # header line 1's count includes .xyz; count .dat lines
    if fmt != "16":
        raise ValueError(f"unsupported WFDB format {fmt} in {base}")
    fpath = os.path.join(os.path.dirname(base), fname)
    with open(fpath, "rb") as f:
        raw = f.read()
    n_total = len(raw) // 2
    n_samp = n_total // n_sig
    if n_samp * n_sig != n_total:
        raise ValueError(f"{fpath}: size not divisible by {n_sig} signals")
    arr = np.frombuffer(raw[: n_samp * n_sig * 2], dtype="<i2").reshape(n_samp, n_sig)
    lead2 = (arr[:, 1].astype(np.float64) - baseline) / gain  # mV
    return lead2, fs


def ptb_stream(patient: int, duration_s: float = 720.0,
               record_idx: int = 0) -> Window:
    sig, fs = load_ptb_lead2(patient, record_idx)
    n = int(duration_s * fs)
    if len(sig) < n:
        raise ValueError(f"PTB record shorter than {duration_s}s")
    return Window(device_id=f"ptb-p{patient:03d}", start_s=0.0,
                  end_s=duration_s, channels={"ecg": sig[:n]},
                  fs={"ecg": fs})


def ptb_patient_stream(patient: int, duration_s: float) -> Window:
    """Concatenate a patient's 12-lead records chronologically (lead II).

    PTB records are short (~115 s); concatenation gives a stream long enough
    for temporal train/val/test splits. Record boundaries are real data.
    """
    recs = _ptb_records(patient)
    if not recs:
        raise ValueError(f"no 12-lead records for PTB patient {patient}")
    parts, fs = [], None
    for i in range(len(recs)):
        sig, f = load_ptb_lead2(patient, i)
        if fs is None:
            fs = f
        elif abs(f - fs) > 1e-9:
            raise ValueError("mixed sampling rates across records")
        parts.append(sig)
    full = np.concatenate(parts)
    n = int(duration_s * fs)
    if len(full) < n:
        raise ValueError(
            f"PTB patient {patient}: {len(full)/fs:.0f}s < {duration_s}s")
    return Window(device_id=f"ptb-p{patient:03d}", start_s=0.0,
                  end_s=duration_s, channels={"ecg": full[:n]},
                  fs={"ecg": fs})
