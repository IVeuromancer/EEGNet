"""
Preprocessing pipeline for raw EEG motor imagery epochs.

Steps:
    1. Bandpass filter 8–30 Hz (mu + beta bands)
    2. Baseline correction (subtract mean of pre-cue window)
    3. Artifact rejection (amplitude threshold)
    4. Z-score standardization (fit on train split only)

Usage:
    python src/preprocess.py
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

# ── constants ─────────────────────────────────────────────────────────────────
SRATE = 250
LOWCUT = 8.0
HIGHCUT = 30.0
FILTER_ORDER = 4

ARTIFACT_THRESH_UV = 150.0    # µV peak amplitude rejection threshold
BASELINE_SAMPLES = 125        # 0.5s at 250Hz (pre-imagery baseline within epoch)


# ── filtering ─────────────────────────────────────────────────────────────────
def bandpass_filter(eeg: np.ndarray, lowcut: float, highcut: float,
                    fs: float, order: int = 4) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter.

    Args:
        eeg: shape (n_channels, n_samples)

    Returns:
        filtered eeg, same shape
    """
    sos = butter(order, [lowcut, highcut], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, eeg, axis=-1)


# ── baseline correction ───────────────────────────────────────────────────────
def baseline_correct(eeg: np.ndarray, n_baseline: int = BASELINE_SAMPLES) -> np.ndarray:
    """Subtract per-channel mean of the baseline window.

    Args:
        eeg: shape (n_channels, n_samples)
        n_baseline: number of samples at start of epoch to use as baseline

    Returns:
        baseline-corrected eeg, same shape
    """
    baseline_mean = eeg[:, :n_baseline].mean(axis=1, keepdims=True)
    return eeg - baseline_mean


# ── artifact rejection ────────────────────────────────────────────────────────
def reject_artifacts(X: np.ndarray, y: np.ndarray,
                     threshold: float = ARTIFACT_THRESH_UV) -> tuple[np.ndarray, np.ndarray]:
    """Drop trials where any channel exceeds the amplitude threshold.

    Args:
        X: shape (n_trials, n_channels, n_samples)
        y: shape (n_trials,)

    Returns:
        cleaned X, y with bad trials removed
    """
    peak = np.max(np.abs(X), axis=(1, 2))   # (n_trials,)
    good = peak < threshold
    n_dropped = (~good).sum()
    if n_dropped > 0:
        print(f"  Artifact rejection: dropped {n_dropped}/{len(X)} trials "
              f"({100*n_dropped/len(X):.1f}%)")
    return X[good], y[good]


# ── standardization ───────────────────────────────────────────────────────────
class ChannelStandardizer:
    """Fit per-channel mean/std on training data, apply to any split."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray):
        """X: (n_trials, n_channels, n_samples)"""
        self.mean = X.mean(axis=(0, 2), keepdims=True)   # (1, n_channels, 1)
        self.std = X.std(axis=(0, 2), keepdims=True) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: str):
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: str) -> "ChannelStandardizer":
        data = np.load(path)
        s = cls()
        s.mean = data["mean"]
        s.std = data["std"]
        return s


# ── helpers ───────────────────────────────────────────────────────────────────
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── per-epoch pipeline ────────────────────────────────────────────────────────
def preprocess_epoch(eeg: np.ndarray) -> np.ndarray:
    """Apply filtering and baseline correction to a single epoch.

    Args:
        eeg: (n_channels, n_samples)

    Returns:
        preprocessed eeg, same shape
    """
    eeg = bandpass_filter(eeg, LOWCUT, HIGHCUT, SRATE, FILTER_ORDER)
    eeg = baseline_correct(eeg)
    return eeg


# ── full dataset pipeline ─────────────────────────────────────────────────────
def preprocess_dataset(raw_dir: str = "data/raw",
                       out_path: str = "data/processed/dataset.npz",
                       stats_path: str = "data/processed/standardizer.npz"):
    raw_dir = Path(raw_dir)
    session_files = sorted(raw_dir.glob("session_*.npz"))

    if not session_files:
        raise FileNotFoundError(f"No session files found in {raw_dir}")

    # load per-session electrode metadata if present
    meta_path = Path("session_metadata.json")
    session_meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            session_meta = json.load(f)

    all_X, all_y, all_sessions = [], [], []
    session_records = []

    for i, fp in enumerate(session_files):
        data = np.load(fp)
        X, y = data["eeg"], data["labels"]   # (n_trials, 8, 1000), (n_trials,)
        X = X * 1e-3   # BrainFlow returns nV; convert to µV for thresholds and model
        n_raw = len(y)
        print(f"  Session {i+1}: {fp.name}  — {n_raw} trials")

        # filter and baseline correct each trial
        X_proc = np.stack([preprocess_epoch(X[t]) for t in range(len(X))], axis=0)

        # artifact rejection per session
        X_proc, y = reject_artifacts(X_proc, y)

        all_X.append(X_proc)
        all_y.append(y)
        all_sessions.extend([i] * len(y))
        record = {
            "filename": fp.name,
            "sha256": _sha256_file(fp),
            "n_trials_raw": n_raw,
            "n_trials_kept": int(len(y)),
        }
        if fp.name in session_meta:
            record.update(session_meta[fp.name])
        session_records.append(record)

    X = np.concatenate(all_X, axis=0).astype(np.float32)
    y = np.concatenate(all_y, axis=0).astype(np.int64)
    sessions = np.array(all_sessions, dtype=np.int64)

    print(f"\nTotal: {len(y)} trials | Left: {(y==0).sum()} | Right: {(y==1).sum()}")
    print(f"Shape: {X.shape}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, X=X, y=y, sessions=sessions)

    # full 8-channel standardizer (used by LOSO / dataset-level ops)
    standardizer = ChannelStandardizer()
    standardizer.fit(X)
    standardizer.save(stats_path)
    print(f"Saved to {out_path}")
    print(f"Standardizer saved to {stats_path}")

    # 2-channel standardizer for realtime inference (CH3+CH4 only)
    rt_stats_path = Path(stats_path).with_name("standardizer_realtime.npz")
    rt_standardizer = ChannelStandardizer()
    rt_standardizer.fit(X[:, [2, 3], :])
    rt_standardizer.save(str(rt_stats_path))
    print(f"Realtime standardizer saved to {rt_stats_path}")

    manifest = {
        "preprocessed_at": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": _sha256_file(Path(out_path)),
        "total_trials": int(len(y)),
        "label_counts": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "sessions": session_records,
    }
    manifest_path = Path(out_path).with_name("dataset_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    preprocess_dataset()
