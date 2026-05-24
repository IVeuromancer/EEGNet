"""PyTorch Dataset for preprocessed EEG motor imagery epochs."""

import numpy as np
import torch
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    """
    Args:
        X: (n_trials, n_channels, n_samples) float32
        y: (n_trials,) int64  — 0=left, 1=right
        augment: whether to apply training augmentations
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X = torch.from_numpy(X)   # (N, 8, 1000)
        self.y = torch.from_numpy(y)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx].clone()   # (8, 1000)

        if self.augment:
            x = self._gaussian_noise(x)
            x = self._temporal_jitter(x)

        return x, self.y[idx]

    # ── augmentations ─────────────────────────────────────────────────────────
    def _gaussian_noise(self, x: torch.Tensor, std: float = 0.05) -> torch.Tensor:
        return x + torch.randn_like(x) * std

    def _temporal_jitter(self, x: torch.Tensor, max_shift: int = 12) -> torch.Tensor:
        """Randomly shift the signal along the time axis by up to max_shift samples."""
        shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        return torch.roll(x, shift, dims=-1)


def make_splits(X: np.ndarray, y: np.ndarray, channels=None,
                train_ratio: float = 0.70, dev_ratio: float = 0.15, seed: int = 42):
    """Stratified train/dev/test split over all trials.

    Args:
        channels: list of 0-indexed channel indices to select, or None for all
        train_ratio, dev_ratio: fractions; test gets the remainder
    """
    from sklearn.model_selection import train_test_split
    from src.preprocess import ChannelStandardizer

    if channels is not None:
        X = X[:, channels, :]

    test_ratio = 1.0 - train_ratio - dev_ratio
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=(1.0 - train_ratio), stratify=y, random_state=seed
    )
    dev_of_tmp = dev_ratio / (dev_ratio + test_ratio)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_tmp, y_tmp, test_size=(1.0 - dev_of_tmp), stratify=y_tmp, random_state=seed
    )

    standardizer = ChannelStandardizer()
    X_tr = standardizer.fit_transform(X_tr)
    X_dev = standardizer.transform(X_dev)
    X_test = standardizer.transform(X_test)

    train_ds = EEGDataset(X_tr.astype(np.float32), y_tr.astype(np.int64), augment=True)
    dev_ds   = EEGDataset(X_dev.astype(np.float32), y_dev.astype(np.int64), augment=False)
    test_ds  = EEGDataset(X_test.astype(np.float32), y_test.astype(np.int64), augment=False)

    return train_ds, dev_ds, test_ds, standardizer


def make_loso_splits(X: np.ndarray, y: np.ndarray, sessions: np.ndarray,
                     test_session: int, standardizer=None):
    """Leave-one-session-out split.

    Returns train and test EEGDatasets with standardization fit on train only.
    """
    from src.preprocess import ChannelStandardizer

    train_mask = sessions != test_session
    test_mask = sessions == test_session

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    if standardizer is None:
        standardizer = ChannelStandardizer()
        X_train = standardizer.fit_transform(X_train)
    else:
        X_train = standardizer.transform(X_train)

    X_test = standardizer.transform(X_test)

    train_ds = EEGDataset(X_train.astype(np.float32), y_train.astype(np.int64), augment=True)
    test_ds = EEGDataset(X_test.astype(np.float32), y_test.astype(np.int64), augment=False)

    return train_ds, test_ds, standardizer
