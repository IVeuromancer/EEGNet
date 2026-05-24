"""
EEG motor imagery model architectures.

- EEGNet: compact, parameter-efficient, best for <500 trials
- UNet1D: deeper encoder-decoder with skip connections, better for 500+ trials

Both accept input shape (batch, n_channels, n_samples) = (B, 8, 1000).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── EEGNet ────────────────────────────────────────────────────────────────────
class EEGNet(nn.Module):
    """
    Lawhern et al. (2018). EEGNet: A Compact Convolutional Neural Network
    for EEG-based Brain-Computer Interfaces.

    Input:  (B, 1, C, T)  where C=n_channels, T=n_samples
    Output: (B, n_classes)
    """

    def __init__(self, n_channels: int = 8, n_samples: int = 1000,
                 n_classes: int = 2, F1: int = 8, D: int = 2, F2: int = 16,
                 dropout: float = 0.5):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples

        # Block 1: temporal conv → depthwise spatial conv
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
            # depthwise: one filter per input channel
            nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )

        # Block 2: depthwise separable conv
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16), padding=(0, 8),
                      groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )

        # compute flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            out = self.block2(self.block1(dummy))
            flat = out.numel()

        self.classifier = nn.Linear(flat, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) → unsqueeze to (B, 1, C, T) for Conv2d
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(1)
        return self.classifier(x)


# ── 1D U-Net ──────────────────────────────────────────────────────────────────
class ConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, dropout: float = 0.3):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ELU(),
        )

    def forward(self, x):
        return self.net(x)


class UNet1D(nn.Module):
    """
    1D U-Net for EEG time series classification.

    Input:  (B, C, T)  C=n_channels, T=n_samples
    Output: (B, n_classes)

    Recommended for 500+ trials.
    """

    def __init__(self, n_channels: int = 8, n_samples: int = 1000,
                 n_classes: int = 2, base_filters: int = 32, dropout: float = 0.3):
        super().__init__()

        f = base_filters

        # encoder
        self.enc1 = ConvBlock1D(n_channels, f, dropout=dropout)
        self.enc2 = ConvBlock1D(f, f * 2, dropout=dropout)
        self.enc3 = ConvBlock1D(f * 2, f * 4, dropout=dropout)

        self.pool = nn.MaxPool1d(2)

        # bottleneck
        self.bottleneck = ConvBlock1D(f * 4, f * 8, dropout=dropout)

        # decoder
        self.up3 = nn.ConvTranspose1d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock1D(f * 8, f * 4, dropout=dropout)

        self.up2 = nn.ConvTranspose1d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock1D(f * 4, f * 2, dropout=dropout)

        self.up1 = nn.ConvTranspose1d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = ConvBlock1D(f * 2, f, dropout=dropout)

        # global average pool → classify
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(f, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        # decoder with skip connections (crop if sizes mismatch due to odd dims)
        d3 = self.up3(b)
        d3 = self._crop_cat(d3, e3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._crop_cat(d2, e2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._crop_cat(d1, e1)
        d1 = self.dec1(d1)

        return self.head(d1)

    @staticmethod
    def _crop_cat(upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Trim upsampled to match skip length, then concatenate along channel dim."""
        min_len = min(upsampled.size(-1), skip.size(-1))
        return torch.cat([upsampled[..., :min_len], skip[..., :min_len]], dim=1)


# ── factory ───────────────────────────────────────────────────────────────────
def build_model(arch: str = "eegnet", **kwargs) -> nn.Module:
    if arch == "eegnet":
        return EEGNet(**kwargs)
    elif arch == "unet1d":
        return UNet1D(**kwargs)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
