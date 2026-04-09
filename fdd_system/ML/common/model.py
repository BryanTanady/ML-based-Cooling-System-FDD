from __future__ import annotations

import torch
import torch.nn as nn


class FanSpectrogramCNN(nn.Module):
    def __init__(self, n_classes: int = 2):
        super().__init__()

        self.features = nn.Sequential(
            # (N, 1, F, T) -> (N, 16, F/2, T/2)
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            # (N, 16, F/2, T/2) -> (N, 32, F/4, T/4)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            # (N, 32, F/4, T/4) -> (N, 64, F/8, T/4)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


class ResBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.dropout(self.net(x)))


class Fan1DCNN(nn.Module):
    def __init__(self, n_classes: int, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class Fan1DCNNV2(nn.Module):
    def __init__(self, n_classes: int, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, n_classes),
        )

    def extract_embedding(self, x):
        h = self.features(x)
        h = self.classifier[0](h)
        h = self.classifier[1](h)
        h = self.classifier[2](h)
        return h

    def forward(self, x):
        return self.classifier(self.features(x))


class HybridTimeFreq1DCNN(nn.Module):
    def __init__(self, n_classes: int, in_channels: int = 3, width: int = 48):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, width, kernel_size=9, padding=4),
            nn.BatchNorm1d(width),
            nn.GELU(),
        )
        self.time_branch = nn.Sequential(
            ResBlock1D(width, kernel_size=7, dilation=1, dropout=0.1),
            nn.MaxPool1d(2),
            ResBlock1D(width, kernel_size=5, dilation=2, dropout=0.1),
            nn.MaxPool1d(2),
            ResBlock1D(width, kernel_size=3, dilation=4, dropout=0.1),
            nn.AdaptiveAvgPool1d(1),
        )
        self.freq_branch = nn.Sequential(
            nn.Conv1d(in_channels, width, kernel_size=5, padding=2),
            nn.BatchNorm1d(width),
            nn.GELU(),
            ResBlock1D(width, kernel_size=5, dilation=1, dropout=0.1),
            nn.MaxPool1d(2),
            ResBlock1D(width, kernel_size=3, dilation=2, dropout=0.1),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 2, 128),
            nn.GELU(),
            nn.Dropout(p=0.25),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        time_features = self.time_branch(self.stem(x))
        freq_input = torch.log1p(torch.abs(torch.fft.rfft(x, dim=2)))
        freq_features = self.freq_branch(freq_input)
        return self.classifier(torch.cat([time_features, freq_features], dim=1))


def build_classifier_model(architecture: str, *, n_classes: int, in_channels: int = 3) -> nn.Module:
    architecture_key = str(architecture).strip().lower()
    if architecture_key == "fan1d":
        return Fan1DCNN(n_classes=n_classes, in_channels=in_channels)
    if architecture_key in {"fan1d_v2", "fan1dcnn_v2"}:
        return Fan1DCNNV2(n_classes=n_classes, in_channels=in_channels)
    if architecture_key == "hybrid_timefreq":
        return HybridTimeFreq1DCNN(n_classes=n_classes, in_channels=in_channels)
    if architecture_key in {"spectrogram2d", "fan_spectrogram_cnn"}:
        return FanSpectrogramCNN(n_classes=n_classes)
    raise ValueError(f"Unsupported classifier architecture '{architecture}'.")
