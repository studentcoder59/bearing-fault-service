"""The 1D-CNN architecture (kept separate so the API can import it without training code)."""
import torch.nn as nn


class CNN1D(nn.Module):
    """Small 1D-CNN: wide first kernel (like a learned filter bank), then two small conv blocks."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64, stride=8, padding=28), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):  # x: (batch, 1, window)
        return self.net(x)
