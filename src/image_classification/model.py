from __future__ import annotations

import torch
from torch import nn


class FashionMNISTCNN(nn.Module):
    """Notebook-derived CNN for Fashion-MNIST."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.Dropout(p=0.5),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6272, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.features(inputs)
        return self.classifier(x)
