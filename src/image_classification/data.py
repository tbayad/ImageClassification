from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from .utils import FASHION_MNIST_MEAN, FASHION_MNIST_STD


def get_data_loaders(
    train_batch_size: int = 128,
    val_batch_size: int = 128,
    data_dir: str | Path = "data",
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create reproducible train/validation/test dataloaders."""

    data_dir = Path(data_dir)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((FASHION_MNIST_MEAN,), (FASHION_MNIST_STD,)),
        ]
    )

    train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    val_size = int(len(train_dataset) * val_ratio)
    train_size = len(train_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(train_subset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=val_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
