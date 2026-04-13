from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import random

import torch
from torch import nn

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional at runtime
    np = None


CLASS_NAMES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)
FASHION_MNIST_MEAN = 0.2860
FASHION_MNIST_STD = 0.3530


@dataclass(slots=True)
class RunConfig:
    data_dir: Path = Path("data")
    model_dir: Path = Path("models")
    reports_dir: Path = Path("reports")
    model_name: str = "fashion_mnist_cnn.pt"
    batch_size: int = 128
    epochs: int = 2
    learning_rate: float = 1e-4
    val_ratio: float = 0.1
    seed: int = 42


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(model: nn.Module, model_dir: str | Path, model_name: str) -> Path:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / model_name
    torch.save(model.state_dict(), model_path)
    return model_path


def load_model(
    model: nn.Module,
    model_dir: str | Path,
    model_name: str,
    device: torch.device,
) -> nn.Module:
    model_path = Path(model_dir) / model_name
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model
