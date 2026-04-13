from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
import torch
from torchvision import transforms

from .model import FashionMNISTCNN
from .utils import (
    CLASS_NAMES,
    FASHION_MNIST_MEAN,
    FASHION_MNIST_STD,
    load_model,
    resolve_device,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument("image", help="Path to a grayscale clothing image.")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--model-name", default="fashion_mnist_cnn.pt")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device()
    model = FashionMNISTCNN().to(device)
    load_model(model, args.model_dir, args.model_name, device)

    image = Image.open(Path(args.image)).convert("L").resize((28, 28))
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((FASHION_MNIST_MEAN,), (FASHION_MNIST_STD,)),
        ]
    )
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        prediction = logits.argmax(dim=1).item()

    print(f"Predicted class: {CLASS_NAMES[prediction]}")
