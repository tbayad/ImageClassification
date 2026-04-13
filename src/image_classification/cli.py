from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from .data import get_data_loaders
from .engine import evaluate, train
from .metrics import (
    confusion_matrix,
    misclassified_examples,
    save_metrics_report,
    stack_batches,
)
from .model import FashionMNISTCNN
from .plots import save_confusion_matrix, save_misclassified_grid, save_training_curves
from .utils import CLASS_NAMES, RunConfig, resolve_device, save_model, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Fashion-MNIST CNN.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--model-name", default="fashion_mnist_cnn.pt")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = RunConfig(
        data_dir=Path(args.data_dir),
        model_dir=Path(args.model_dir),
        reports_dir=Path(args.reports_dir),
        model_name=args.model_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    set_seed(config.seed)
    device = resolve_device()
    print(f"Using device: {device}")

    model = FashionMNISTCNN().to(device)
    train_loader, val_loader, test_loader = get_data_loaders(
        train_batch_size=config.batch_size,
        val_batch_size=config.batch_size,
        data_dir=config.data_dir,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        epochs=config.epochs,
    )
    figures_dir = config.reports_dir / "figures"
    metrics_dir = config.reports_dir / "metrics"
    model_path = save_model(model, config.model_dir, config.model_name)
    save_training_curves(history, figures_dir)

    val_metrics = evaluate(model, val_loader, loss_fn, device, collect_outputs=True)
    test_metrics = evaluate(model, test_loader, loss_fn, device, collect_outputs=True)

    predictions = stack_batches(test_metrics["predictions"])
    targets = stack_batches(test_metrics["targets"])
    images = stack_batches(test_metrics["images"])
    matrix = confusion_matrix(predictions, targets, num_classes=len(CLASS_NAMES))
    bad_images, bad_predictions, bad_targets = misclassified_examples(
        images,
        predictions,
        targets,
    )

    save_confusion_matrix(matrix, CLASS_NAMES, figures_dir / "confusion_matrix.png")
    save_misclassified_grid(
        bad_images,
        bad_predictions,
        bad_targets,
        CLASS_NAMES,
        figures_dir / "misclassified_examples.png",
    )
    save_metrics_report(
        metrics_dir / "validation_metrics.md",
        "validation",
        float(val_metrics["loss"]),
        float(val_metrics["accuracy"]),
        CLASS_NAMES,
        confusion_matrix(
            stack_batches(val_metrics["predictions"]),
            stack_batches(val_metrics["targets"]),
            num_classes=len(CLASS_NAMES),
        ),
    )
    save_metrics_report(
        metrics_dir / "test_metrics.md",
        "test",
        float(test_metrics["loss"]),
        float(test_metrics["accuracy"]),
        CLASS_NAMES,
        matrix,
    )

    print(f"Final validation loss: {float(val_metrics['loss']):.4f}")
    print(f"Final validation accuracy: {float(val_metrics['accuracy']):.2%}")
    print(f"Final test loss: {float(test_metrics['loss']):.4f}")
    print(f"Final test accuracy: {float(test_metrics['accuracy']):.2%}")
    print(f"Model saved to: {model_path}")
