from __future__ import annotations

from pathlib import Path

import torch


def stack_batches(batches: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat(batches, dim=0) if batches else torch.empty(0)


def confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for target, prediction in zip(targets, predictions, strict=False):
        matrix[target.long(), prediction.long()] += 1
    return matrix


def per_class_accuracy(matrix: torch.Tensor) -> list[float]:
    scores: list[float] = []
    for idx in range(matrix.shape[0]):
        total = matrix[idx].sum().item()
        correct = matrix[idx, idx].item()
        scores.append(correct / total if total else 0.0)
    return scores


def misclassified_examples(
    images: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    limit: int = 9,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = predictions != targets
    bad_images = images[mask][:limit]
    bad_predictions = predictions[mask][:limit]
    bad_targets = targets[mask][:limit]
    return bad_images, bad_predictions, bad_targets


def save_metrics_report(
    output_path: str | Path,
    split_name: str,
    loss: float,
    accuracy: float,
    class_names: tuple[str, ...],
    matrix: torch.Tensor,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    class_scores = per_class_accuracy(matrix)
    lines = [
        f"# {split_name.title()} Metrics",
        "",
        f"- loss: {loss:.4f}",
        f"- accuracy: {accuracy:.2%}",
        "",
        "## Per-class Accuracy",
        "",
    ]
    lines.extend(
        f"- {class_name}: {score:.2%}"
        for class_name, score in zip(class_names, class_scores, strict=False)
    )
    output_path.write_text("\n".join(lines) + "\n")
    return output_path
