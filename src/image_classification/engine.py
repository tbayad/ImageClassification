from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def batch_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    collect_outputs: bool = False,
) -> dict[str, float | list[torch.Tensor]]:
    model.eval()
    losses: list[float] = []
    accuracies: list[float] = []
    predictions: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []
    images: list[torch.Tensor] = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            losses.append(loss_fn(logits, targets).item())
            accuracies.append(batch_accuracy(logits, targets))
            if collect_outputs:
                predictions.append(logits.argmax(dim=1).cpu())
                targets_list.append(targets.cpu())
                images.append(inputs.cpu())

    metrics: dict[str, float | list[torch.Tensor]] = {
        "loss": sum(losses) / len(losses),
        "accuracy": sum(accuracies) / len(accuracies),
    }
    if collect_outputs:
        metrics["predictions"] = predictions
        metrics["targets"] = targets_list
        metrics["images"] = images
    return metrics


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 2,
) -> dict[str, list[float]]:
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        running_losses: list[float] = []
        running_accuracies: list[float] = []

        for step, (inputs, targets) in enumerate(train_loader, start=1):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()

            running_losses.append(loss.item())
            running_accuracies.append(batch_accuracy(logits, targets))

            if step % 100 == 0 or step == 1:
                current = step * len(inputs)
                total = len(train_loader.dataset)
                print(f"epoch {epoch} | loss: {loss.item():.4f} [{current:>5d}/{total:>5d}]")

        val_metrics = evaluate(model, val_loader, loss_fn, device)
        train_loss = sum(running_losses) / len(running_losses)
        train_accuracy = sum(running_accuracies) / len(running_accuracies)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        print(
            f"epoch {epoch} summary | "
            f"train_acc: {train_accuracy:.2%} | val_acc: {val_metrics['accuracy']:.2%}"
        )

    return history
