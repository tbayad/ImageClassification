import torch

from image_classification.metrics import confusion_matrix, per_class_accuracy


def test_confusion_matrix_counts_predictions() -> None:
    predictions = torch.tensor([0, 1, 1, 2])
    targets = torch.tensor([0, 1, 2, 2])
    matrix = confusion_matrix(predictions, targets, num_classes=3)
    assert matrix.tolist() == [
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 1],
    ]


def test_per_class_accuracy_handles_empty_rows() -> None:
    matrix = torch.tensor([[2, 0], [0, 0]])
    assert per_class_accuracy(matrix) == [1.0, 0.0]
