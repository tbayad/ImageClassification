import torch

from image_classification.model import FashionMNISTCNN


def test_model_output_shape() -> None:
    model = FashionMNISTCNN()
    batch = torch.randn(4, 1, 28, 28)
    output = model(batch)
    assert output.shape == (4, 10)
