import pytest
import torch
import torch.nn as nn
from PIL import Image

from ship_detection.predict import predict_image
from ship_detection.data import load_image_tensor

class FixedPredictionModel(nn.Module):
    def forward(self, inputs):
        batch_size= inputs.shape[0]

        return torch.tensor(
            [[0.2, 1.8]],
            dtype=torch.float32
        ).repeat(batch_size, 1)


def test_predict_image_returns_expected_prediction(tmp_path):
    image_path = tmp_path / "test.png"

    image = Image.new(
        "RGB",
        (80, 80),
        color=(128, 128, 128)
    )
    image.save(image_path)

    model = FixedPredictionModel()
    device = torch.device("cpu")

    result = predict_image(
        model, 
        image_path,
        device
    )

    assert result["class_id"] == 1
    assert result["label"] == "ship"

    expected_confidence = torch.softmax(
        torch.tensor([[0.2, 1.8]]),
        dim=1
    )[0, 1].item()

    assert result["confidence"] == pytest.approx(
        expected_confidence
    )


def test_load_image_tensor_rejects_wrong_size(tmp_path):
    image_path = tmp_path / "wrong_size.png"

    image = Image.new(
        "RGB",
        (100, 100)
    )
    image.save(image_path)

    with pytest.raises(ValueError):
        load_image_tensor(image_path)