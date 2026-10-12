import torch

from ship_detection.model import create_model

def test_model_output_shape():
    model = create_model()

    inputs = torch.randn(4, 3, 80, 80)
    outputs = model(inputs)

    assert outputs.shape == (4, 2)

def test_model_parameter_count():
    model = create_model()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    assert parameter_count == 14194