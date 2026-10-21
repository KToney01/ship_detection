import torch

from ship_detection.model import load_model

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



def test_load_model_restores_checkpoint(tmp_path):
    device = torch.device("cpu")

    model = create_model().to(device)

    checkpoint_path = tmp_path / "model.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": 1
        },
        checkpoint_path
    )

    loaded_model = create_model()

    loaded_model, checkpoint = load_model(
        loaded_model,
        checkpoint_path,
        device
    )

    assert next(loaded_model.parameters()).device.type == "cpu"
    assert checkpoint["epoch"] == 1