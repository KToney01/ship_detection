import pytest
import torch

from ship_detection.cli import resolve_device

def test_resolve_device_cpu():
    device = resolve_device("cpu")

    assert device.type == "cpu"

    def test_resolve_device_auto():
        device = resolve_device("auto")

        expected = (
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        assert device.type == expected

def test_resolve_device_rejects_unavailable_cuda():
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this machine.")
        
    with pytest.raises(RuntimeError):
        resolve_device("cuda")