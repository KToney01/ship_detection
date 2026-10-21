import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ship_detection.evaluate import evaluate

class IdentityModel(nn.Module):
    def forward(self, inputs):
        return inputs

def test_evaluate_returns_correct_metrics():
    logits = torch.tensor(
        [
            [0.1, 0.9], # Predict 1, actual 1 -> TP
            [0.9, 0.1], # Predict 0, actual 0 -> TN
            [0.1, 0.9], # Predict 1, actual 0 -> FP
            [0.9, 0.1], # Predict 0, actual 1 -> FN
        ],
        dtype=torch.float32
    )

    labels = torch.tensor(
        [1, 0, 0, 1],
        dtype=torch.long
    )

    dataset = TensorDataset(
        logits,
        labels
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False
    )

    model = IdentityModel()
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")

    (
        loss,
        accuracy,
        precision,
        recall,
        f1,
        true_positive,
        true_negative,
        false_positive,
        false_negative
    ) = evaluate(
        model,
        loader,
        criterion,
        device
    )

    assert loss >= 0.0

    assert accuracy == pytest.approx(0.5)
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(0.5)
    assert f1 == pytest.approx(0.5)

    assert true_positive == 1
    assert true_negative == 1
    assert false_positive == 1
    assert false_negative == 1