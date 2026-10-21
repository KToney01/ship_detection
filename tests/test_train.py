import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pathlib import Path

from ship_detection.model import create_model
from ship_detection.train import (
    train_one_epoch,
    run_training_epochs
)

def test_train_one_epoch_updates_model():
    torch.manual_seed(42)

    images = torch.randn(
        8,
        3,
        80,
        80
    )

    labels = torch.tensor(
        [0, 1, 0, 1, 0, 1, 0, 1]
    )

    dataset = TensorDataset(
        images,
        labels
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False
    )

    device = torch.device("cpu")

    model = create_model().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01
    )

    parameters_before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    loss, accuracy = train_one_epoch(
        model,
        loader,
        criterion,
        optimizer,
        device
    )

    parameters_after = list(
        model.parameters()
    )

    assert loss >= 0.0
    assert 0.0 <= accuracy <= 1.0

    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            parameters_before,
            parameters_after
        )
    )


def test_run_training_epochs_saves_outputs(tmp_path):
    torch.manual_seed(42)

    images = torch.randn(
        8,
        3,
        80,
        80
    )

    labels = torch.tensor(
        [0, 1, 0, 1, 0, 1, 0, 1]
    )


    dataset = TensorDataset(
        images,
        labels
    )

    train_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False
    )

    val_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False
    )

    device = torch.device("cpu")

    model = create_model().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01
    )

    checkpoint_path = tmp_path / "best_model.pt"
    history_path = tmp_path / "training_history.csv"

    history = run_training_epochs(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_epochs=1,
        checkpoint_path=checkpoint_path,
        history_path=history_path
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )
    assert checkpoint["epoch"] == 1
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert "val_f1" in checkpoint

    assert checkpoint_path.exists()
    assert history_path.exists()

    assert len(history) == 1
    assert history[0]["epoch"] == 1

    assert "train_loss" in history[0]
    assert "train_accuracy" in history[0]
    assert "val_loss" in history[0]
    assert "val_accuracy" in history[0]
    assert "val_precision" in history[0]
    assert "val_recall" in history[0]
    assert "val_f1" in history[0]