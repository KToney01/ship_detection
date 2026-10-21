import torch
from torch.utils.data import TensorDataset
import pytest
import numpy as np
from PIL import Image
from ship_detection.data import(
    ShipsNetDataset,
    extract_labels_and_groups,
    create_scene_aware_split,
    create_data_loaders
)


def test_shipsnet_dataset_loads_image(tmp_path):
    image_path = tmp_path / "1__test_scene__0.png"

    image = Image.new(
        "RGB",
        (80, 80),
        color=(255, 128, 0)
    )

    image.save(image_path)

    dataset = ShipsNetDataset(tmp_path)

    image_tensor, label = dataset[0]

    assert len(dataset) == 1
    assert image_tensor.shape == (3, 80, 80)
    assert image_tensor.dtype == torch.float32
    assert label.item() == 1
    assert image_tensor.min() >= 0.0
    assert image_tensor.max() <= 1.0



def test_extract_labels_and_groups(tmp_path):
    filenames = [
        "0__scene_a__0.png",
        "1__scene_b__0.png",
        "1__scene_c__0.png"
    ]

    for filename in filenames:
        (tmp_path / filename).touch()

    dataset = ShipsNetDataset(tmp_path)

    labels, groups = extract_labels_and_groups(dataset)

    assert labels.tolist() == [0, 1, 1]
    assert groups.tolist() == [
        "scene_a",
        "scene_b",
        "scene_c"
    ]

def test_shipsnet_dataset_raises_for_empty_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        ShipsNetDataset(tmp_path)


def test_scene_aware_split_prevents_group_leakage():
    labels = np.tile([0, 1], 40)

    groups = np.repeat(
        [f"scene_{i}" for i in range(40)], 
        2
    )

    train_indices, val_indices, test_indices = create_scene_aware_split(
        labels,
        groups,
        seed=42
    )

    train_groups = set(groups[train_indices])
    val_groups = set(groups[val_indices])
    test_groups = set(groups[test_indices])

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

    all_incices = np.concatenate(
        [
            train_indices,
            val_indices,
            test_indices
        ]
    )

    assert len(all_incices) == len(labels)
    assert len(np.unique(all_incices) == len(labels))

def test_create_data_loaders():
    images = torch.randn(20, 3, 80, 80)
    labels = torch.randint(0, 2, (20,))

    dataset = torch.utils.data.TensorDataset(
        images,
        labels
    )

    train_indices = np.arange(0, 12)
    val_indices = np.arange(12, 16)
    test_indices = np.arange(16, 20)

    train_loader, val_loader, test_loader = create_data_loaders(
        dataset,
        train_indices,
        val_indices,
        test_indices,
        batch_size=4
    )

    assert len(train_loader.dataset) == 12
    assert len(val_loader.dataset) == 4
    assert len(test_loader.dataset) == 4

    assert train_loader.batch_size == 4
    assert val_loader.batch_size == 4
    assert test_loader.batch_size == 4    