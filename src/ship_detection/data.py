from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedGroupKFold


class ShipsNetDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.image_paths = sorted(self.data_dir.glob("*.png"))

        if not self.image_paths:
            raise FileNotFoundError(
                f"No PNG images found in {self.data_dir}"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)

        image_tensor = torch.from_numpy(image_array)
        image_tensor = image_tensor.permute(2, 0, 1)
        image_tensor = image_tensor.float() / 255.0

        label = int(image_path.name.split("__")[0])
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, label_tensor


def extract_labels_and_groups(dataset):
    labels = []
    groups = []

    for image_path in dataset.image_paths:
        parts = image_path.name.split("__")

        if len(parts) < 2:
            raise ValueError(
                f" unexpected ShipsNet filename format: {image_path.name}"
            )

        labels.append(int(parts[0]))
        groups.append(parts[1])

    return np.array(labels), np.array(groups)


def create_scene_aware_split(
        labels,
        groups,
        seed,
        n_splits=20
):
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )

    dummy_x = np.zeros(len(labels))
    folds =[]

    for _, fold_indices in splitter.split(
        dummy_x, 
        labels, 
        groups
    ): 
        folds.append(fold_indices)
    
    train_indices = np.concatenate(folds[:14])
    val_indicies = np.concatenate(folds[14:17])
    test_indices = np.concatenate(folds[17:20])

    return train_indices, val_indicies, test_indices