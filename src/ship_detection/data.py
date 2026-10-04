from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

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