from typing import Tuple
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets
from ..utils.transforms import build_transforms
import numpy as np


def build_dataloaders(train_dir: str, val_dir: str, img_size: int, aug_cfg: dict, batch_size: int, num_workers: int, sampler_mode: str = "none") -> Tuple[DataLoader, DataLoader, int, list]:
    train_tfms, val_tfms = build_transforms(img_size, aug_cfg)

    train_ds = datasets.ImageFolder(root=train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(root=val_dir, transform=val_tfms)

    class_names = train_ds.classes
    num_classes = len(class_names)

    sampler = None
    if sampler_mode == "weighted":
        targets = np.array(train_ds.targets)
        class_counts = np.bincount(targets)
        class_weights = 1.0 / np.clip(class_counts, 1, None)
        sample_weights = class_weights[targets]
        sampler = WeightedRandomSampler(sample_weights.tolist(), num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, num_classes, class_names


