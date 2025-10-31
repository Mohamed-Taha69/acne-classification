from typing import Tuple
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import datasets
from ..utils.transforms import build_transforms
import numpy as np


class ClassConditionalTransformDataset(Dataset):
    def __init__(self, base: datasets.ImageFolder, default_transform, per_class_transforms: dict | None = None):
        self.base = base
        self.default_transform = default_transform
        self.per_class_transforms = per_class_transforms or {}
        self.classes = base.classes
        self.targets = base.targets

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        path, target = self.base.samples[idx]
        img = self.base.loader(path)
        transform = self.per_class_transforms.get(target, self.default_transform)
        if transform is not None:
            img = transform(img)
        return img, target


def build_dataloaders(train_dir: str, val_dir: str, img_size: int, aug_cfg: dict, batch_size: int, num_workers: int, sampler_mode: str = "none", oversample_factors=None, minority_aug: bool = False) -> Tuple[DataLoader, DataLoader, int, list]:
    train_tfms, val_tfms = build_transforms(img_size, aug_cfg)

    base_train_ds = datasets.ImageFolder(root=train_dir, transform=None)
    if minority_aug:
        # stronger transforms for Severe (2) and Very_Severe (3)
        severe_tfms, _ = build_transforms(img_size, {
            "hflip": aug_cfg.get("hflip", 0.5),
            "vflip": aug_cfg.get("vflip", 0.15),
            "color_jitter": [0.4, 0.4, 0.4, 0.2],
            "random_erasing": max(aug_cfg.get("random_erasing", 0.0), 0.35),
            "rotate": max(aug_cfg.get("rotate", 0), 35),
            "affine": aug_cfg.get("affine", [0.12, 12]),
        })
        per_class = {2: severe_tfms, 3: severe_tfms}
        train_ds = ClassConditionalTransformDataset(base_train_ds, train_tfms, per_class)
    else:
        train_ds = datasets.ImageFolder(root=train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(root=val_dir, transform=val_tfms)

    class_names = train_ds.classes
    num_classes = len(class_names)

    sampler = None
    if sampler_mode in ("weighted", "weighted_plus"):
        targets = np.array(train_ds.targets)
        class_counts = np.bincount(targets)
        class_weights = 1.0 / np.clip(class_counts, 1, None)
        if sampler_mode == "weighted_plus" and oversample_factors is not None:
            factors = np.array(oversample_factors, dtype=float)
            if factors.shape[0] == class_weights.shape[0]:
                class_weights = class_weights * factors
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


