from typing import Tuple, Optional
from torchvision import transforms


def build_transforms(img_size: int, aug_cfg: dict) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tfms = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=aug_cfg.get("hflip", 0.5)),
    ]
    if aug_cfg.get("vflip", 0.0) > 0:
        train_tfms.append(transforms.RandomVerticalFlip(p=aug_cfg.get("vflip", 0.0)))
    if aug_cfg.get("rotate", 0):
        train_tfms.append(transforms.RandomRotation(degrees=aug_cfg.get("rotate", 0)))
    if isinstance(aug_cfg.get("affine"), (list, tuple)) and len(aug_cfg.get("affine")) == 2:
        translate, shear = aug_cfg.get("affine")
        train_tfms.append(transforms.RandomAffine(degrees=0, translate=(translate, translate), shear=shear))
    cj = aug_cfg.get("color_jitter")
    if isinstance(cj, (list, tuple)) and len(cj) == 4:
        train_tfms.append(transforms.ColorJitter(*cj))
    train_tfms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if aug_cfg.get("random_erasing", 0.0) > 0:
        train_tfms.append(transforms.RandomErasing(p=aug_cfg.get("random_erasing", 0.0)))

    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transforms.Compose(train_tfms), val_tfms


