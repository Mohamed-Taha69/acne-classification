from typing import Tuple
from torchvision import transforms
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


class RandomBlur:
    """Apply Gaussian blur with given probability."""
    def __init__(self, p=0.5, radius_range=(0.5, 2.0)):
        self.p = p
        self.radius_range = radius_range

    def __call__(self, img):
        if random.random() < self.p:
            radius = random.uniform(*self.radius_range)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img


class RandomSharpen:
    """Apply sharpening with given probability."""
    def __init__(self, p=0.5, factor_range=(1.0, 2.0)):
        self.p = p
        self.factor_range = factor_range

    def __call__(self, img):
        if random.random() < self.p:
            factor = random.uniform(*self.factor_range)
            enhancer = ImageEnhance.Sharpness(img)
            return enhancer.enhance(factor)
        return img


class RandomCutout:
    """Randomly mask out square patches of the image."""
    def __init__(self, p=0.5, n_holes=1, length=16):
        self.p = p
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        # Convert PIL to numpy for easier manipulation
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        
        # Calculate cutout size as percentage of image
        if isinstance(self.length, (int, float)):
            length = int(self.length * min(h, w) / 224)  # Scale based on image size
        else:
            length = random.randint(*[int(l * min(h, w) / 224) for l in self.length])
        
        length = max(1, min(length, min(h, w) // 2))  # Ensure valid size
        
        for _ in range(self.n_holes):
            y = random.randint(0, max(1, h - length))
            x = random.randint(0, max(1, w - length))
            
            # Fill with mean color (better than black)
            if len(img_np.shape) == 3:
                mean_color = img_np.mean(axis=(0, 1))
                img_np[y:y+length, x:x+length] = mean_color
            else:
                mean_color = img_np.mean()
                img_np[y:y+length, x:x+length] = mean_color
        
        return Image.fromarray(img_np.astype(np.uint8))


def build_transforms(img_size: int, aug_cfg: dict) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tfms = [
        transforms.Resize((img_size, img_size)),
    ]
    
    # Blur augmentation (before other transforms)
    if aug_cfg.get("blur", 0.0) > 0:
        train_tfms.append(RandomBlur(p=aug_cfg.get("blur", 0.0)))
    
    # Sharpen augmentation
    if aug_cfg.get("sharpen", 0.0) > 0:
        train_tfms.append(RandomSharpen(p=aug_cfg.get("sharpen", 0.0)))
    
    # Geometric transforms
    train_tfms.append(transforms.RandomHorizontalFlip(p=aug_cfg.get("hflip", 0.5)))
    
    if aug_cfg.get("vflip", 0.0) > 0:
        train_tfms.append(transforms.RandomVerticalFlip(p=aug_cfg.get("vflip", 0.0)))
    
    if aug_cfg.get("rotate", 0):
        train_tfms.append(transforms.RandomRotation(degrees=aug_cfg.get("rotate", 0)))
    
    if isinstance(aug_cfg.get("affine"), (list, tuple)) and len(aug_cfg.get("affine")) == 2:
        translate, shear = aug_cfg.get("affine")
        train_tfms.append(transforms.RandomAffine(degrees=0, translate=(translate, translate), shear=shear))
    
    # Color transforms
    cj = aug_cfg.get("color_jitter")
    if isinstance(cj, (list, tuple)) and len(cj) == 4:
        train_tfms.append(transforms.ColorJitter(*cj))
    
    # Cutout (before tensor conversion)
    if aug_cfg.get("cutout", 0.0) > 0:
        cutout_length = int(0.1 * img_size)  # 10% of image size
        train_tfms.append(RandomCutout(p=aug_cfg.get("cutout", 0.0), length=cutout_length))
    
    # Convert to tensor and normalize
    train_tfms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Random erasing (applied after normalization, similar to cutout)
    if aug_cfg.get("random_erasing", 0.0) > 0:
        train_tfms.append(transforms.RandomErasing(p=aug_cfg.get("random_erasing", 0.0)))

    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transforms.Compose(train_tfms), val_tfms


