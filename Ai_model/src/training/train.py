import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from ..utils.config import load_config
from ..utils.seed import set_global_seed
from ..data.acne04_dataset import build_dataloaders
from ..models.resnet import build_resnet
from ..utils.losses import FocalLoss, ClassAwareFocalLoss, BoundaryAwareFocalLoss
import random


def build_optimizer(params, name: str, lr: float, weight_decay: float) -> Optimizer:
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer: Optimizer, name: str, epochs: int, warmup_epochs: int = 0) -> _LRScheduler:
    name = name.lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "cosine_warmup":
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=max(1, warmup_epochs))
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs))
        return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    if name == "none":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    raise ValueError(f"Unsupported scheduler: {name}")


def apply_mixup_cutmix(images: torch.Tensor, targets: torch.Tensor, mixup_alpha: float, cutmix_alpha: float):
    lam = 1.0
    indices = torch.randperm(images.size(0), device=images.device)
    shuffled_targets = targets[indices]

    do_mixup = mixup_alpha > 0 and random.random() < 0.5
    do_cutmix = (not do_mixup) and cutmix_alpha > 0

    if do_mixup:
        lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item()
        images = lam * images + (1 - lam) * images[indices]
        return images, targets, shuffled_targets, lam, 'mixup'
    if do_cutmix:
        lam = torch.distributions.Beta(cutmix_alpha, cutmix_alpha).sample().item()
        _, _, H, W = images.shape
        cut_w = int(W * (1 - lam) ** 0.5)
        cut_h = int(H * (1 - lam) ** 0.5)
        cx = random.randint(0, W)
        cy = random.randint(0, H)
        x1 = max(cx - cut_w // 2, 0)
        y1 = max(cy - cut_h // 2, 0)
        x2 = min(cx + cut_w // 2, W)
        y2 = min(cy + cut_h // 2, H)
        images[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]
        box_area = (x2 - x1) * (y2 - y1)
        lam = 1 - box_area / float(W * H)
        return images, targets, shuffled_targets, lam, 'cutmix'
    return images, targets, targets, lam, 'none'


def train_one_epoch(model, loader, criterion, optimizer, device, mixup_alpha: float = 0.0, cutmix_alpha: float = 0.0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, targets in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        mixed_images, t1, t2, lam, mode = apply_mixup_cutmix(images, targets, mixup_alpha, cutmix_alpha)
        outputs = model(mixed_images)
        if mode in ('mixup', 'cutmix'):
            loss = lam * criterion(outputs, t1) + (1 - lam) * criterion(outputs, t2)
        else:
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    return running_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, targets in tqdm(loader, desc="val", leave=False):
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
    return running_loss / max(1, total), correct / max(1, total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg.get("project.seed", 42)
    set_global_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    img_size = int(cfg.get("data.img_size", 224))
    train_dir = cfg.get("data.train_dir")
    val_dir = cfg.get("data.val_dir")
    batch_size = int(cfg.get("train.batch_size", 32))
    num_workers = int(cfg.get("data.num_workers", 4))
    aug_cfg = cfg.get("aug", {})
    sampler_mode = cfg.get("data.sampler", "none")
    oversample_factors = cfg.get("data.oversample_factors")
    minority_aug = bool(cfg.get("data.minority_aug", False))
    hard_mining = bool(cfg.get("data.hard_mining", False))

    train_loader, val_loader, inferred_num_classes, class_names = build_dataloaders(
        train_dir, val_dir, img_size, aug_cfg, batch_size, num_workers, sampler_mode, oversample_factors, minority_aug, hard_mining
    )

    cfg_num_classes = cfg.get("data.num_classes", inferred_num_classes)

    # Model
    model_name = cfg.get("train.model", "resnet18")
    pretrained = bool(cfg.get("train.pretrained", True))
    model, _ = build_resnet(model_name, num_classes=cfg_num_classes, pretrained=pretrained)
    model.to(device)

    # Optimizer, scheduler, criterion
    lr = float(cfg.get("train.lr", 1e-3))
    wd = float(cfg.get("train.weight_decay", 5e-4))
    optimizer_name = cfg.get("train.optimizer", "adamw")
    optimizer = build_optimizer(model.parameters(), optimizer_name, lr, wd)

    epochs = int(cfg.get("train.epochs", 20))
    scheduler_name = cfg.get("train.scheduler", "cosine")
    warmup_epochs = int(cfg.get("train.warmup_epochs", 0))
    scheduler = build_scheduler(optimizer, scheduler_name, epochs, warmup_epochs)

    # Build loss with optional class weights and focal loss
    loss_name = str(cfg.get("train.loss", "cross_entropy")).lower()
    class_weights_cfg = cfg.get("train.class_weights", "none")
    weight_tensor = None
    if class_weights_cfg == "auto":
        # compute from training targets
        targets = torch.tensor(getattr(train_loader.dataset, 'targets'))
        counts = torch.bincount(targets)
        weights = 1.0 / torch.clamp(counts.float(), min=1.0)
        weights = weights * (len(counts) / weights.sum())
        weight_tensor = weights.to(device)
    elif isinstance(class_weights_cfg, (list, tuple)):
        weight_tensor = torch.tensor(class_weights_cfg, dtype=torch.float).to(device)

    if loss_name == "boundary_aware_focal":
        gamma_per_class = cfg.get("train.focal_gamma", [2.0] * cfg_num_classes)
        confusion_penalty = float(cfg.get("train.confusion_penalty", 2.0))
        criterion = BoundaryAwareFocalLoss(gamma_per_class=gamma_per_class, weight=weight_tensor, confusion_penalty=confusion_penalty)
    elif loss_name == "class_focal":
        gamma_per_class = cfg.get("train.focal_gamma", [2.0] * cfg_num_classes)
        criterion = ClassAwareFocalLoss(gamma_per_class=gamma_per_class, weight=weight_tensor)
    elif loss_name == "focal":
        criterion = FocalLoss(gamma=2.0, weight=weight_tensor)
    else:
        label_smoothing = float(cfg.get("train.label_smoothing", 0.0))
        criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)

    # Output dirs
    checkpoints_dir = Path(cfg.get("project.checkpoints_dir", "checkpoints"))
    output_dir = Path(cfg.get("project.output_dir", "outputs"))
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    best_path = checkpoints_dir / "best.pt"

    mixup_alpha = float(cfg.get("train.mixup", 0.0))
    cutmix_alpha = float(cfg.get("train.cutmix", 0.0))

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, mixup_alpha, cutmix_alpha)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:03d} | train_loss {train_loss:.4f} acc {train_acc:.4f} | val_loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model": model.state_dict(),
                "class_names": class_names,
                "config": cfg.raw,
            }, best_path)
            print(f"Saved new best checkpoint to: {best_path}")


if __name__ == "__main__":
    main()


