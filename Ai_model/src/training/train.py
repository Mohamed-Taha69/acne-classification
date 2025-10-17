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


def build_optimizer(params, name: str, lr: float, weight_decay: float) -> Optimizer:
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer: Optimizer, name: str, epochs: int) -> _LRScheduler:
    name = name.lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "none":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    raise ValueError(f"Unsupported scheduler: {name}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, targets in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
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

    train_loader, val_loader, inferred_num_classes, class_names = build_dataloaders(
        train_dir, val_dir, img_size, aug_cfg, batch_size, num_workers
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
    scheduler = build_scheduler(optimizer, scheduler_name, epochs)

    criterion = nn.CrossEntropyLoss()

    # Output dirs
    checkpoints_dir = Path(cfg.get("project.checkpoints_dir", "checkpoints"))
    output_dir = Path(cfg.get("project.output_dir", "outputs"))
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    best_path = checkpoints_dir / "best.pt"

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
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


