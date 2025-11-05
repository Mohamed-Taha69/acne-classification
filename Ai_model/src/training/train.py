import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
import copy

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


def train_one_epoch(model, loader, criterion, optimizer, device, mixup_alpha: float = 0.0, cutmix_alpha: float = 0.0,
                    accumulation_steps: int = 1, gradient_clip: float = 0.0, ema_model=None, ema_decay: float = 0.9999):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()
    
    for batch_idx, (images, targets) in enumerate(tqdm(loader, desc="train", leave=False)):
        images = images.to(device)
        targets = targets.to(device)

        mixed_images, t1, t2, lam, mode = apply_mixup_cutmix(images, targets, mixup_alpha, cutmix_alpha)
        outputs = model(mixed_images)
        if mode in ('mixup', 'cutmix'):
            loss = lam * criterion(outputs, t1) + (1 - lam) * criterion(outputs, t2)
        else:
            loss = criterion(outputs, targets)
        
        # Scale loss for accumulation
        loss = loss / accumulation_steps
        loss.backward()

        # Accumulate gradients
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Update EMA model
            if ema_model is not None:
                update_ema_model(ema_model, model, decay=ema_decay)

        running_loss += loss.item() * images.size(0) * accumulation_steps
        _, predicted = outputs.max(1)
        # For accuracy calculation, use original targets (not mixed)
        if mode in ('mixup', 'cutmix'):
            # Use original targets for accuracy
            correct += (predicted.eq(targets).sum().item())
        else:
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


def update_ema_model(ema_model, model, decay=0.9999):
    """Update EMA model parameters."""
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


def create_ema_model(model):
    """Create EMA model copy."""
    ema_model = copy.deepcopy(model)
    for param in ema_model.parameters():
        param.requires_grad_(False)
    return ema_model


class EarlyStopping:
    """Early stopping utility."""
    def __init__(self, patience=10, min_delta=0.0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'max':
            if score < self.best_score + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        else:  # mode == 'min'
            if score > self.best_score - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        
        return self.early_stop


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
    gradient_clip = float(cfg.get("train.gradient_clip", 0.0))
    accumulation_steps = int(cfg.get("train.accumulation_steps", 1))
    
    # EMA setup
    ema_enabled = cfg.get("train.ema.enabled", False)
    ema_decay = float(cfg.get("train.ema.decay", 0.9999))
    ema_model = None
    if ema_enabled:
        ema_model = create_ema_model(model)
        print("✅ EMA enabled with decay:", ema_decay)
    
    # SWA setup
    swa_enabled = cfg.get("train.swa.enabled", False)
    swa_start_epoch = int(cfg.get("train.swa.start_epoch", epochs // 2))
    swa_lr = float(cfg.get("train.swa.lr", lr * 0.1))
    swa_model = None
    swa_scheduler = None
    if swa_enabled:
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=swa_lr)
        print(f"✅ SWA enabled starting from epoch {swa_start_epoch}")
    
    # Early stopping setup
    early_stopping_cfg = cfg.get("train.early_stopping", {})
    early_stopping = None
    if early_stopping_cfg:
        patience = int(early_stopping_cfg.get("patience", 10))
        min_delta = float(early_stopping_cfg.get("min_delta", 0.0))
        early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, mode='max')
        print(f"✅ Early stopping enabled: patience={patience}, min_delta={min_delta}")

    for epoch in range(1, epochs + 1):
        # Update EMA decay in train_one_epoch call
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, 
            mixup_alpha, cutmix_alpha, accumulation_steps, gradient_clip,
            ema_model if ema_enabled else None, ema_decay
        )
        
        # Validation - use EMA model if enabled
        eval_model = ema_model if (ema_enabled and ema_model is not None) else model
        val_loss, val_acc = validate(eval_model, val_loader, criterion, device)
        
        # Update SWA and scheduler
        swa_val_loss, swa_val_acc = None, None
        if swa_enabled and epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            # Use SWA model for validation after it starts
            swa_val_loss, swa_val_acc = validate(swa_model.module, val_loader, criterion, device)
            print(f"Epoch {epoch:03d} | train_loss {train_loss:.4f} acc {train_acc:.4f} | "
                  f"val_loss {val_loss:.4f} acc {val_acc:.4f} | "
                  f"swa_val_loss {swa_val_loss:.4f} acc {swa_val_acc:.4f}")
        else:
            scheduler.step()
            print(f"Epoch {epoch:03d} | train_loss {train_loss:.4f} acc {train_acc:.4f} | "
                  f"val_loss {val_loss:.4f} acc {val_acc:.4f}")

        # Determine best model to save
        # Priority: SWA > EMA > regular model
        best_model_to_save = None
        best_acc_to_save = val_acc
        
        if swa_enabled and epoch >= swa_start_epoch and swa_val_acc is not None:
            if swa_val_acc > val_acc:
                best_model_to_save = swa_model.module
                best_acc_to_save = swa_val_acc
            else:
                best_model_to_save = eval_model
                best_acc_to_save = val_acc
        else:
            best_model_to_save = eval_model
            best_acc_to_save = val_acc

        # Check for best model
        if best_acc_to_save > best_val_acc:
            best_val_acc = best_acc_to_save
            torch.save({
                "model": best_model_to_save.state_dict(),
                "class_names": class_names,
                "config": cfg.raw,
                "epoch": epoch,
                "val_acc": best_val_acc,
                "is_ema": ema_enabled and best_model_to_save == ema_model,
                "is_swa": swa_enabled and best_model_to_save == swa_model.module if swa_enabled else False,
            }, best_path)
            print(f"💾 Saved new best checkpoint (acc={best_val_acc:.4f}) to: {best_path}")
        
        # Early stopping check (use best accuracy we're tracking)
        if early_stopping is not None:
            if early_stopping(best_acc_to_save):
                print(f"⏹️  Early stopping triggered at epoch {epoch}")
                break

    # Final SWA update if enabled
    if swa_enabled:
        print("🔄 Finalizing SWA model...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model.module, device=device)
        swa_val_loss, swa_val_acc = validate(swa_model.module, val_loader, criterion, device)
        print(f"Final SWA validation: loss={swa_val_loss:.4f}, acc={swa_val_acc:.4f}")
        
        # Save final SWA model if it's better
        if swa_val_acc > best_val_acc:
            torch.save({
                "model": swa_model.module.state_dict(),
                "class_names": class_names,
                "config": cfg.raw,
                "epoch": epoch,
                "val_acc": swa_val_acc,
                "swa": True,
            }, checkpoints_dir / "swa_final.pt")
            print(f"💾 Saved final SWA model (acc={swa_val_acc:.4f})")


if __name__ == "__main__":
    main()


