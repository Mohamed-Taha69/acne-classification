import argparse
import torch
from sklearn.metrics import classification_report, confusion_matrix
from ..utils.config import load_config
from ..data.acne04_dataset import build_dataloaders
from ..models.resnet import build_resnet
from torchvision import transforms
from torchvision.transforms import functional as F


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data (use val set for evaluation here)
    img_size = int(cfg.get("data.img_size", 224))
    train_dir = cfg.get("data.train_dir")
    val_dir = cfg.get("data.val_dir")
    batch_size = int(cfg.get("train.batch_size", 32))
    num_workers = int(cfg.get("data.num_workers", 4))
    aug_cfg = cfg.get("aug", {})

    _, val_loader, num_classes, class_names = build_dataloaders(
        train_dir, val_dir, img_size, aug_cfg, batch_size, num_workers
    )

    # Model
    model_name = cfg.get("train.model", "resnet18")
    model, _ = build_resnet(model_name, num_classes=num_classes, pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    tta = int(cfg.get("infer.tta", 0))
    for images, targets in val_loader:
        images = images.to(device)
        if tta <= 1:
            outputs = model(images)
        else:
            # simple TTA: original + hflip + vflip + h+v (limited by tta)
            variants = [images]
            if tta >= 2:
                variants.append(torch.flip(images, dims=[3]))  # hflip
            if tta >= 3:
                variants.append(torch.flip(images, dims=[2]))  # vflip
            if tta >= 4:
                variants.append(torch.flip(images, dims=[2, 3]))  # hv
            logits = None
            for v in variants[:tta]:
                out = model(v)
                logits = out if logits is None else logits + out
            outputs = logits / len(variants[:tta])
        pred = outputs.argmax(dim=1).cpu()
        all_preds.extend(pred.tolist())
        all_targets.extend(targets.tolist())

    print(classification_report(all_targets, all_preds, target_names=class_names, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(all_targets, all_preds))


if __name__ == "__main__":
    main()


