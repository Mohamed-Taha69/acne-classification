import argparse
import shutil
from pathlib import Path
import random


def collect_class_images(source_dir: Path):
    class_to_images = {}
    for cls_dir in sorted([p for p in source_dir.iterdir() if p.is_dir()]):
        images = [p for p in cls_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        if images:
            class_to_images[cls_dir.name] = images
    return class_to_images


def split_and_copy(class_to_images, dest_dir: Path, val_ratio: float, seed: int):
    random.seed(seed)
    train_root = dest_dir / "train"
    val_root = dest_dir / "val"
    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)

    for cls_name, images in class_to_images.items():
        images = list(images)
        random.shuffle(images)
        n_val = max(1, int(len(images) * val_ratio)) if len(images) > 0 else 0
        val_images = set(images[:n_val])

        (train_root / cls_name).mkdir(exist_ok=True)
        (val_root / cls_name).mkdir(exist_ok=True)

        for img_path in images:
            target_root = val_root if img_path in val_images else train_root
            dest_path = target_root / cls_name / img_path.name
            if not dest_path.exists():
                shutil.copy2(img_path, dest_path)


def main():
    parser = argparse.ArgumentParser(description="Prepare Acne04 dataset splits into ImageFolder layout.")
    parser.add_argument("--source", type=str, required=True, help="Path to original Acne04 folder with class subfolders.")
    parser.add_argument("--dest", type=str, required=True, help="Destination root to create train/val splits.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    args = parser.parse_args()

    source_dir = Path(args.source)
    dest_dir = Path(args.dest)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    class_to_images = collect_class_images(source_dir)
    if not class_to_images:
        raise RuntimeError("No class subfolders with images were found in source.")

    dest_dir.mkdir(parents=True, exist_ok=True)
    split_and_copy(class_to_images, dest_dir, args.val_ratio, args.seed)
    print(f"Done. Created train/val splits under: {dest_dir}")


if __name__ == "__main__":
    main()


