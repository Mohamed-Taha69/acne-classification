import argparse
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image


def is_image_file(path: Path) -> bool:
    """Check if file is a valid image."""
    try:
        Image.open(path).verify()
        return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    except Exception:
        return False


def collect_class_images(source_dir: Path):
    """Collect all image files from class directories."""
    class_to_images = {}
    for cls_dir in sorted([p for p in source_dir.iterdir() if p.is_dir()]):
        images = []
        for img_path in cls_dir.rglob("*"):
            if img_path.is_file() and is_image_file(img_path):
                images.append(img_path)
        if images:
            class_to_images[cls_dir.name] = images
            print(f"Found {len(images)} images in class: {cls_dir.name}")
    return class_to_images


def split_and_copy(class_to_images, dest_dir: Path, val_ratio: float, seed: int):
    """Perform stratified 80/20 split and copy files."""
    from sklearn.model_selection import train_test_split
    
    train_root = dest_dir / "train"
    val_root = dest_dir / "val"
    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)

    total_train = 0
    total_val = 0

    for cls_name, images in class_to_images.items():
        images = list(images)
        if len(images) == 0:
            continue
        
        # Create class directories
        (train_root / cls_name).mkdir(exist_ok=True)
        (val_root / cls_name).mkdir(exist_ok=True)
        
        # Stratified split (80/20)
        train_images, val_images = train_test_split(
            images, 
            test_size=val_ratio, 
            random_state=seed,
            shuffle=True
        )
        
        # Copy training images
        for img_path in train_images:
            dest_path = train_root / cls_name / img_path.name
            if not dest_path.exists():
                shutil.copy2(img_path, dest_path)
        total_train += len(train_images)
        
        # Copy validation images
        for img_path in val_images:
            dest_path = val_root / cls_name / img_path.name
            if not dest_path.exists():
                shutil.copy2(img_path, dest_path)
        total_val += len(val_images)
        
        print(f"Class {cls_name}:")
        print(f"  Train: {len(train_images)} images")
        print(f"  Val: {len(val_images)} images")
    
    print(f"\nTotal:")
    print(f"  Train: {total_train} images")
    print(f"  Val: {total_val} images")
    print(f"  Total: {total_train + total_val} images")


def main():
    parser = argparse.ArgumentParser(description="Split Vitiligo dataset into train/val with 80/20 stratified split.")
    parser.add_argument("--source", type=str, default="data/raw/vitiligo", help="Path to raw vitiligo folder with Healthy/Vitiligo subfolders.")
    parser.add_argument("--dest", type=str, default="data/processed/vitiligo", help="Destination root to create train/val splits.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio (default: 0.2 for 80/20).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    args = parser.parse_args()

    source_dir = Path(args.source)
    dest_dir = Path(args.dest)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    print(f"Collecting images from: {source_dir}")
    class_to_images = collect_class_images(source_dir)
    if not class_to_images:
        raise RuntimeError("No class subfolders with images were found in source.")

    print(f"\nPerforming {int((1-args.val_ratio)*100)}/{int(args.val_ratio*100)} stratified split...")
    dest_dir.mkdir(parents=True, exist_ok=True)
    split_and_copy(class_to_images, dest_dir, args.val_ratio, args.seed)
    print(f"\nDone. Created train/val splits under: {dest_dir}")


if __name__ == "__main__":
    main()

