## Acne Classification (Acne04) - Project Scaffold

This repository provides a clean scaffold to train, evaluate, and run inference for an image classification model on the Acne04 dataset.

### Folder Structure

```
Ai_model/
  configs/
    default.yaml
  data/
    raw/            # place original Acne04 here (unmodified)
    processed/      # generated splits (train/val/test) if needed
  scripts/
    prepare_acne04.py
    train.ps1
  src/
    data/acne04_dataset.py
    models/resnet.py
    training/train.py
    evaluation/evaluate.py
    inference/predict.py
    utils/config.py
    utils/seed.py
    utils/transforms.py
  requirements.txt
  .gitignore
  README.md
```

### Prerequisites

- Python 3.9+
- GPU with CUDA (optional but recommended)

### Installation

```bash
python -m venv .venv
.c.venv\Scripts\activate
pip install -r requirements.txt
```

### Prepare Dataset

1) Place the original Acne04 dataset under `data/raw/Acne04`.

2) Optionally, generate train/val splits:

```bash
python scripts/prepare_acne04.py --source data/raw/Acne04 --dest data/processed/acne04 --val_ratio 0.2 --seed 42
```

Ensure the final training data follows an ImageFolder-like structure:

```
data/processed/acne04/
  train/
    class_0/
    class_1/
    ...
  val/
    class_0/
    class_1/
    ...
```

### Configure Training

- `configs/default.yaml` – baseline desktop config (ResNet / EfficientNet)
- `configs/acne_ultimate_colab.yaml` – high-accuracy Google Colab recipe with full preprocessing
- `configs/convnext_colab_fast.yaml` – ConvNeXt-Tiny setup tuned for ~90% accuracy while keeping training under ~2 hours on free Colab

Pick the file that matches your environment and adjust dataset paths if needed.

### Train

```bash
python -m src.training.train --config configs/default.yaml
```

On Windows PowerShell, a convenience script is available:

```powershell
scripts\train.ps1 -ConfigPath "configs/default.yaml"
```

### Evaluate

```bash
python -m src.evaluation.evaluate --config configs/default.yaml --checkpoint checkpoints/best.pt
```

### Inference

```bash
python -m src.inference.predict --checkpoint checkpoints/best.pt --image path/to/image.jpg
```

### Notes

- Swap backbones via `train.model` (supports `resnet18/34/50`, `efficientnet_b3`, `efficientnet_b3_cbam`, `convnext_tiny`).
- Toggle preprocessing in configs with `data.enable_preprocessing` and `data.preprocessing` to balance speed vs. accuracy.
- If your Acne04 layout differs, adjust `scripts/prepare_acne04.py` accordingly.


