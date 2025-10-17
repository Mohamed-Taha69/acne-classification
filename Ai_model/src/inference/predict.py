import argparse
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from ..models.resnet import build_resnet


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    class_names = ckpt.get("class_names")
    cfg = ckpt.get("config", {})

    num_classes = len(class_names) if class_names else int(cfg.get("data", {}).get("num_classes", 2))
    model_name = cfg.get("train", {}).get("model", "resnet18")
    model, _ = build_resnet(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((int(cfg.get("data", {}).get("img_size", 224)), int(cfg.get("data", {}).get("img_size", 224)))),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(args.image).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    logits = model(x)
    prob = torch.softmax(logits, dim=1)
    conf, pred_idx = prob.max(dim=1)
    pred_idx = pred_idx.item()
    conf = conf.item()
    pred_label = class_names[pred_idx] if class_names else str(pred_idx)
    print({"label": pred_label, "confidence": round(conf, 4)})


if __name__ == "__main__":
    main()


