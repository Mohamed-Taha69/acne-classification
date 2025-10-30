from typing import Tuple
import torch
import torch.nn as nn
from torchvision import models


def build_resnet(model_name: str, num_classes: int, pretrained: bool = True) -> Tuple[nn.Module, int]:
    if model_name == "resnet18":
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
        return backbone, in_features
    elif model_name == "resnet34":
        backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
        return backbone, in_features
    elif model_name == "resnet50":
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
        return backbone, in_features
    elif model_name == "efficientnet_b3":
        backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT if pretrained else None)
        in_features = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Linear(in_features, num_classes)
        return backbone, in_features
    else:
        raise ValueError(f"Unsupported model: {model_name}")


