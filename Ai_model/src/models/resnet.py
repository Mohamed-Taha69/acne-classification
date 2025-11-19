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
    elif model_name == "efficientnet_b3_cbam":
        backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT if pretrained else None)
        # Define a simple CBAM block
        class CBAM(nn.Module):
            def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
                super().__init__()
                self.mlp = nn.Sequential(
                    nn.Linear(channels, channels // reduction),
                    nn.ReLU(inplace=True),
                    nn.Linear(channels // reduction, channels),
                )
                self.spatial = nn.Sequential(
                    nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                b, c, h, w = x.size()
                # channel attention
                gap = nn.functional.adaptive_avg_pool2d(x, 1).view(b, c)
                gmp = nn.functional.adaptive_max_pool2d(x, 1).view(b, c)
                ca = torch.sigmoid(self.mlp(gap) + self.mlp(gmp)).view(b, c, 1, 1)
                x = x * ca
                # spatial attention
                avg = torch.mean(x, dim=1, keepdim=True)
                mx, _ = torch.max(x, dim=1, keepdim=True)
                sa = self.spatial(torch.cat([avg, mx], dim=1))
                x = x * sa
                return x

        # Replace classifier with CBAM + pooling + linear
        features_out = backbone.classifier[1].in_features
        attention = CBAM(features_out)
        classifier = nn.Linear(features_out, num_classes)
        # Build wrapper
        class EfficientNetB3CBAM(nn.Module):
            def __init__(self, b, attn, cls):
                super().__init__()
                self.features = b.features
                self.avgpool = b.avgpool
                self.dropout = getattr(b, "dropout", nn.Dropout(0.3))
                self.attn = attn
                self.classifier = cls

            def forward(self, x):
                x = self.features(x)
                x = self.attn(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.dropout(x)
                x = self.classifier(x)
                return x

        model = EfficientNetB3CBAM(backbone, attention, classifier)
        return model, features_out
    elif model_name == "convnext_tiny":
        backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        in_features = backbone.classifier[2].in_features
        backbone.classifier[2] = nn.Linear(in_features, num_classes)
        return backbone, in_features
    else:
        raise ValueError(f"Unsupported model: {model_name}")


