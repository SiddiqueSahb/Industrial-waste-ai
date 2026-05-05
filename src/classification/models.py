"""Model factory — append a builder to MODEL_BUILDERS as you train each new model in Colab.

The classifier head matches what was actually trained:
    classifier[2] = Sequential(Dropout(p=dropout), Linear(in_features, num_classes))
"""
from __future__ import annotations
from typing import Callable, Dict
import torch.nn as nn
from torchvision import models as tvm


def build_convnext_tiny(num_classes: int = 28, dropout: float = 0.2,
                        pretrained: bool = False) -> nn.Module:
    weights = tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.convnext_tiny(weights=weights)
    in_features = model.classifier[2].in_features  # 768
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def build_efficientnetv2_s(num_classes: int = 28, dropout: float = 0.3,
                           pretrained: bool = False) -> nn.Module:
    weights = tvm.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.efficientnet_v2_s(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def build_vit_b_16(num_classes: int = 28, dropout: float = 0.1,
                   pretrained: bool = False) -> nn.Module:
    weights = tvm.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.vit_b_16(weights=weights)
    in_features = model.heads.head.in_features
    model.heads = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


MODEL_BUILDERS: Dict[str, Callable[..., nn.Module]] = {
    "convnext_tiny":     build_convnext_tiny,
    "efficientnetv2_s":  build_efficientnetv2_s,
    "vit_b_16":          build_vit_b_16,
}


def get_model(name: str, num_classes: int = 28, **kwargs) -> nn.Module:
    if name not in MODEL_BUILDERS:
        raise KeyError(
            f"Unknown model '{name}'. Available: {list(MODEL_BUILDERS)}"
        )
    return MODEL_BUILDERS[name](num_classes=num_classes, **kwargs)

def get_model_names() -> list[str]:
    return list(MODEL_BUILDERS)