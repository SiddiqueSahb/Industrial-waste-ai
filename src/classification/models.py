"""Model factory — append a builder to MODEL_BUILDERS as you train each new model in Colab.

The classifier head matches what was actually trained:
    classifier[2] = Sequential(Dropout(p=dropout), Linear(in_features, num_classes))
"""
from __future__ import annotations
from typing import Callable, Dict
import torch
import torch.nn as nn
from torchvision import models as tvm

try:
    import timm  # required only for ConvNeXtV2 (real V2 weights aren't in torchvision)
except ImportError:
    timm = None


# ConvNeXt-Tiny (V1)
def build_convnext_tiny(num_classes: int = 28, dropout: float = 0.3,
                        pretrained: bool = False) -> nn.Module:
    weights = tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.convnext_tiny(weights=weights)
    in_features = model.classifier[2].in_features  # 768
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


# ConvNeXt V2-Tiny -- via timm (real V2 with FCMAE-pretrained weights)
# Must match COLAB_MULTI_MODEL_V2.py:
#     timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k',
#                       pretrained=..., num_classes=..., drop_rate=...)
# Final classifier is `head.fc` (a single Linear). The "dropout" knob maps to
# timm's `drop_rate`, which inserts dropout before head.fc internally.
def build_convnextv2_tiny(num_classes: int = 28, dropout: float = 0.3,
                          pretrained: bool = False) -> nn.Module:
    if timm is None:
        raise ImportError(
            "timm is required for convnextv2_tiny. Install with: pip install -U timm"
        )
    # When loading user-trained weights we still want the same module layout,
    # but we don't need the pretrained weights downloaded again.
    model = timm.create_model(
        'convnextv2_tiny.fcmae_ft_in22k_in1k',
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout,
    )
    return model




# ViT-B/16
def build_vit_b_16(num_classes: int = 28, dropout: float = 0.1,
                   pretrained: bool = False) -> nn.Module:
    weights = tvm.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.vit_b_16(weights=weights)
    in_features = model.heads.head.in_features
    # Replace .heads.head (NOT .heads itself) so the state-dict keys match
    # the Colab training script.
    model.heads.head = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


# Swin-V2-Tiny
def build_swin_v2_t(num_classes: int = 28, dropout: float = 0.3,
                    pretrained: bool = False) -> nn.Module:
    weights = tvm.Swin_V2_T_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.swin_v2_t(weights=weights)
    in_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


# MaxViT-Tiny -- CNN + Transformer hybrid
# Block layout per stage: MBConv -> block-attention -> grid-attention.
# Final classifier head layout is:
#   classifier = Sequential(AdaptiveAvgPool2d, Flatten, LayerNorm,
#                           Linear, Tanh, Linear)
# We replace classifier[5] (the final Linear) with Dropout + Linear so the
# state-dict keys match the Colab training script.
def build_maxvit_t(num_classes: int = 28, dropout: float = 0.3,
                   pretrained: bool = False) -> nn.Module:
    weights = tvm.MaxVit_T_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.maxvit_t(weights=weights)
    in_features = model.classifier[5].in_features  # 512
    model.classifier[5] = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


# Registry
MODEL_BUILDERS: Dict[str, Callable[..., nn.Module]] = {
    "convnext_tiny":    build_convnext_tiny,
    "convnextv2_tiny":  build_convnextv2_tiny,
    "vit_b_16":         build_vit_b_16,
    "swin_v2_t":        build_swin_v2_t,
    "maxvit_t":         build_maxvit_t,
}


def get_model(name: str, num_classes: int = 28, **kwargs) -> nn.Module:
    if name not in MODEL_BUILDERS:
        raise KeyError(
            f"Unknown model '{name}'. Available: {list(MODEL_BUILDERS)}"
        )
    return MODEL_BUILDERS[name](num_classes=num_classes, **kwargs)