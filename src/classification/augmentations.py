"""Eval transforms — must match the pipeline used in Colab to get matching numbers.

Standard ImageNet aspect-preserving evaluation:
    Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize
"""
from __future__ import annotations
from typing import Optional
from torchvision import transforms


def get_eval_transform(img_size: int = 224, resize: int = 256,
                       mean=None, std=None):
    mean = mean or [0.485, 0.456, 0.406]
    std  = std  or [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def get_train_transform(img_size: int = 224, mean=None, std=None,
                        use_trivial_augment: bool = True):
    """Provided for completeness — training is done in Colab."""
    mean = mean or [0.485, 0.456, 0.406]
    std  = std  or [0.229, 0.224, 0.225]
    ops = [
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    if use_trivial_augment:
        ops.append(transforms.TrivialAugmentWide())
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25),
    ]
    return transforms.Compose(ops)
