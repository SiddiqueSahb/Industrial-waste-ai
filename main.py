"""
Quick smoke test — verifies the project can:
  1. Load YAML config
  2. Discover trained model artifacts
  3. Build the model architecture and load weights
  4. Run prediction on a synthetic image
  5. Build the cross-model comparison table

Run from project root:
    python main.py
"""

"""YAML -> typed dataclasses. Tolerant of partial schemas."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml


@dataclass
class AugConfig:
    image_size: int = 224
    resize_for_eval: int = 256
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std:  List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class PhaseConfig:
    epochs: int = 0
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adamw"
    scheduler: str = "cosine"


@dataclass
class ModelConfig:
    name: str
    family: str
    pretrained: bool = True
    image_size: int = 224
    dropout: float = 0.2
    weights_file: Optional[str] = None
    metrics_file: Optional[str] = None
    history_file: Optional[str] = None
    preds_file:   Optional[str] = None
    phase1: Optional[PhaseConfig] = None
    phase2: Optional[PhaseConfig] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    project: str
    course: str
    num_classes: int
    classes: List[str]
    augmentation: AugConfig
    models: Dict[str, ModelConfig]
    raw: Dict[str, Any] = field(default_factory=dict)


def _parse_model(name: str, m: Dict[str, Any]) -> ModelConfig:
    phase1 = PhaseConfig(**m["phase1"]) if isinstance(m.get("phase1"), dict) else None
    phase2 = PhaseConfig(**m["phase2"]) if isinstance(m.get("phase2"), dict) else None
    return ModelConfig(
        name=name,
        family=m["family"],
        pretrained=bool(m.get("pretrained", True)),
        image_size=int(m.get("image_size", 224)),
        dropout=float(m.get("dropout", 0.2)),
        weights_file=m.get("weights_file"),
        metrics_file=m.get("metrics_file"),
        history_file=m.get("history_file"),
        preds_file=m.get("preds_file"),
        phase1=phase1,
        phase2=phase2,
        raw=m,
    )


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f)

    ds = raw.get("dataset", {})
    aug_raw = raw.get("augmentation", {})
    aug = AugConfig(
        image_size=int(aug_raw.get("image_size", 224)),
        resize_for_eval=int(aug_raw.get("resize_for_eval", 256)),
        mean=list(aug_raw.get("normalize", {}).get("mean", [0.485, 0.456, 0.406])),
        std=list(aug_raw.get("normalize", {}).get("std",  [0.229, 0.224, 0.225])),
    )
    models_raw = raw.get("models", {})
    models = {n: _parse_model(n, m) for n, m in models_raw.items()}

    return Config(
        project=raw.get("project", ""),
        course=raw.get("course", ""),
        num_classes=int(ds.get("num_classes", 28)),
        classes=list(ds.get("classes", [])),
        augmentation=aug,
        models=models,
        raw=raw,
    )
