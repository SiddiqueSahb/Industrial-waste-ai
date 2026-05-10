"""
Quick smoke test — verifies the project can:
  1. Load YAML config
  2. Discover trained model artifacts
  3. Build the model architecture and load weights
  4. Run prediction on a synthetic image
  5. Build the cross-model comparison table

"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image

from src.config.loader import load_config
from src.classification.augmentations import get_eval_transform
from src.classification.inference import load_trained_model, predict_image_topk
from src.analysis.load_artifacts import (
    artifacts_exist, load_metrics, load_history, load_predictions,
)
from src.analysis.comparison_table import build_comparison


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "classification.yaml"
MODELS_DIR  = ROOT / "models" / "classification"


def main():

    print("WaRP-C Industrial Waste Classification — test")
   

    # 1. config
    cfg = load_config(CONFIG_PATH)
    print(f"\nProject     : {cfg.project}")
    print(f"Course      : {cfg.course}")
    print(f"Num classes : {cfg.num_classes}")
    print(f"Models in cfg: {list(cfg.models)}")

    # 2. discover artifacts
    available = [n for n in cfg.models if artifacts_exist(n)]
    print(f"\nModels with artifacts: {available}")
    if not available:
        print("No artifacts found in results/metrics/. Add a *_metrics.pkl to enable.")
        return

    # 3. inspect first model
    name = available[0]
    print(f"\n--- Inspecting `{name}` ---")
    m = load_metrics(name) or {}
    print(f"metrics keys  : {list(m.keys())}")
    h = load_history(name) or {}
    print(f"history keys  : {list(h.keys())}  "
          f"epochs={len(h.get('train_loss', []))}")
    p = load_predictions(name) or {}
    print(f"preds keys    : {list(p.keys())}")
    if "y_true" in p:
        print(f"  y_true     : shape={p['y_true'].shape}")
    if "y_prob" in p:
        print(f"  y_prob     : shape={p['y_prob'].shape}")

    # 4. build model + load weights
    mc = cfg.models[name]
    weights = MODELS_DIR / (mc.weights_file or f"{name}_best.pth")
    print(f"\nLoading weights: {weights}")
    model = load_trained_model(
        model_name=name, weights_path=weights,
        num_classes=cfg.num_classes, dropout=mc.dropout,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded `{name}`  ({n_params/1e6:.2f}M params)")

    # 5. dummy inference
    rng = np.random.default_rng(0)
    fake = (rng.random((mc.image_size, mc.image_size, 3)) * 255).astype(np.uint8)
    img = Image.fromarray(fake, mode="RGB")
    tfm = get_eval_transform(
        img_size=mc.image_size,
        resize=cfg.augmentation.resize_for_eval,
        mean=cfg.augmentation.mean, std=cfg.augmentation.std,
    )
    top5 = predict_image_topk(model, img, tfm, cfg.classes, k=5)
    print("\nTop-5 prediction on a random image:")
    for cn, prob in top5:
        print(f"  {cn:30s} {prob*100:6.2f}%")

    # 6. comparison table
    print("\n--- Cross-model comparison ---")
    df = build_comparison(available)
    print(df.to_string(index=False))
    print("\nOK.")


if __name__ == "__main__":
    main()

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
