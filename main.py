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

