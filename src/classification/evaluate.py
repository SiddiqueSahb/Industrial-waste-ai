"""Evaluation module — full per-class report, not just Top-1 accuracy.

For one model (or --all), produces under results/evaluation/<model>/:
    summary.json               top-1/top-5, macro & weighted P/R/F1, AUC
    per_class_metrics.csv      precision / recall / F1 / support per class
    per_class_f1.png           per-class F1 bar chart
    confusion_matrix.png       row-normalised confusion-matrix heatmap
    classification_report.txt  sklearn text report

With --all it also writes results/evaluation/model_comparison.csv (+ .md),
one row per model — ready to paste into the README.

Predictions are read from results/metrics/<model>_preds.npz (produced by
training). Use --run-inference to regenerate them from the checkpoint instead.

Run from the project root:
    python -m src.classification.evaluate --model convnext_tiny
    python -m src.classification.evaluate --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.classification.metrics import (
    compute_per_class,
    compute_summary,
    confusion,
    text_classification_report,
)
from src.config.loader import load_config

RESULTS_DIR = ROOT / "results" / "metrics"
EVAL_DIR = ROOT / "results" / "evaluation"
MODELS_DIR = ROOT / "models" / "classification"
DEFAULT_DATA_DIR = ROOT / "data" / "raw" / "warp_c" / "Warp-C"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate WaRP-C classifiers")
    p.add_argument("--model", help="Model name from configs/classification.yaml")
    p.add_argument("--all", action="store_true", help="Evaluate every model with predictions on disk")
    p.add_argument("--config", default=str(ROOT / "configs" / "classification.yaml"))
    p.add_argument("--preds-suffix", default="", help="e.g. '_smoke' to evaluate smoke-run artifacts")
    p.add_argument("--run-inference", action="store_true",
                   help="Regenerate predictions from the checkpoint instead of reading the saved .npz")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    if not args.model and not args.all:
        p.error("Provide --model NAME or --all")
    return args


def load_or_generate_preds(model_name: str, cfg, args):
    """Return (y_true, y_pred, y_proba) or None if unavailable."""
    mc = cfg.models[model_name]
    stem = (mc.preds_file or f"{model_name}_preds.npz").removesuffix(".npz")
    preds_path = RESULTS_DIR / f"{stem}{args.preds_suffix}.npz"

    if not args.run_inference:
        if not preds_path.exists():
            return None
        data = np.load(preds_path)
        proba_key = "y_proba" if "y_proba" in data.files else "y_prob"
        return data["y_true"], data["y_pred"], data[proba_key]

    # Regenerate from checkpoint
    import torch
    from torch.utils.data import DataLoader
    from src.classification.augmentations import get_eval_transform
    from src.classification.dataset import WarpCDataset
    from src.classification.inference import load_trained_model
    from src.classification.train import collect_predictions, pick_device

    weights = MODELS_DIR / (mc.weights_file or f"{model_name}_best.pth")
    if not weights.exists():
        return None
    device = pick_device(args.device)
    model = load_trained_model(model_name, weights, num_classes=cfg.num_classes,
                               dropout=mc.dropout, device=device)
    tfm = get_eval_transform(
        img_size=mc.image_size,
        resize=mc.raw.get("resize_for_eval", cfg.augmentation.resize_for_eval),
        mean=cfg.augmentation.mean,
        std=cfg.augmentation.std,
    )
    ds = WarpCDataset(Path(args.data_dir) / "test_crops", transform=tfm, class_names=cfg.classes)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    return collect_predictions(model, loader, device)


def plot_confusion_matrix(cm: np.ndarray, class_names, out_path: Path, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{model_name} — confusion matrix (row-normalised)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] >= 0.01:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                        fontsize=5, color="white" if cm[i, j] > 0.5 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_class_f1(df: pd.DataFrame, out_path: Path, model_name: str) -> None:
    df_sorted = df.sort_values("f1")
    fig, ax = plt.subplots(figsize=(9, 10))
    ax.barh(df_sorted["class"], df_sorted["f1"], color="#2b7bba")
    ax.set_xlabel("F1 score")
    ax.set_xlim(0, 1)
    ax.set_title(f"{model_name} — per-class F1")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_model(model_name: str, cfg, args) -> dict | None:
    preds = load_or_generate_preds(model_name, cfg, args)
    if preds is None:
        print(f"[skip] {model_name}: no predictions found "
              f"(train first, or use --run-inference with a checkpoint)")
        return None
    y_true, y_pred, y_proba = preds

    out_dir = EVAL_DIR / f"{model_name}{args.preds_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = compute_summary(y_true, y_pred, y_proba, cfg.num_classes)
    per_class = compute_per_class(y_true, y_pred, cfg.classes)
    cm = confusion(y_true, y_pred, cfg.num_classes, normalize="true")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    per_class.to_csv(out_dir / "per_class_metrics.csv", index=False)
    (out_dir / "classification_report.txt").write_text(
        text_classification_report(y_true, y_pred, cfg.classes)
    )
    plot_confusion_matrix(cm, cfg.classes, out_dir / "confusion_matrix.png", model_name)
    plot_per_class_f1(per_class, out_dir / "per_class_f1.png", model_name)

    print(f"[ok] {model_name}: top1={summary['top1']:.4f} "
          f"macro_f1={summary['macro_f1']:.4f} -> {out_dir.relative_to(ROOT)}/")
    return summary


def write_comparison(rows: dict[str, dict], out_stem: Path) -> None:
    df = pd.DataFrame([
        {
            "Model": name,
            "Top-1": s["top1"],
            "Top-5": s["top5"],
            "Macro Precision": s["macro_precision"],
            "Macro Recall": s["macro_recall"],
            "Macro F1": s["macro_f1"],
            "Weighted F1": s["weighted_f1"],
            "Macro AUC (OvR)": s["macro_auc_ovr"],
        }
        for name, s in rows.items()
    ]).sort_values("Top-1", ascending=False)
    df.to_csv(out_stem.with_suffix(".csv"), index=False)
    df_fmt = df.copy()
    for col in df_fmt.columns[1:]:
        df_fmt[col] = df_fmt[col].map(lambda v: f"{v:.4f}")
    out_stem.with_suffix(".md").write_text(df_fmt.to_markdown(index=False) + "\n")
    print(f"[ok] comparison table -> {out_stem.with_suffix('.csv').relative_to(ROOT)} (+ .md)")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    targets = list(cfg.models) if args.all else [args.model]

    results: dict[str, dict] = {}
    for name in targets:
        if name not in cfg.models:
            raise SystemExit(f"Unknown model {name!r}. Available: {list(cfg.models)}")
        summary = evaluate_model(name, cfg, args)
        if summary is not None:
            results[name] = summary

    if args.all and results:
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        write_comparison(results, EVAL_DIR / "model_comparison")

    if not results:
        raise SystemExit("No models evaluated — nothing to report.")


if __name__ == "__main__":
    main()
