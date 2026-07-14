"""Train a WaRP-C classifier with MLflow experiment tracking.

Reproduces the Colab pipeline (pretrained backbone, AdamW + cosine schedule,
label smoothing) for any architecture registered in MODEL_BUILDERS, and logs
hyperparameters, per-epoch metrics, and the final model artifact to MLflow.

Run from the project root, e.g.:

    python -m src.classification.train --model convnext_tiny --epochs 30

Quick smoke test (tiny subset, 1 epoch):

    python -m src.classification.train --model convnext_tiny \
        --epochs 1 --limit-per-class 4 --batch-size 8

Artifacts are written both to MLflow and to the paths the Streamlit app
already reads (models/classification/, results/metrics/).
"""
from __future__ import annotations

import argparse
import pickle
import random
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.classification.augmentations import get_eval_transform, get_train_transform
from src.classification.dataset import WarpCDataset
from src.classification.metrics import compute_summary
from src.classification.models import get_model
from src.config.loader import load_config

DEFAULT_DATA_DIR = ROOT / "data" / "raw" / "warp_c" / "Warp-C"
MODELS_DIR = ROOT / "models" / "classification"
RESULTS_DIR = ROOT / "results" / "metrics"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a WaRP-C classifier with MLflow tracking")
    p.add_argument("--model", required=True, help="Architecture name from configs/classification.yaml")
    p.add_argument("--config", default=str(ROOT / "configs" / "classification.yaml"))
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    p.add_argument("--experiment", default="warp-c-classification")
    p.add_argument("--limit-per-class", type=int, default=None,
                   help="Cap images per class — for smoke tests only")
    p.add_argument("--artifact-suffix", default=None,
                   help="Suffix appended to output filenames. Defaults to '_smoke' "
                        "when --limit-per-class is set, so smoke runs never "
                        "overwrite real trained artifacts.")
    p.add_argument("--log-pytorch-model", action="store_true",
                   help="Also log the full model in MLflow pytorch flavor (larger runs)")
    args = p.parse_args()
    if args.artifact_suffix is None:
        args.artifact_suffix = "_smoke" if args.limit_per_class else ""
    return args


def pick_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over `loader`. Trains if an optimizer is given, else evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += labels.size(0)

    return total_loss / max(seen, 1), correct / max(seen, 1)


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    trues, preds, probas = [], [], []
    for images, labels in loader:
        logits = model(images.to(device))
        proba = torch.softmax(logits, dim=1).cpu().numpy()
        probas.append(proba)
        preds.append(proba.argmax(1))
        trues.append(labels.numpy())
    return np.concatenate(trues), np.concatenate(preds), np.concatenate(probas)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    cfg = load_config(args.config)
    if args.model not in cfg.models:
        raise SystemExit(f"Unknown model {args.model!r}. Available: {list(cfg.models)}")
    mc = cfg.models[args.model]

    device = pick_device(args.device)
    print(f"[train] model={args.model}  device={device.type}  epochs={args.epochs}")

    data_dir = Path(args.data_dir)
    train_tfm = get_train_transform(mc.image_size, cfg.augmentation.mean, cfg.augmentation.std)
    eval_tfm = get_eval_transform(
        img_size=mc.image_size,
        resize=mc.raw.get("resize_for_eval", cfg.augmentation.resize_for_eval),
        mean=cfg.augmentation.mean,
        std=cfg.augmentation.std,
    )

    full_train = WarpCDataset(data_dir / "train_crops", transform=None,
                              class_names=cfg.classes, limit_per_class=args.limit_per_class)
    test_ds = WarpCDataset(data_dir / "test_crops", transform=eval_tfm,
                           class_names=cfg.classes, limit_per_class=args.limit_per_class)

    n_val = max(1, int(len(full_train) * args.val_fraction))
    generator = torch.Generator().manual_seed(args.seed)
    train_subset, val_subset = random_split(full_train, [len(full_train) - n_val, n_val],
                                            generator=generator)

    # Wrap subsets so train/val get different transforms over the same folder scan
    class _Transformed(torch.utils.data.Dataset):
        def __init__(self, subset, tfm):
            self.subset, self.tfm = subset, tfm

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, i):
            img, label = self.subset[i]
            return self.tfm(img), label

    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                         pin_memory=(device.type == "cuda"))
    train_loader = DataLoader(_Transformed(train_subset, train_tfm), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(_Transformed(val_subset, eval_tfm), shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    print(f"[train] samples: train={len(train_subset)}  val={len(val_subset)}  test={len(test_ds)}")

    model = get_model(args.model, num_classes=cfg.num_classes,
                      pretrained=mc.pretrained, dropout=mc.dropout).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def _with_suffix(filename: str) -> str:
        stem, dot, ext = filename.rpartition(".")
        return f"{stem}{args.artifact_suffix}{dot}{ext}" if args.artifact_suffix else filename

    weights_path = MODELS_DIR / _with_suffix(mc.weights_file or f"{args.model}_best.pth")

    mlflow.set_tracking_uri(f"file://{ROOT / 'mlruns'}")
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=args.model):
        mlflow.log_params({
            "model": args.model,
            "backend": mc.raw.get("backend", "torchvision"),
            "pretrained": mc.pretrained,
            "image_size": mc.image_size,
            "dropout": mc.dropout,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "val_fraction": args.val_fraction,
            "seed": args.seed,
            "device": device.type,
            "num_classes": cfg.num_classes,
            "train_samples": len(train_subset),
            "limit_per_class": args.limit_per_class or "none",
        })

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
        best_val_acc = -1.0

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
            lr_now = scheduler.get_last_lr()[0]
            scheduler.step()

            for key, value in [("train_loss", train_loss), ("train_acc", train_acc),
                               ("val_loss", val_loss), ("val_acc", val_acc), ("lr", lr_now)]:
                history[key].append(value)
                mlflow.log_metric(key, value, step=epoch)
            mlflow.log_metric("epoch_seconds", time.time() - t0, step=epoch)

            marker = ""
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), weights_path)
                marker = "  (saved)"
            print(f"[epoch {epoch:03d}/{args.epochs}] "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                  f"lr={lr_now:.2e} ({time.time() - t0:.0f}s){marker}")

        # Final evaluation on the held-out test split, with the best checkpoint
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        y_true, y_pred, y_proba = collect_predictions(model, test_loader, device)
        summary = compute_summary(y_true, y_pred, y_proba, cfg.num_classes)
        summary["accuracy"] = summary["top1"]  # alias used by the Streamlit app
        mlflow.log_metrics({f"test_{k}": v for k, v in summary.items()
                            if not np.isnan(v)})
        print("[test] " + "  ".join(f"{k}={v:.4f}" for k, v in summary.items()))

        # Persist the artifacts the Streamlit app already consumes
        history_path = RESULTS_DIR / _with_suffix(mc.history_file or f"{args.model}_history.pkl")
        metrics_path = RESULTS_DIR / _with_suffix(mc.metrics_file or f"{args.model}_metrics.pkl")
        preds_path = RESULTS_DIR / _with_suffix(mc.preds_file or f"{args.model}_preds.npz")
        with history_path.open("wb") as f:
            pickle.dump(history, f)
        with metrics_path.open("wb") as f:
            pickle.dump(summary, f)
        np.savez(preds_path, y_true=y_true, y_pred=y_pred, y_proba=y_proba)

        mlflow.log_artifact(str(weights_path), artifact_path="model")
        mlflow.log_artifact(str(history_path), artifact_path="results")
        mlflow.log_artifact(str(metrics_path), artifact_path="results")
        mlflow.log_artifact(str(preds_path), artifact_path="results")
        if args.log_pytorch_model:
            mlflow.pytorch.log_model(model, name="pytorch_model")

        print(f"[done] best_val_acc={best_val_acc:.4f}  weights={weights_path}")
        print(f"[done] MLflow run: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
