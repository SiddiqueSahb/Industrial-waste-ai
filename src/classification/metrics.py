"""Metric computation helpers — pure functions, no I/O.

Used by scripts/analyze_model.py and scripts/compare_models.py.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
    top_k_accuracy_score,
    classification_report,
    confusion_matrix,
)


def load_predictions(npz_path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (y_true, y_pred, y_proba). y_proba shape (N, num_classes)."""
    data = np.load(str(npz_path))
    y_true  = data["y_true"]
    y_pred  = data["y_pred"]
    y_proba = data["y_proba"]
    return y_true, y_pred, y_proba


def load_metrics(pkl_path: str | Path) -> dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def load_history(pkl_path: str | Path) -> dict:
    """Expected keys: train_loss, train_acc, val_loss, val_acc, ema_val_acc, lr."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def compute_summary(y_true: np.ndarray, y_pred: np.ndarray,
                    y_proba: np.ndarray, num_classes: int) -> Dict[str, float]:
    """Top-line metrics for a single model."""
    top1 = accuracy_score(y_true, y_pred)
    top5 = top_k_accuracy_score(y_true, y_proba, k=5,
                                labels=np.arange(num_classes))

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    try:
        auc_macro = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro",
            labels=np.arange(num_classes),
        )
    except ValueError:
        auc_macro = float("nan")


    y_true_onehot = np.eye(num_classes)[y_true]
    try:
        map_macro = average_precision_score(
            y_true_onehot, y_proba, average="macro"
        )
        map_weighted = average_precision_score(
            y_true_onehot, y_proba, average="weighted"
        )
    except ValueError:
        map_macro = float("nan")
        map_weighted = float("nan")

    return {
        "top1": float(top1),
        "top5": float(top5),
        "macro_precision": float(p_macro),
        "macro_recall":    float(r_macro),
        "macro_f1":        float(f1_macro),
        "weighted_precision": float(p_weighted),
        "weighted_recall":    float(r_weighted),
        "weighted_f1":        float(f1_weighted),
        "macro_auc_ovr":   float(auc_macro),
        "map_macro":       float(map_macro),
        "map_weighted":    float(map_weighted),
    }


def compute_per_class_ap(y_true: np.ndarray, y_proba: np.ndarray,
                         class_names: List[str]) -> pd.DataFrame:
    """Per-class Average Precision (area under the PR curve, one-vs-rest).

    Returns DataFrame with columns: class, ap, support.
    """
    num_classes = len(class_names)
    y_true_onehot = np.eye(num_classes)[y_true]
    aps = []
    supports = []
    for c in range(num_classes):
        support = int((y_true == c).sum())
        supports.append(support)
        if support == 0:
            aps.append(float("nan"))
            continue
        try:
            ap = average_precision_score(y_true_onehot[:, c], y_proba[:, c])
        except ValueError:
            ap = float("nan")
        aps.append(float(ap))
    return pd.DataFrame({
        "class":   class_names,
        "ap":      aps,
        "support": supports,
    })


def compute_per_class(y_true: np.ndarray, y_pred: np.ndarray,
                      class_names: List[str]) -> pd.DataFrame:
    """One row per class with precision/recall/f1/support."""
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred,
        labels=np.arange(len(class_names)),
        zero_division=0,
    )
    return pd.DataFrame({
        "class":     class_names,
        "precision": p,
        "recall":    r,
        "f1":        f1,
        "support":   support,
    })


def text_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                               class_names: List[str]) -> str:
    return classification_report(
        y_true, y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        digits=4, zero_division=0,
    )


def confusion(y_true: np.ndarray, y_pred: np.ndarray,
              num_classes: int, normalize: str | None = "true") -> np.ndarray:
    """`normalize`: 'true', 'pred', 'all', or None."""
    return confusion_matrix(
        y_true, y_pred,
        labels=np.arange(num_classes),
        normalize=normalize,
    )