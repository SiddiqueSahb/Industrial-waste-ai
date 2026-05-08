"""Cross-model comparison utilities — load multiple model artifacts at once."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

from . import metrics as M


def model_dir(results_root: Path | str, name: str) -> Path:
    return Path(results_root) / name


def load_model_bundle(results_root: Path | str, name: str) -> Dict:
    """Load all four artifacts for one model."""
    d = model_dir(results_root, name)
    bundle = {
        "name":    name,
        "history": M.load_history(d / f"{name}_history.pkl"),
        "metrics": M.load_metrics(d / f"{name}_metrics.pkl"),
    }
    y_true, y_pred, y_proba = M.load_predictions(d / f"{name}_preds.npz")
    bundle.update({"y_true": y_true, "y_pred": y_pred, "y_proba": y_proba})
    return bundle


def build_summary_table(results_root: Path | str,
                        model_names: List[str],
                        class_names: List[str]) -> pd.DataFrame:
    """One row per model with top-line metrics."""
    rows = []
    for name in model_names:
        b = load_model_bundle(results_root, name)
        s = M.compute_summary(b["y_true"], b["y_pred"], b["y_proba"],
                              num_classes=len(class_names))
        s["model"] = name
        rows.append(s)
    df = pd.DataFrame(rows)
    cols = ["model", "top1", "top5", "macro_precision", "macro_recall",
            "macro_f1", "weighted_f1", "macro_auc_ovr",
            "map_macro", "map_weighted"]
    return df[cols] 