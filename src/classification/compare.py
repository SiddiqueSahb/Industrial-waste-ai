<<<<<<< HEAD
<<<<<<< HEAD
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


def build_per_class_table(results_root: Path | str,
                          model_names: List[str],
                          class_names: List[str]) -> pd.DataFrame:
    """Long-format DataFrame: (class, model) -> precision/recall/f1/support/ap."""
    frames = []
    for name in model_names:
        b = load_model_bundle(results_root, name)
        df = M.compute_per_class(b["y_true"], b["y_pred"], class_names)
        ap_df = M.compute_per_class_ap(b["y_true"], b["y_proba"], class_names)
        df["ap"] = ap_df["ap"].values
        df["model"] = name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_per_class_ap_pivot(results_root: Path | str,
                             model_names: List[str],
                             class_names: List[str]) -> pd.DataFrame:
    """Wide table: rows = classes, columns = models, values = AP.
    The last row is the macro-mAP per model."""
    cols = {}
    for name in model_names:
        b = load_model_bundle(results_root, name)
        ap_df = M.compute_per_class_ap(b["y_true"], b["y_proba"], class_names)
        cols[name] = ap_df["ap"].values
    df = pd.DataFrame(cols, index=class_names)
    df.index.name = "class"
    df.loc["__mAP_macro__"] = df.mean(axis=0, skipna=True)
    return df


def discover_trained_models(results_root: Path | str) -> List[str]:
    """Returns list of subfolders in results/ that contain a .pth file."""
    root = Path(results_root)
    if not root.exists():
        return []
    out = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and any(d.glob("*.pth")):
            out.append(d.name)
    return out
=======
=======
>>>>>>> 87bc0c9b (Solved conflict)
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
<<<<<<< HEAD
>>>>>>> 0785ee38 (Added load model and build summary function)
=======
=======
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


def build_per_class_table(results_root: Path | str,
                          model_names: List[str],
                          class_names: List[str]) -> pd.DataFrame:
    """Long-format DataFrame: (class, model) -> precision/recall/f1/support/ap."""
    frames = []
    for name in model_names:
        b = load_model_bundle(results_root, name)
        df = M.compute_per_class(b["y_true"], b["y_pred"], class_names)
        ap_df = M.compute_per_class_ap(b["y_true"], b["y_proba"], class_names)
        df["ap"] = ap_df["ap"].values
        df["model"] = name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_per_class_ap_pivot(results_root: Path | str,
                             model_names: List[str],
                             class_names: List[str]) -> pd.DataFrame:
    """Wide table: rows = classes, columns = models, values = AP.
    The last row is the macro-mAP per model."""
    cols = {}
    for name in model_names:
        b = load_model_bundle(results_root, name)
        ap_df = M.compute_per_class_ap(b["y_true"], b["y_proba"], class_names)
        cols[name] = ap_df["ap"].values
    df = pd.DataFrame(cols, index=class_names)
    df.index.name = "class"
    df.loc["__mAP_macro__"] = df.mean(axis=0, skipna=True)
    return df


def discover_trained_models(results_root: Path | str) -> List[str]:
    """Returns list of subfolders in results/ that contain a .pth file."""
    root = Path(results_root)
    if not root.exists():
        return []
    out = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and any(d.glob("*.pth")):
            out.append(d.name)
    return out
>>>>>>> fc902cb7 (Added compare.py to compare  models)
>>>>>>> 87bc0c9b (Solved conflict)
