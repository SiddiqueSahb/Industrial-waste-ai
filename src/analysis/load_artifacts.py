"""Load saved artifacts produced by the Colab training pipeline."""
from __future__ import annotations
from pathlib import Path
import pickle
from typing import Optional, Dict, Any
import numpy as np

ARTIFACT_DIR = Path("results/metrics")


def _path(name: str, suffix: str) -> Path:
    return ARTIFACT_DIR / f"{name}_{suffix}"


def artifacts_exist(model_name: str) -> bool:
    """A model is considered loaded if metrics.pkl exists."""
    return _path(model_name, "metrics.pkl").exists()


def load_metrics(model_name: str) -> Optional[Dict[str, Any]]:
    p = _path(model_name, "metrics.pkl")
    if not p.exists():
        return None
    with p.open("rb") as f:
        return pickle.load(f)


def load_history(model_name: str) -> Optional[Dict[str, Any]]:
    p = _path(model_name, "history.pkl")
    if not p.exists():
        return None
    with p.open("rb") as f:
        return pickle.load(f)


def load_predictions(model_name: str) -> Optional[Dict[str, np.ndarray]]:
    """Returns dict with keys y_true, y_pred, y_prob (whichever exist)."""
    p = _path(model_name, "preds.npz")
    if not p.exists():
        return None
    data = np.load(p)
    out: Dict[str, np.ndarray] = {}
    for k in data.files:
        out[k] = data[k]
    # alias y_prob <-> y_proba so downstream code can use either
    if "y_prob" in out and "y_proba" not in out:
        out["y_proba"] = out["y_prob"]
    if "y_proba" in out and "y_prob" not in out:
        out["y_prob"] = out["y_proba"]
    return out
