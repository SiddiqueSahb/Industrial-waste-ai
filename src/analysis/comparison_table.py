"""Schema-tolerant cross-model comparison table."""
from __future__ import annotations
from typing import Iterable
import numpy as np
import pandas as pd
from .load_artifacts import load_metrics, artifacts_exist


def _g(d: dict, *keys, default=float("nan")):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def build_comparison(model_names: Iterable[str]) -> pd.DataFrame:
    rows = []
    for n in model_names:
        if not artifacts_exist(n):
            continue
        m = load_metrics(n)
        if not isinstance(m, dict):
            print(f"[skip] {n}: metrics is {type(m).__name__}")
            continue
        acc  = _g(m, "accuracy", "acc", "test_acc")
        prec = _g(m, "precision", "macro_precision", "prec")
        rec  = _g(m, "recall", "macro_recall", "rec")
        f1   = _g(m, "f1", "macro_f1", "f1_macro")
        auc  = _g(m, "auc", "macro_auc", "roc_auc")
        par  = _g(m, "params_total", "params", "n_params")
        inf  = _g(m, "inference_time_per_img_s", "infer_s", "latency_s")

        rows.append({
            "Model": n,
            "Accuracy":     float(acc) if acc == acc else np.nan,
            "Macro F1":     float(f1)  if f1  == f1  else np.nan,
            "Macro AUC":    float(auc) if auc == auc else np.nan,
            "Precision":    float(prec) if prec == prec else np.nan,
            "Recall":       float(rec)  if rec  == rec  else np.nan,
            "Params (M)":   (float(par) / 1e6) if par == par else np.nan,
            "Infer ms/img": (float(inf) * 1000) if inf == inf else np.nan,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("Accuracy", ascending=False).reset_index(drop=True)
