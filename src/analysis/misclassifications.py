"""Misclassification analysis — identify which class pairs the model confuses."""
from __future__ import annotations
from typing import List, Sequence
import numpy as np
import pandas as pd


def top_confused_pairs(y_true: np.ndarray, y_pred: np.ndarray,
                       class_names: Sequence[str], top_n: int = 20) -> pd.DataFrame:
    """Return the top-N most frequent (true → predicted) error pairs.

    Columns: True class, Predicted as, Errors, % of true class
    """
    mask = y_true != y_pred
    if mask.sum() == 0:
        return pd.DataFrame(columns=["True class", "Predicted as", "Errors", "% of true class"])

    yt = y_true[mask]
    yp = y_pred[mask]

    pairs = pd.DataFrame({"true": yt, "pred": yp})
    counts = pairs.value_counts().reset_index(name="Errors")

    # how many samples of each true class exist (for the % column)
    per_class_n = pd.Series(y_true).value_counts()
    counts["% of true class"] = (
        counts["Errors"] / counts["true"].map(per_class_n) * 100
    )
    counts["True class"]   = counts["true"].map(lambda i: class_names[i])
    counts["Predicted as"] = counts["pred"].map(lambda i: class_names[i])

    out = counts[["True class", "Predicted as", "Errors", "% of true class"]]
    return out.head(top_n).reset_index(drop=True)


def confusions_for_true_class(y_true: np.ndarray, y_pred: np.ndarray,
                              class_names: Sequence[str], target_idx: int) -> pd.DataFrame:
    """For one true class, return a row of (predicted_as, count, %) — including the diagonal.

    Sorted descending. Diagonal (correct) row stays so you can see how dominant it is.
    """
    mask = y_true == target_idx
    n = int(mask.sum())
    if n == 0:
        return pd.DataFrame(columns=["Predicted as", "Count", "%", "Correct"])

    yp = y_pred[mask]
    counts = pd.Series(yp).value_counts()
    df = pd.DataFrame({
        "Predicted as": [class_names[i] for i in counts.index],
        "Count":        counts.values,
        "%":            (counts.values / n * 100),
        "Correct":      [i == target_idx for i in counts.index],
    })
    return df.reset_index(drop=True)


def worst_classes(y_true: np.ndarray, y_pred: np.ndarray,
                  class_names: Sequence[str]) -> pd.DataFrame:
    """Per-class error stats sorted worst first."""
    rows = []
    for i, cn in enumerate(class_names):
        mask = y_true == i
        n = int(mask.sum())
        if n == 0:
            continue
        correct = int((y_pred[mask] == i).mean() * n)
        acc = correct / n
        # most-frequent wrong prediction (if any)
        wrong = y_pred[mask & (y_pred != i)]
        if len(wrong) > 0:
            top_wrong_idx = int(pd.Series(wrong).value_counts().idxmax())
            top_wrong = class_names[top_wrong_idx]
            top_wrong_n = int((wrong == top_wrong_idx).sum())
        else:
            top_wrong, top_wrong_n = "—", 0
        rows.append({
            "class": cn, "n": n, "correct": correct, "errors": n - correct,
            "accuracy": acc,
            "most confused with": top_wrong,
            "× confused": top_wrong_n,
        })
    return pd.DataFrame(rows).sort_values("accuracy").reset_index(drop=True)
