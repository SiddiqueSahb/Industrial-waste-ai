
from __future__ import annotations
from typing import Dict, List, Sequence
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sk_auc
from sklearn.preprocessing import label_binarize


# Training curves
def plot_training_curves(history: Dict[str, list], title: str = "") -> plt.Figure:
    """Plots train/val loss and accuracy side-by-side. Adds EMA val if present."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    #plotting training loss curves
    axes[0].plot(epochs, history["train_loss"], label="train", marker="o", ms=3)
    #plotting validation loss curves
    axes[0].plot(epochs, history["val_loss"],   label="val",   marker="s", ms=3)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].set_title(f"{title} — loss".strip(" —"))
    axes[0].grid(True, alpha=0.3); axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train", marker="o", ms=3)
    axes[1].plot(epochs, history["val_acc"],   label="val",   marker="s", ms=3)
    if "ema_val_acc" in history:
        axes[1].plot(epochs, history["ema_val_acc"],
                     label="EMA val", marker="^", ms=3, linestyle="--")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy")
    axes[1].set_title(f"{title} — accuracy".strip(" —"))
    axes[1].grid(True, alpha=0.3); axes[1].legend()

    fig.tight_layout()
    return fig
