"""
WaRP-C Industrial Waste Classification — Streamlit App
Run from the project root:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

# ---------------------------------------------------------------------------
# Path bootstrap — must happen before any src.* import
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.comparison_table import build_comparison
from src.analysis.load_artifacts import artifacts_exist, load_history, load_metrics, load_predictions
from src.analysis.misclassifications import confusions_for_true_class, top_confused_pairs, worst_classes
from src.classification.augmentations import get_eval_transform
from src.classification.inference import load_trained_model, predict_image
from src.config.loader import load_config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(name)s  %(message)s")
log = logging.getLogger("warpc")

# ---------------------------------------------------------------------------
# Application constants
# ---------------------------------------------------------------------------
CONFIG_PATH = ROOT / "configs" / "classification.yaml"
MODELS_DIR  = ROOT / "models" / "classification"
RESULTS_DIR = ROOT / "results" / "metrics"

PAGES = ["Live prediction", "Per-model report", "Cross-model comparison", "Dataset"]

# ---------------------------------------------------------------------------
# App-wide config — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="WaRP-C Waste Classifier",
    page_icon="♻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Load global config once
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_app_config():
    return load_config(CONFIG_PATH)

cfg = _load_app_config()
ALL_MODELS  = list(cfg.models.keys())
AVAILABLE   = [n for n in ALL_MODELS if artifacts_exist(n)]
CLASS_NAMES = cfg.classes
NUM_CLASSES = cfg.num_classes

# ---------------------------------------------------------------------------
# Cached resource / data loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model weights…")
def _get_model(name: str):
    mc      = cfg.models[name]
    weights = MODELS_DIR / (mc.weights_file or f"{name}_best.pth")
    log.info("Loading model %s from %s", name, weights)
    return load_trained_model(
        model_name=name,
        weight_path=weights,
        num_classes=NUM_CLASSES,
        dropout=mc.dropout,
    )

@st.cache_data(show_spinner=False)
def _get_metrics(name: str):
    return load_metrics(name)

@st.cache_data(show_spinner=False)
def _get_history(name: str):
    return load_history(name)

@st.cache_data(show_spinner=False)
def _get_preds(name: str):
    return load_predictions(name)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def _render_sidebar() -> str:
    with st.sidebar:
        st.title("WaRP-C Demo")
        st.caption(cfg.course)
        st.divider()

        page = st.radio("Section", PAGES, label_visibility="collapsed")

        st.divider()
        st.caption(f"Models loaded: **{len(AVAILABLE)} / {len(ALL_MODELS)}**")
        for name in ALL_MODELS:
            icon = "✔" if name in AVAILABLE else "·"
            st.write(f"{icon}  `{name}`")

    return page

# ---------------------------------------------------------------------------
# Metric helper — avoids repetitive None-guarding inline
# ---------------------------------------------------------------------------
def _safe_metric(value: Optional[float], fmt: str, fallback: str = "—") -> str:
    if value is None:
        return fallback
    try:
        return fmt.format(value)
    except (ValueError, TypeError):
        return fallback

# ---------------------------------------------------------------------------
# Page: Live prediction
# ---------------------------------------------------------------------------
def page_live_prediction() -> None:
    st.header("Live prediction")
    st.write("Upload a waste image and run it through every loaded model.")

    if not AVAILABLE:
        st.warning("No trained models found — train at least one model first.")
        return

    col_upload, col_preview = st.columns([1, 1.3])

    with col_upload:
        uploaded_file = st.file_uploader("Image (JPG / PNG)", type=["jpg", "jpeg", "png"])
        chosen_models = st.multiselect("Models to run", AVAILABLE, default=AVAILABLE)
        top_k         = st.slider("Top-k predictions", min_value=1, max_value=10, value=5)
        run_clicked   = st.button("Predict", type="primary", use_container_width=True)

    with col_preview:
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption=uploaded_file.name, use_container_width=True)

    if not run_clicked:
        return
    if uploaded_file is None:
        st.error("Please upload an image first.")
        return
    if not chosen_models:
        st.error("Select at least one model.")
        return

    st.divider()
    st.subheader("Results")

    result_cols = st.columns(len(chosen_models))
    for col, model_name in zip(result_cols, chosen_models):
        with col:
            _run_single_prediction(model_name, image, top_k)


def _run_single_prediction(model_name: str, image: Image.Image, top_k: int) -> None:
    st.markdown(f"**{model_name}**")
    try:
        model = _get_model(model_name)
        mc    = cfg.models[model_name]
        tfm   = get_eval_transform(
            img_size=mc.image_size,
            resize=cfg.augmentation.resize_for_eval,
            mean=cfg.augmentation.mean,
            std=cfg.augmentation.std,
        )

        t0       = time.perf_counter()
        probs    = predict_image(model, image, tfm)
        latency  = (time.perf_counter() - t0) * 1000

        top_idx = probs.argsort()[::-1][:top_k]
        df = pd.DataFrame({
            "class": [CLASS_NAMES[j] for j in top_idx],
            "prob":  [float(probs[j]) for j in top_idx],
        })

        top1 = df.iloc[0]
        st.metric("Top-1", top1["class"], f"{top1['prob'] * 100:.1f}%")
        st.bar_chart(df.set_index("class")["prob"], height=220)
        st.caption(f"Latency: {latency:.1f} ms")

    except Exception:
        log.exception("Prediction failed for model %s", model_name)
        st.error(f"Failed to run {model_name}. Check logs for details.")

# ---------------------------------------------------------------------------
# Page: Per-model report
# ---------------------------------------------------------------------------
def page_per_model_report() -> None:
    st.header("Per-model report")

    if not AVAILABLE:
        st.warning("No trained models found in `results/metrics/`.")
        return

    model_name = st.selectbox("Model", AVAILABLE)
    metrics    = _get_metrics(model_name) or {}
    history    = _get_history(model_name)
    preds      = _get_preds(model_name)

    _render_summary_metrics(metrics)
    st.divider()

    if history is not None:
        _render_training_curves(history)

    cm = _resolve_confusion_matrix(metrics, preds)
    if cm is not None:
        _render_confusion_matrix(cm)

    if preds is not None:
        _render_per_class_accuracy(preds)
        st.divider()
        _render_misclassification_analysis(preds)


def _get_metric(metrics: dict, *keys: str) -> Optional[float]:
    for k in keys:
        v = metrics.get(k)
        if v is not None:
            return v
    return None


def _render_summary_metrics(metrics: dict) -> None:
    acc  = _get_metric(metrics, "accuracy", "acc")
    f1   = _get_metric(metrics, "f1", "macro_f1")
    auc  = _get_metric(metrics, "auc", "macro_auc")
    prec = _get_metric(metrics, "precision", "macro_precision")
    rec  = _get_metric(metrics, "recall", "macro_recall")
    par  = _get_metric(metrics, "params_total", "params")
    inf  = _get_metric(metrics, "inference_time_per_img_s", "infer_s")

    cols = st.columns(6)
    cols[0].metric("Accuracy",     _safe_metric(acc,  "{:.2%}"))
    cols[1].metric("Macro F1",     _safe_metric(f1,   "{:.4f}"))
    cols[2].metric("Macro AUC",    _safe_metric(auc,  "{:.4f}"))
    cols[3].metric("Precision",    _safe_metric(prec, "{:.4f}"))
    cols[4].metric("Params",       _safe_metric(par,  "{:.1f}M", "—") if par is None else f"{par / 1e6:.1f}M")
    cols[5].metric("Infer ms/img", _safe_metric(inf,  "{:.2f}") if inf is None else f"{inf * 1000:.2f}")


def _render_training_curves(history: dict) -> None:
    st.subheader("Training curves")
    col_loss, col_acc = st.columns(2)

    with col_loss:
        st.caption("Loss")
        st.line_chart(pd.DataFrame({
            "train": history.get("train_loss", []),
            "val":   history.get("val_loss", []),
        }))

    with col_acc:
        st.caption("Accuracy")
        st.line_chart(pd.DataFrame({
            "train": history.get("train_acc", []),
            "val":   history.get("val_acc", []),
        }))


def _resolve_confusion_matrix(metrics: dict, preds: Optional[dict]) -> Optional[np.ndarray]:
    cm = metrics.get("confusion_matrix")
    if cm is not None:
        return np.asarray(cm, dtype=float)

    if preds is not None and "y_true" in preds and "y_pred" in preds:
        return sk_confusion_matrix(
            preds["y_true"], preds["y_pred"],
            labels=list(range(NUM_CLASSES)),
        ).astype(float)

    return None


def _render_confusion_matrix(cm: np.ndarray) -> None:
    st.subheader("Confusion matrix (row-normalised)")
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    df_cm   = pd.DataFrame(
        cm_norm,
        index=CLASS_NAMES[:cm.shape[0]],
        columns=CLASS_NAMES[:cm.shape[1]],
    )
    st.dataframe(
        df_cm.style.format("{:.2f}").background_gradient(cmap="Blues"),
        use_container_width=True,
        height=620,
    )


def _render_per_class_accuracy(preds: dict) -> None:
    st.subheader("Per-class accuracy")
    y_true, y_pred = preds["y_true"], preds["y_pred"]

    rows = [
        {
            "class":    cn,
            "n":        int((y_true == i).sum()),
            "accuracy": float((y_pred[y_true == i] == i).mean()) if (y_true == i).any() else float("nan"),
        }
        for i, cn in enumerate(CLASS_NAMES)
    ]
    df_pc = pd.DataFrame(rows).sort_values("accuracy")
    st.bar_chart(df_pc.set_index("class")["accuracy"], height=320)

    with st.expander("Per-class table"):
        st.dataframe(df_pc, use_container_width=True)

    # Store for use in misclassification drill-down
    st.session_state["_df_pc"] = df_pc


def _render_misclassification_analysis(preds: dict) -> None:
    st.subheader("Misclassification analysis")
    st.caption(
        "Which class pairs the model can't tell apart — "
        "use this to guide oversampling, augmentation, or architecture changes."
    )

    y_true, y_pred = preds["y_true"], preds["y_pred"]
    tab_pairs, tab_worst, tab_drill = st.tabs([
        "Top confused pairs", "Worst classes", "Drill into one class",
    ])

    with tab_pairs:
        n_pairs  = st.slider("Show top N pairs", 5, 50, 20, key="n_pairs")
        df_pairs = top_confused_pairs(y_true, y_pred, CLASS_NAMES, top_n=n_pairs)

        if df_pairs.empty:
            st.success("No misclassifications found.")
        else:
            st.dataframe(
                df_pairs.style
                    .format({"Errors": "{:d}", "% of true class": "{:.1f}%"})
                    .background_gradient(subset=["Errors"], cmap="Reds"),
                use_container_width=True,
                height=min(540, 36 + 36 * len(df_pairs)),
            )
            error_rate = (y_true != y_pred).sum() / len(y_true) * 100
            st.caption(
                f"Total errors: {int(df_pairs['Errors'].sum())} "
                f"(of {len(y_true)} test images, {error_rate:.2f}% error rate)"
            )

    with tab_worst:
        df_worst = worst_classes(y_true, y_pred, CLASS_NAMES)
        st.dataframe(
            df_worst.style
                .format({"accuracy": "{:.2%}"})
                .background_gradient(subset=["accuracy"], cmap="RdYlGn"),
            use_container_width=True,
            height=min(620, 36 + 36 * len(df_worst)),
        )
        st.caption("`most confused with` shows the most-frequent wrong label for each true class.")

    with tab_drill:
        df_pc       = st.session_state.get("_df_pc", pd.DataFrame())
        default_idx = int(df_pc.iloc[0].name) if not df_pc.empty else 0
        target      = st.selectbox("True class", CLASS_NAMES, index=default_idx, key="drill_class")
        target_idx  = CLASS_NAMES.index(target)
        df_one      = confusions_for_true_class(y_true, y_pred, CLASS_NAMES, target_idx)

        if df_one.empty:
            st.warning("No test samples for this class.")
        else:
            st.bar_chart(df_one.set_index("Predicted as")["%"], height=320)
            st.dataframe(
                df_one.drop(columns=["Correct"]).style.format({"%": "{:.1f}%"}),
                use_container_width=True,
            )
            n_total = int(df_one["Count"].sum())
            n_wrong = int(df_one[~df_one["Correct"]]["Count"].sum())
            st.caption(
                f"Of {n_total} test images of `{target}`, "
                f"{n_total - n_wrong} correct, {n_wrong} misclassified."
            )

# ---------------------------------------------------------------------------
# Page: Cross-model comparison
# ---------------------------------------------------------------------------
def page_cross_model_comparison() -> None:
    st.header("Cross-model comparison")

    if not AVAILABLE:
        st.warning("No models available yet.")
        return

    df = build_comparison(AVAILABLE)
    st.dataframe(df, use_container_width=True)

    if "Params (M)" in df.columns and df["Params (M)"].notna().any():
        st.divider()
        st.subheader("Accuracy vs parameters")
        st.scatter_chart(
            df.dropna(subset=["Params (M)", "Accuracy"]),
            x="Params (M)",
            y="Accuracy",
            color="Model",
            height=380,
        )

# ---------------------------------------------------------------------------
# Page: Dataset
# ---------------------------------------------------------------------------
def page_dataset() -> None:
    st.header("Dataset — WaRP-C")
    st.write(
        f"{NUM_CLASSES} fine-grained recyclable-waste classes. "
        "~8.8k training images, ~1.5k test images. "
        "Class distribution is highly imbalanced."
    )
    st.divider()
    st.subheader("Classes")
    st.write(", ".join(f"`{c}`" for c in CLASS_NAMES))

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
_PAGE_MAP = {
    "Live prediction":        page_live_prediction,
    "Per-model report":       page_per_model_report,
    "Cross-model comparison": page_cross_model_comparison,
    "Dataset":                page_dataset,
}


def main() -> None:
    page = _render_sidebar()
    handler = _PAGE_MAP.get(page)
    if handler is None:
        st.error(f"Unknown page: {page!r}")
        return
    handler()


if __name__ == "__main__":
    main()
