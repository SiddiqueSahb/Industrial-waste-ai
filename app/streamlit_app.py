"""
WaRP-C Industrial Waste Classification — Streamlit Demo

Run from the project root:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

# Make `src.*` importable when running `streamlit run app/streamlit_app.py`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config.loader import load_config
from src.classification.augmentations import get_eval_transform
from src.classification.inference import load_trained_model, predict_image
from src.analysis.load_artifacts import (
    load_history, load_metrics, load_predictions, artifacts_exist,
)
from src.analysis.comparison_table import build_comparison
from src.analysis.misclassifications import (
    top_confused_pairs, confusions_for_true_class, worst_classes,
)

# Page setup
st.set_page_config(page_title="WaRP-C Waste Classifier", page_icon="♻", layout="wide")

CONFIG_PATH = ROOT / "configs" / "classification.yaml"
MODELS_DIR  = ROOT / "models" / "classification"
RESULTS_DIR = ROOT / "results" / "metrics"

cfg = load_config(CONFIG_PATH)
ALL_MODELS  = list(cfg.models.keys())
AVAILABLE   = [n for n in ALL_MODELS if artifacts_exist(n)]
CLASS_NAMES = cfg.classes
NUM_CLASSES = cfg.num_classes

# Sidebar
with st.sidebar:
    st.title("WaRP-C Demo")
    st.caption(cfg.course)
    st.divider()
    page = st.radio(
        "Section",
        ["Live prediction", "Per-model report", "Cross-model comparison", "Dataset"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(f"Models loaded: **{len(AVAILABLE)} / {len(ALL_MODELS)}**")
    for n in ALL_MODELS:
        ok = "✔" if n in AVAILABLE else "·"
        st.write(f"{ok}  `{n}`")

# Cached loaders
@st.cache_resource(show_spinner="Loading model…")
def _get_model(name: str):
    mc = cfg.models[name]
    weights = MODELS_DIR / (mc.weights_file or f"{name}_best.pth")
    return load_trained_model(
        model_name=name,
        weights_path=weights,
        num_classes=NUM_CLASSES,
        dropout=mc.dropout,
    )

@st.cache_data
def _get_metrics(name: str):  return load_metrics(name)
@st.cache_data
def _get_history(name: str):  return load_history(name)
@st.cache_data
def _get_preds(name: str):    return load_predictions(name)

# 1. LIVE PREDICTION
if page == "Live prediction":
    st.header("Live prediction")
    st.write("Upload a waste image and run it through every loaded model.")

    col_l, col_r = st.columns([1, 1.3])
    with col_l:
        upl = st.file_uploader("Image (JPG / PNG)", type=["jpg", "jpeg", "png"])
        chosen = st.multiselect("Models to run", AVAILABLE, default=AVAILABLE)
        topk = st.slider("Top-k", 1, 10, 5)
        run = st.button("Predict", type="primary", use_container_width=True)
    with col_r:
        if upl is not None:
            img = Image.open(upl).convert("RGB")
            st.image(img, caption=upl.name, use_container_width=True)

    if run and upl is not None and chosen:
        st.divider()
        st.subheader("Results")
        cols = st.columns(len(chosen))
        for i, name in enumerate(chosen):
            with cols[i]:
                st.markdown(f"**{name}**")
                model = _get_model(name)
                tfm = get_eval_transform(
                    img_size=cfg.models[name].image_size,
                    resize=cfg.augmentation.resize_for_eval,
                    mean=cfg.augmentation.mean, std=cfg.augmentation.std,
                )
                t0 = time.perf_counter()
                probs = predict_image(model, img, tfm)
                latency_ms = (time.perf_counter() - t0) * 1000

                top_idx = probs.argsort()[::-1][:topk]
                df = pd.DataFrame({
                    "class": [CLASS_NAMES[j] for j in top_idx],
                    "prob":  [float(probs[j]) for j in top_idx],
                })
                st.metric("Top-1", df.iloc[0]["class"], f"{df.iloc[0]['prob']*100:.1f}%")
                st.bar_chart(df.set_index("class")["prob"], height=220)
                st.caption(f"latency: {latency_ms:.1f} ms")

# 2. PER-MODEL REPORT
elif page == "Per-model report":
    st.header("Per-model report")
    if not AVAILABLE:
        st.warning("No trained models found in `results/metrics/`.")
        st.stop()

    name = st.selectbox("Model", AVAILABLE)
    metrics = _get_metrics(name) or {}
    hist    = _get_history(name)
    preds   = _get_preds(name)

    def _g(*keys):
        for k in keys:
            if k in metrics and metrics[k] is not None:
                return metrics[k]
        return None

    acc  = _g("accuracy", "acc")
    f1   = _g("f1", "macro_f1")
    auc  = _g("auc", "macro_auc")
    prec = _g("precision", "macro_precision")
    rec  = _g("recall", "macro_recall")
    par  = _g("params_total", "params")
    inf  = _g("inference_time_per_img_s", "infer_s")

    c = st.columns(6)
    c[0].metric("Accuracy",     f"{acc*100:.2f}%" if acc is not None else "—")
    c[1].metric("Macro F1",     f"{f1:.4f}"      if f1  is not None else "—")
    c[2].metric("Macro AUC",    f"{float(auc):.4f}" if auc is not None else "—")
    c[3].metric("Precision",    f"{prec:.4f}"    if prec is not None else "—")
    c[4].metric("Params",       f"{par/1e6:.1f}M" if par is not None else "—")
    c[5].metric("Infer ms/img", f"{inf*1000:.2f}" if inf is not None else "—")

    st.divider()

    # training curves 
    if hist is not None:
        st.subheader("Training curves")
        cc1, cc2 = st.columns(2)
        with cc1:
            st.caption("Loss")
            st.line_chart(pd.DataFrame({
                "train": hist.get("train_loss", []),
                "val":   hist.get("val_loss", []),
            }))
        with cc2:
            st.caption("Accuracy")
            st.line_chart(pd.DataFrame({
                "train": hist.get("train_acc", []),
                "val":   hist.get("val_acc", []),
            }))

    # confusion matrix (computed from preds if not in metrics)
    cm = metrics.get("confusion_matrix")
    if cm is None and preds is not None and "y_true" in preds and "y_pred" in preds:
        cm = sk_confusion_matrix(
            preds["y_true"], preds["y_pred"],
            labels=list(range(NUM_CLASSES)),
        )
    if cm is not None:
        st.subheader("Confusion matrix (row-normalised)")
        cm = np.asarray(cm, dtype=float)
        cm_n = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        df_cm = pd.DataFrame(cm_n, index=CLASS_NAMES[:cm.shape[0]],
                             columns=CLASS_NAMES[:cm.shape[1]])
        st.dataframe(
            df_cm.style.format("{:.2f}").background_gradient(cmap="Blues"),
            use_container_width=True, height=620,
        )

    # per-class accuracy 
    if preds is not None:
        st.subheader("Per-class accuracy")
        y_true, y_pred = preds["y_true"], preds["y_pred"]
        rows = []
        for i, cn in enumerate(CLASS_NAMES):
            mask = y_true == i
            n = int(mask.sum())
            a = float((y_pred[mask] == i).mean()) if n else float("nan")
            rows.append({"class": cn, "n": n, "accuracy": a})
        df_pc = pd.DataFrame(rows).sort_values("accuracy")
        st.bar_chart(df_pc.set_index("class")["accuracy"], height=320)
        with st.expander("Per-class table"):
            st.dataframe(df_pc, use_container_width=True)

        # misclassification analysis
        st.divider()
        st.subheader("Misclassification analysis")
        st.caption(
            "Which class pairs the model can't tell apart. Use this to decide "
            "where to focus oversampling, augmentation, or model design."
        )

        tab1, tab2, tab3 = st.tabs([
            "Top confused pairs", "Worst classes", "Drill into one class",
        ])

        with tab1:
            n_pairs = st.slider("Show top N pairs", 5, 50, 20, key="n_pairs")
            df_pairs = top_confused_pairs(y_true, y_pred, CLASS_NAMES, top_n=n_pairs)
            if df_pairs.empty:
                st.success("No misclassifications found.")
            else:
                st.dataframe(
                    df_pairs.style.format({
                        "Errors": "{:d}",
                        "% of true class": "{:.1f}%",
                    }).background_gradient(subset=["Errors"], cmap="Reds"),
                    use_container_width=True, height=min(540, 36 + 36 * len(df_pairs)),
                )
                st.caption(
                    f"Total errors: {int(df_pairs['Errors'].sum())} "
                    f"(of {len(y_true)} test images, "
                    f"{(y_true != y_pred).sum() / len(y_true) * 100:.2f}% error rate)"
                )

        with tab2:
            df_worst = worst_classes(y_true, y_pred, CLASS_NAMES)
            st.dataframe(
                df_worst.style.format({
                    "accuracy": "{:.2%}",
                }).background_gradient(subset=["accuracy"], cmap="RdYlGn"),
                use_container_width=True, height=min(620, 36 + 36 * len(df_worst)),
            )
            st.caption(
                "`most confused with` shows the single most-frequent wrong "
                "label this true class is predicted as."
            )

        with tab3:
            target = st.selectbox(
                "True class", CLASS_NAMES,
                index=int(df_pc.iloc[0].name) if len(df_pc) else 0,
                key="drill_class",
            )
            target_idx = CLASS_NAMES.index(target)
            df_one = confusions_for_true_class(y_true, y_pred, CLASS_NAMES, target_idx)
            if df_one.empty:
                st.warning("No test samples for this class.")
            else:
                # bar chart with correct vs incorrect colored differently
                df_chart = df_one.copy()
                df_chart["bucket"] = df_chart["Correct"].map(
                    {True: "correct", False: "wrong"}
                )
                st.bar_chart(
                    df_chart.set_index("Predicted as")["%"],
                    height=320,
                )
                st.dataframe(
                    df_one.drop(columns=["Correct"]).style.format({"%": "{:.1f}%"}),
                    use_container_width=True,
                )
                n_wrong = int(df_one[~df_one["Correct"]]["Count"].sum())
                n_total = int(df_one["Count"].sum())
                st.caption(
                    f"Of {n_total} test images of `{target}`, "
                    f"{n_total - n_wrong} were predicted correctly and "
                    f"{n_wrong} were misclassified."
                )

# 3. CROSS-MODEL COMPARISON
elif page == "Cross-model comparison":
    st.header("Cross-model comparison")
    if not AVAILABLE:
        st.warning("No models available yet.")
        st.stop()

    df = build_comparison(AVAILABLE)
    st.dataframe(df, use_container_width=True)

    if "Params (M)" in df.columns and df["Params (M)"].notna().any():
        st.divider()
        st.subheader("Accuracy vs parameters")
        st.scatter_chart(
            df.dropna(subset=["Params (M)", "Accuracy"