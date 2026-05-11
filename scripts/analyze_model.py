"""Run after each Colab training: builds figures + tables for ONE model.

Usage:
    python scripts/analyze_model.py --model convnext_tiny
    python scripts/analyze_model.py --model swin_v2_t --results-root results
"""
from __future__ import annotations
import argparse, sys, yaml
from pathlib import Path

# ensure src/ importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classification import metrics as M
from src.classification import visualize as V
from src.classification import compare as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Model name, must match a key in configs/classification.yaml")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--reports-root", default="reports")
    ap.add_argument("--config", default="configs/classification.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    class_names = cfg["dataset"]["classes"]
    if args.model not in cfg["models"]:
        raise SystemExit(f"Unknown model '{args.model}'. "
                         f"Known: {list(cfg['models'])}")

    bundle = C.load_model_bundle(args.results_root, args.model)

    fig_dir   = Path(args.reports_root) / "figures"
    table_dir = Path(args.reports_root) / "tables"
    cls_dir   = table_dir / "classification_reports"
    for d in (fig_dir, table_dir, cls_dir):
        d.mkdir(parents=True, exist_ok=True)

    name = args.model
    n = len(class_names)

    # 1) training curves
    fig = V.plot_training_curves(bundle["history"], title=name)
    fig.savefig(fig_dir / f"{name}_curves.png", dpi=150)

    # 2) confusion matrix (row-normalised)
    cm = M.confusion(bundle["y_true"], bundle["y_pred"], num_classes=n)
    fig = V.plot_confusion(cm, class_names, title=f"{name} — confusion matrix")
    fig.savefig(fig_dir / f"{name}_confusion.png", dpi=150)

    # 3) ROC
    fig = V.plot_roc(bundle["y_true"], bundle["y_proba"], class_names, title=name)
    fig.savefig(fig_dir / f"{name}_roc.png", dpi=150)

    # 4) per-class F1 bars + per-class AP merged into the same table
    per_class = M.compute_per_class(bundle["y_true"], bundle["y_pred"], class_names)
    per_class_ap = M.compute_per_class_ap(bundle["y_true"], bundle["y_proba"], class_names)
    per_class["ap"] = per_class_ap["ap"].values
    fig = V.plot_per_class_bars(per_class, metric="f1",
                                title=f"{name} — per-class F1 (red = weakest 5)")
    fig.savefig(fig_dir / f"{name}_perclass_f1.png", dpi=150)
    per_class.to_csv(table_dir / f"{name}_per_class.csv", index=False)

    # 5) text classification report
    report = M.text_classification_report(bundle["y_true"], bundle["y_pred"], class_names)
    (cls_dir / f"{name}.txt").write_text(report)

    # 6) summary
    summary = M.compute_summary(bundle["y_true"], bundle["y_pred"],
                                bundle["y_proba"], num_classes=n)
    print(f"\n=== {name} ===")
    for k, v in summary.items():
        print(f"  {k:>20s}: {v:.4f}")
    print(f"\nFigures saved to: {fig_dir}/{name}_*.png")
    print(f"Tables saved to:  {table_dir}/{name}_per_class.csv")
    print(f"Report saved to:  {cls_dir}/{name}.txt")


if __name__ == "__main__":
    main()
