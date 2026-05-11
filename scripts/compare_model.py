"""Build the cross-model comparison artifacts (incl. mAP) for every model
that has been trained and dropped into results/.

Usage:
    python scripts/compare_models.py
    python scripts/compare_models.py --models convnext_tiny vit_b_16 swin_v2_t
    python scripts/compare_models.py --results-root results --reports-root reports

Outputs (under reports/):
    tables/summary.csv               -- one row per model, all top-line metrics
    tables/per_class_long.csv        -- long format (model, class) -> p/r/f1/ap
    tables/per_class_ap_pivot.csv    -- wide: rows = classes, cols = models
    figures/mAP_comparison.png       -- bar chart, macro-mAP across models
    figures/macro_f1_vs_map.png      -- scatter (sanity: ranks should align)
"""
from __future__ import annotations
import argparse, sys, yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from src.classification import compare as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="Subset of model names to compare. Default: all in results/")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--reports-root", default="reports")
    ap.add_argument("--config", default="configs/classification.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    class_names = cfg["dataset"]["classes"]

    models = args.models or C.discover_trained_models(args.results_root)
    if not models:
        raise SystemExit(f"No trained models found under {args.results_root}/")
    print(f"Comparing {len(models)} model(s): {models}")

    fig_dir   = Path(args.reports_root) / "figures"
    table_dir = Path(args.reports_root) / "tables"
    for d in (fig_dir, table_dir):
        d.mkdir(parents=True, exist_ok=True)

    # -- 1. summary table (incl. mAP) --
    summary = C.build_summary_table(args.results_root, models, class_names)
    summary.to_csv(table_dir / "summary.csv", index=False)
    print("\n=== Summary (sorted by macro mAP) ===")
    print(summary.sort_values("map_macro", ascending=False).to_string(index=False))

    # -- 2. per-class long table (precision/recall/f1/support/ap) --
    long_df = C.build_per_class_table(args.results_root, models, class_names)
    long_df.to_csv(table_dir / "per_class_long.csv", index=False)

    # -- 3. per-class AP pivot (rows = classes, cols = models, last row = mAP) --
    pivot = C.build_per_class_ap_pivot(args.results_root, models, class_names)
    pivot.to_csv(table_dir / "per_class_ap_pivot.csv")
    print("\n=== Per-class AP pivot ===")
    print(pivot.round(4).to_string())

    # -- 4. mAP bar chart --
    s = summary.sort_values("map_macro", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(s) + 1.5))
    ax.barh(s["model"], s["map_macro"])
    for i, v in enumerate(s["map_macro"]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center")
    ax.set_xlabel("Macro mAP (one-vs-rest)")
    ax.set_xlim(0, 1.0)
    ax.set_title("WaRP-C: Macro mAP across models")
    fig.tight_layout()
    fig.savefig(fig_dir / "mAP_comparison.png", dpi=150)

    # -- 5. macro-F1 vs macro-mAP scatter 
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(summary["macro_f1"], summary["map_macro"])
    for _, row in summary.iterrows():
        ax.annotate(row["model"],
                    (row["macro_f1"], row["map_macro"]),
                    textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("Macro F1 (single-threshold)")
    ax.set_ylabel("Macro mAP (threshold-free)")
    ax.set_title("F1 vs mAP — should rank similarly")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "macro_f1_vs_map.png", dpi=150)

    print(f"\nTables saved to:  {table_dir}/")
    print(f"Figures saved to: {fig_dir}/")


if __name__ == "__main__":
    main()
