#!/usr/bin/env python3
"""
ONE figure with TWO panels side-by-side:

Left panel:  GroundTruth = Mismatch  -> per method, 100%-stacked prediction distribution
Right panel: GroundTruth = Match     -> per method, 100%-stacked prediction distribution

X-axis: method names
Y-axis: percentage (0..100)
Each bar sums to 100 and is stacked by predicted class:
  Predicted Mismatch / Predicted Match / Predicted Gap
Gap aggregates: "gap" + "missing" + empty/unknown.

This matches your request: "both in one chart beside each other".
"""

import os
import sys
from typing import List, Tuple

import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Config
# -------------------------
INPUT_CSV = os.environ.get("GT_REPORT_PATH", "allLLM_match_report_groundTruth.csv")
OUTPUT_FIG = os.environ.get("OUT_FIG_BOTH", "fig_gt_match_vs_mismatch_side_by_side.png")

GT_COL = "groundTruth"

METHODS: List[Tuple[str, str]] = [
    ("Rule-based (Static)", "result"),
    ("OpenAI GPT-OSS-120B", "LLM-result-openai/gpt-oss-120b"),
    ("Mistral Small 3.2 24B Instruct", "LLM-result-mistralai/Mistral-Small-3.2-24B-Instruct-2506"),
    ("Llama 3.3 70B Instruct (AWQ)", "LLM-result-default-text-large"),
]

PRED_CLASSES = ["mismatch", "match", "gap"]  # stacked segments order


# -------------------------
# Helpers
# -------------------------
def norm_label_3(x: object) -> str:
    """Normalize to one of: match / mismatch / gap (gap includes missing/unknown)."""
    if x is None:
        return "gap"
    if isinstance(x, float) and pd.isna(x):
        return "gap"
    s = str(x).strip().lower()
    if not s or s == "nan":
        return "gap"
    if "mismatch" in s:
        return "mismatch"
    if "missing" in s or "gap" in s:
        return "gap"
    if s == "match" or s.startswith("match") or " match" in s:
        return "match"
    return "gap"


def compute_distribution_for_gt(df: pd.DataFrame, gt_value: str) -> pd.DataFrame:
    """
    For fixed GT (match or mismatch), compute per-method percent distribution of predictions
    across PRED_CLASSES. Returns columns: Method, n, mismatch, match, gap.
    """
    df_gt = df[df["_gt"] == gt_value].copy()

    rows = []
    for method_name, pred_col in METHODS:
        preds = df_gt[pred_col].map(norm_label_3)
        n = len(preds)
        counts = preds.value_counts().to_dict()

        row = {"Method": method_name, "n": n}
        for cls in PRED_CLASSES:
            row[cls] = (counts.get(cls, 0) / n * 100.0) if n else 0.0
        rows.append(row)

    return pd.DataFrame(rows)


def plot_panel(ax, df_dist: pd.DataFrame, title: str):
    methods = df_dist["Method"].tolist()
    x = list(range(len(methods)))

    # distinct segment colors
    color_map = {
        "match": "#4CAF50",  # professional green
        "mismatch": "#E57373",  # soft red
        "gap": "#FFB74D"  # orange
    }


    bottom = [0.0] * len(methods)
    for cls in PRED_CLASSES:
        vals = df_dist[cls].tolist()
        ax.bar(
            x,
            vals,
            bottom=bottom,
            color=color_map[cls],
            edgecolor="white",
            linewidth=0.7,
            label=f"Predicted {cls.capitalize()}",
        )
        bottom = [b + v for b, v in zip(bottom, vals)]

    # annotate n at top
    for i, n in enumerate(df_dist["n"].tolist()):
        ax.text(i, 102, f"n={int(n)}", ha="center", va="bottom", fontsize=9)

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=18, ha="right")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Percentage of predictions")
    ax.grid(axis="y", linestyle="--", alpha=0.3)


# -------------------------
# Main
# -------------------------
def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    if GT_COL not in df.columns:
        raise KeyError(f"Missing '{GT_COL}' column. Available: {list(df.columns)}")

    for method_name, col in METHODS:
        if col not in df.columns:
            raise KeyError(f"Missing prediction column '{col}' for method '{method_name}'. Available: {list(df.columns)}")

    df["_gt"] = df[GT_COL].map(norm_label_3)

    dist_mismatch = compute_distribution_for_gt(df, "mismatch")
    dist_match = compute_distribution_for_gt(df, "match")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.2), sharey=True)

    plot_panel(
        axes[0],
        dist_mismatch,
        title="A) Ground Truth = Mismatch",
    )
    plot_panel(
        axes[1],
        dist_match,
        title="B) Ground Truth = Match",
    )

    # Single shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)

    fig.suptitle("Per-class prediction behavior by method (Match vs Mismatch)", y=1.02, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    fig.savefig(OUTPUT_FIG, dpi=200)
    plt.close(fig)

    print(f"[OK] Saved figure to: {OUTPUT_FIG}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)