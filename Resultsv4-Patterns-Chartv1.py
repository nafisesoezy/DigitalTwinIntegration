#!/usr/bin/env python3
"""
Side-by-side HEATMAPS of ACCURACY per integration pattern:
  - Left:  with missing  (allLLM_match_report_groundTruth.csv)
  - Right: without missing (allLLM_match_report_groundTruth_withoutMissing.csv)

Binary evaluation: "Mismatch detection"
  positive := mismatch
  negative := everything else (match, gap, missing, ...)

We drop rows where:
- groundTruth is missing/empty
- pattern is missing/empty
- (optional) if a prediction column is missing

Outputs:
- fig_pattern_accuracy_heatmaps_with_vs_without_missing.png
- (optional) prints the ACCURACY matrices to stdout for sanity check
"""

import os
import sys
from typing import List, Tuple

import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap


# -------------------------
# Inputs / Output
# -------------------------
WITH_MISSING_CSV = os.environ.get(
    "GT_REPORT_PATH_WITH_MISSING",
    "allLLM_match_report_groundTruth.csv",
)
WITHOUT_MISSING_CSV = os.environ.get(
    "GT_REPORT_PATH_WITHOUT_MISSING",
    "allLLM_match_report_groundTruth_withoutMissing.csv",
)

OUTPUT_FIG = os.environ.get(
    "OUT_FIG_HEATMAPS_PATTERN",
    "fig_pattern_accuracy_heatmaps_with_vs_without_missing.png",
)

GT_COL = "groundTruth"
PATTERN_COL = "pattern"

METHODS: List[Tuple[str, str]] = [
    ("Rule-based (Static)", "result"),
    ("OpenAI GPT-OSS-120B", "LLM-result-openai/gpt-oss-120b"),
    ("Mistral Small 3.2 24B Instruct", "LLM-result-mistralai/Mistral-Small-3.2-24B-Instruct-2506"),
    ("Llama 3.3 70B Instruct (AWQ)", "LLM-result-default-text-large"),
]

POS_LABEL = "mismatch"


# -------------------------
# Helpers
# -------------------------
def norm_label(x: object) -> str:
    """Normalize labels to: 'mismatch', 'match', 'missing', or other."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "missing"
    s = str(x).strip().lower()
    if not s or s == "nan":
        return "missing"
    if "mismatch" in s:
        return "mismatch"
    if "match" in s:
        return "match"
    if "missing" in s:
        return "missing"
    return s  # e.g., gap/other


def to_binary(label: str) -> int:
    """Binary mapping: 1 if mismatch else 0."""
    return 1 if label == POS_LABEL else 0


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def compute_accuracy(y_true, y_pred) -> float:
    """Accuracy = fraction of correct binary predictions."""
    correct = sum(yt == yp for yt, yp in zip(y_true, y_pred))
    return safe_div(correct, len(y_true))


def compute_accuracy_matrix_by_pattern(csv_path: str) -> pd.DataFrame:
    """Return a dataframe: rows=patterns, cols=methods, values=Accuracy."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    for col in [GT_COL, PATTERN_COL]:
        if col not in df.columns:
            raise KeyError(f"[{csv_path}] Missing '{col}'. Available: {list(df.columns)}")

    for _, pred_col in METHODS:
        if pred_col not in df.columns:
            raise KeyError(f"[{csv_path}] Missing prediction column '{pred_col}'. Available: {list(df.columns)}")

    # Normalize
    df["_gt_norm"] = df[GT_COL].map(norm_label)

    # Keep only evaluable rows: non-missing GT + pattern exists
    df_eval = df[(df["_gt_norm"] != "missing") & (df[PATTERN_COL].notna())].copy()
    if df_eval.empty:
        raise ValueError(f"[{csv_path}] No evaluable rows after filtering (non-missing GT + pattern exists).")

    patterns = sorted(df_eval[PATTERN_COL].astype(str).unique().tolist())
    method_names = [m[0] for m in METHODS]

    mat = pd.DataFrame(index=patterns, columns=method_names, dtype=float)

    # Precompute normalized preds
    for _, pred_col in METHODS:
        df_eval[f"_pred__{pred_col}"] = df_eval[pred_col].map(norm_label)

    for pat in patterns:
        df_pat = df_eval[df_eval[PATTERN_COL].astype(str) == pat]
        if df_pat.empty:
            continue

        y_true = df_pat["_gt_norm"].map(to_binary).tolist()

        for method_name, pred_col in METHODS:
            y_pred = df_pat[f"_pred__{pred_col}"].map(to_binary).tolist()
            mat.loc[pat, method_name] = compute_accuracy(y_true, y_pred)

    return mat


def draw_heatmap(ax, mat: pd.DataFrame, title: str):
    data = mat.values.astype(float)

    # Create lighter red → green colormap (red=0, green=1)
    base = mpl.colormaps["RdYlGn"]
    colors = base(np.linspace(0.15, 0.85, 256))  # trim darkest ends
    soft_cmap = LinearSegmentedColormap.from_list("SoftRdYlGn", colors)
    soft_cmap.set_bad(color="white")  # NaNs shown as white

    im = ax.imshow(
        data,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        cmap=soft_cmap,
        interpolation="nearest"
    )

    ax.set_title(title)

    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=20, ha="right")

    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index)

    # Annotate
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iat[i, j]
            if pd.isna(val):
                continue
            ax.text(
                j, i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="black"
            )

    ax.grid(False)
    return im


# -------------------------
# Main
# -------------------------
def main():
    mat_with = compute_accuracy_matrix_by_pattern(WITH_MISSING_CSV)
    mat_without = compute_accuracy_matrix_by_pattern(WITHOUT_MISSING_CSV)

    # sanity print
    print("\n--- Accuracy matrix by pattern (with missing) ---")
    print(mat_with.round(2).to_string())
    print("\n--- Accuracy matrix by pattern (without missing) ---")
    print(mat_without.round(2).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.0), sharey=True)

    im1 = draw_heatmap(axes[0], mat_with, "Accuracy by pattern (with missing)")
    im2 = draw_heatmap(axes[1], mat_without, "Accuracy by pattern (without missing)")

    # shared colorbar
    cbar = fig.colorbar(im2, ax=axes.ravel().tolist(), fraction=0.035, pad=0.02)
    cbar.set_label("Accuracy (Mismatch as positive)")
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    fig.suptitle("Detection performance by integration pattern: effect of missing metadata", y=1.02, fontsize=14)
    fig.subplots_adjust(top=0.88)

    fig.savefig(OUTPUT_FIG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[OK] Saved heatmaps to: {OUTPUT_FIG}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)