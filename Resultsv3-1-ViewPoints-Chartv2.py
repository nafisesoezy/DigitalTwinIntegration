#!/usr/bin/env python3
"""
Side-by-side HEATMAPS of F1-score per viewpoint:
  - Left:  with missing  (allLLM_match_report_groundTruth.csv)
  - Right: without missing (allLLM_match_report_groundTruth_withoutMissing.csv)

Binary evaluation: "Mismatch detection"
  positive := mismatch
  negative := everything else (match, gap, missing, ...)

We drop rows where:
- groundTruth is missing/empty
- field is not in any viewpoint list

Outputs:
- fig_viewpoint_f1_heatmaps_with_vs_without_missing.png
- (optional) also prints the F1 matrices to stdout for sanity check
"""

import os
import sys
from typing import Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt


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
    "OUT_FIG_HEATMAPS",
    "fig_viewpoint_f1_heatmaps_with_vs_without_missing.png",
)

GT_COL = "groundTruth"
FIELD_COL = "field"

VIEWPOINT_FIELDS_RAW: Dict[str, List[str]] = {
    "domain": [
        "Title", "Model Version", "Description", "Keywords", "Model Type", "Scope",
        "Purpose & Pattern", "Assumptions", "Links to Publications & Reports",
        "Authors’ Unique Identifier", "Conceptual Model Evaluation", "Calibration Tools/Data",
        "Validation Capabilities", "Sensitivity Analysis", "Uncertainty Analysis",
    ],
    "information": [
        "A.output", "A.input vs AB.input", "B.output vs AB.output",
        "time_steps_temporal_resolution", "temporal_extent_coverage",
        "spatial_resolution", "spatial_extent_coverage", "dimensionality",
        "communication_mechanism", "file_formats",
    ],
    "computational": [
        "error_handling",
    ],
    "engineering": [
        "parallel_execution", "execution_constraints", "acknowledgment_protocols",
        "latency_expectations", "data_synchronization",
    ],
    "technology": [
        "programming_language",
        "availability_of_source_code (A)", "availability_of_source_code (B)",
        "implementation_verification (A)", "implementation_verification (B)",
        "software_specification_and_requirements", "hardware_specification_and_requirements",
        "distribution_version", "execution_instructions", "license",
        "landing_page (A)", "landing_page (B)",
    ],
}

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
    return s  # e.g., gap


def to_binary(label: str) -> int:
    """Binary mapping: 1 if mismatch else 0."""
    return 1 if label == POS_LABEL else 0


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def compute_f1(y_true, y_pred) -> float:
    tp = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))

    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    f1 = safe_div(2 * prec * rec, prec + rec)
    return f1


def build_field_to_viewpoint_map(vp_fields: Dict[str, List[str]]) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for vp, fields in vp_fields.items():
        for f in fields:
            m.setdefault(f, vp)
    return m


def compute_f1_matrix(csv_path: str) -> pd.DataFrame:
    """Return a dataframe: rows=viewpoints, cols=methods, values=F1."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    for col in [GT_COL, FIELD_COL]:
        if col not in df.columns:
            raise KeyError(f"[{csv_path}] Missing '{col}'. Available: {list(df.columns)}")

    for _, pred_col in METHODS:
        if pred_col not in df.columns:
            raise KeyError(f"[{csv_path}] Missing prediction column '{pred_col}'. Available: {list(df.columns)}")

    field_to_vp = build_field_to_viewpoint_map(VIEWPOINT_FIELDS_RAW)

    df["_gt_norm"] = df[GT_COL].map(norm_label)
    df["_viewpoint"] = df[FIELD_COL].map(field_to_vp)

    # keep only evaluable rows
    df_eval = df[(df["_gt_norm"] != "missing") & (df["_viewpoint"].notna())].copy()
    if df_eval.empty:
        raise ValueError(f"[{csv_path}] No evaluable rows after filtering (non-missing GT + viewpoint membership).")

    vp_order = list(VIEWPOINT_FIELDS_RAW.keys())
    method_names = [m[0] for m in METHODS]

    mat = pd.DataFrame(index=vp_order, columns=method_names, dtype=float)

    for vp in vp_order:
        df_vp = df_eval[df_eval["_viewpoint"] == vp]
        if df_vp.empty:
            # keep as NaN (will show blank)
            continue

        y_true = df_vp["_gt_norm"].map(to_binary).tolist()

        for method_name, pred_col in METHODS:
            y_pred = df_vp[pred_col].map(norm_label).map(to_binary).tolist()
            mat.loc[vp, method_name] = compute_f1(y_true, y_pred)

    return mat


import numpy as np
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

def draw_heatmap(ax, mat: pd.DataFrame, title: str):
    data = mat.values.astype(float)

    # Create lighter red → green colormap
    base = mpl.colormaps["RdYlGn"]
    colors = base(np.linspace(0.15, 0.85, 256))  # trim darkest ends
    soft_cmap = LinearSegmentedColormap.from_list(
        "SoftRdYlGn", colors
    )
    soft_cmap.set_bad(color="white")

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
    mat_with = compute_f1_matrix(WITH_MISSING_CSV)
    mat_without = compute_f1_matrix(WITHOUT_MISSING_CSV)

    # sanity print
    print("\n--- F1 matrix (with missing) ---")
    print(mat_with.round(2).to_string())
    print("\n--- F1 matrix (without missing) ---")
    print(mat_without.round(2).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(16, 4.8), sharey=True)

    im1 = draw_heatmap(axes[0], mat_with, "F1 by viewpoint (with missing)")
    im2 = draw_heatmap(axes[1], mat_without, "F1 by viewpoint (without missing)")

    # shared colorbar
    cbar = fig.colorbar(im2, ax=axes.ravel().tolist(), fraction=0.035, pad=0.02)
    cbar.set_label("F1-score (Mismatch as positive)")

    fig.suptitle("Detection performance by viewpoint: effect of missing metadata", y=1.02, fontsize=14)
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