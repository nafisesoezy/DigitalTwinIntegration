#!/usr/bin/env python3
"""
HEATMAP of MACRO-F1 per viewpoint (WITH MISSING FILE ONLY):
  - Input: allLLM_match_report_groundTruth.csv

Evaluation: MULTI-CLASS exact-match over labels:
  classes := {match, mismatch, missing}
  (anything else like gap is mapped to "other" and is NOT included in macro-F1)

Macro-F1 definition:
    F1_macro = (F1_match + F1_mismatch + F1_missing) / 3

Important:
- We KEEP rows whose groundTruth is Missing (we do NOT drop them)
- We drop rows where field is not in any viewpoint list

Outputs:
- fig_viewpoint_macroF1_with_missing.png
- (optional) prints the macro-F1 matrix to stdout for sanity check
"""

import os
import sys
from typing import Dict, List, Tuple

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

OUTPUT_FIG = os.environ.get(
    "OUT_FIG_HEATMAP_VIEWPOINT_MACROF1",
    "fig_viewpoint_macroF1_with_missing.png",
)

GT_COL = "groundTruth"
FIELD_COL = "field"

VIEWPOINT_FIELDS_RAW: Dict[str, List[str]] = {
    "Domain": [
        "Title", "Model Version", "Description", "Keywords", "Model Type", "Scope",
        "Purpose & Pattern", "Assumptions", "Links to Publications & Reports",
        "Authors’ Unique Identifier", "Conceptual Model Evaluation", "Calibration Tools/Data",
        "Validation Capabilities", "Sensitivity Analysis", "Uncertainty Analysis",
    ],
    "Information": [
        "A.output", "A.input vs AB.input", "B.output vs AB.output",
        "time_steps_temporal_resolution", "temporal_extent_coverage",
        "spatial_resolution", "spatial_extent_coverage", "dimensionality",
        "communication_mechanism", "file_formats",
    ],
    "Computational": [
        "error_handling",
    ],
    "Engineering": [
        "parallel_execution", "execution_constraints", "acknowledgment_protocols",
        "latency_expectations", "data_synchronization",
    ],
    "Technology": [
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

# Macro-F1 classes (exactly what you want)
ALLOWED_LABELS = ["match", "mismatch", "missing"]


# -------------------------
# Helpers
# -------------------------
def norm_label(x: object) -> str:
    """
    Normalize to one of:
      match | mismatch | missing | other
    """
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
    return "other"  # e.g., gap/unknown


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def f1_for_class(y_true: List[str], y_pred: List[str], c: str) -> float:
    tp = sum((t == c and p == c) for t, p in zip(y_true, y_pred))
    fp = sum((t != c and p == c) for t, p in zip(y_true, y_pred))
    fn = sum((t == c and p != c) for t, p in zip(y_true, y_pred))

    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    return safe_div(2 * prec * rec, prec + rec)


def macro_f1(y_true: List[str], y_pred: List[str], classes: List[str]) -> float:
    """
    Macro-F1 = average of per-class F1 across the provided classes.
    This matches: (F1_match + F1_mismatch + F1_missing)/3
    """
    f1s = [f1_for_class(y_true, y_pred, c) for c in classes]
    return safe_div(sum(f1s), len(classes))


def build_field_to_viewpoint_map(vp_fields: Dict[str, List[str]]) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for vp, fields in vp_fields.items():
        for f in fields:
            m.setdefault(f, vp)
    return m


def compute_macro_f1_matrix_by_viewpoint(csv_path: str) -> pd.DataFrame:
    """Return a dataframe: rows=viewpoints, cols=methods, values=macro-F1."""
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

    # Normalize GT and map to viewpoints
    df["_gt_norm"] = df[GT_COL].map(norm_label)
    df["_viewpoint"] = df[FIELD_COL].map(field_to_vp)

    # Keep rows where viewpoint exists; KEEP missing GT rows (important)
    df_eval = df[df["_viewpoint"].notna()].copy()
    if df_eval.empty:
        raise ValueError(f"[{csv_path}] No evaluable rows after filtering (viewpoint membership).")

    # Normalize predictions
    for _, pred_col in METHODS:
        df_eval[f"_pred__{pred_col}"] = df_eval[pred_col].map(norm_label)

    vp_order = list(VIEWPOINT_FIELDS_RAW.keys())
    method_names = [m[0] for m in METHODS]

    mat = pd.DataFrame(index=vp_order, columns=method_names, dtype=float)

    for vp in vp_order:
        df_vp = df_eval[df_eval["_viewpoint"] == vp]
        if df_vp.empty:
            continue

        y_true = df_vp["_gt_norm"].tolist()

        for method_name, pred_col in METHODS:
            y_pred = df_vp[f"_pred__{pred_col}"].tolist()
            mat.loc[vp, method_name] = macro_f1(y_true, y_pred, ALLOWED_LABELS)

    return mat


def draw_heatmap(ax, mat: pd.DataFrame, title: str):
    data = mat.values.astype(float)

    # Lighter red → green colormap (red=0, green=1)
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
    mat_with = compute_macro_f1_matrix_by_viewpoint(WITH_MISSING_CSV)

    # sanity print
    print("\n--- F1 Score by viewpoint  ---")
    print(mat_with.round(2).to_string())

    fig, ax = plt.subplots(1, 1, figsize=(10, 4.8))

    im = draw_heatmap(ax, mat_with, "F1 Score by viewpoint")

    cbar = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.03)
    cbar.set_label("F1 Score")
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    fig.suptitle("F1 Score by viewpoint", y=1.02, fontsize=14)
    fig.subplots_adjust(top=0.88)

    fig.savefig(OUTPUT_FIG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[OK] Saved heatmap to: {OUTPUT_FIG}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)