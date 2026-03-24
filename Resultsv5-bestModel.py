#!/usr/bin/env python3
"""
Create the 3 *descriptive overview* charts for the BEST model:
  OpenAI GPT-OSS-120B  -> column: "LLM-result-openai/gpt-oss-120b"
from:
  allLLM_match_report_groundTruth.csv

Charts (prediction-based, NOT ground truth):
1) Pie grid: Match / Mismatch / Gap by integration pattern
   -> fig9_pie_match_mismatch_gap_by_pattern_nolabels_GPTOSS.png

2) Stacked bars: Match / Mismatch / Gap by RM-ODP viewpoint across patterns
   -> fig10_viewpoint_by_pattern_distribution_GPTOSS.png

3) Pie: mismatch types (bottleneck distribution) for predicted mismatches
   -> fig11_pie_mismatch_by_bottleneck_other3_GPTOSS.png

Notes:
- Uses ONLY the GPT-OSS-120B prediction column.
- "Missing" predictions are excluded from the 3-class descriptive plots (Match/Mismatch/Gap),
  because your figures focus on those three outcomes.
- For bottleneck pie: we count bottleneck types only among rows predicted as Mismatch.

Run:
  python make_descriptive_charts_best_model.py

Optional env vars:
  GT_REPORT_PATH_WITH_MISSING : input csv path (default: allLLM_match_report_groundTruth.csv)
  OUT_DIR                    : output folder (default: Figures)
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Config
# -------------------------
INPUT_CSV = os.environ.get(
    "GT_REPORT_PATH_WITH_MISSING",
    "allLLM_match_report_groundTruth.csv",
)

OUT_DIR = os.environ.get("OUT_DIR", "Figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Best model prediction column
PRED_COL = "LLM-result-openai/gpt-oss-120b"

PATTERN_COL = "pattern"
FIELD_COL = "field"
BOTTLENECK_COL = "bottleneck"

# Outcomes shown in the descriptive figures
OUTCOMES_3 = ["match", "mismatch", "gap"]

# Your viewpoint field mapping (same as your evaluation scripts)
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
    "Computational": ["error_handling"],
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


# -------------------------
# Helpers
# -------------------------
def norm_label(x: object) -> str:
    """
    Normalize to one of:
      match | mismatch | gap | missing | other
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
    if "gap" in s:
        return "gap"
    if "missing" in s:
        return "missing"
    return "other"


def build_field_to_viewpoint_map(vp_fields: Dict[str, List[str]]) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for vp, fields in vp_fields.items():
        for f in fields:
            m.setdefault(f, vp)
    return m


def autopct_no_labels(pct: float) -> str:
    # no wedge labels (match your "_nolabels_" style)
    return ""


# -------------------------
# Chart 1: Pie grid by pattern
# -------------------------
def plot_pie_grid_by_pattern(df: pd.DataFrame, outpath: str):
    # keep only rows with pattern and a 3-class outcome (exclude missing/other)
    d = df[df[PATTERN_COL].notna()].copy()
    d["_pred"] = d[PRED_COL].map(norm_label)
    d = d[d["_pred"].isin(OUTCOMES_3)].copy()

    patterns = sorted(d[PATTERN_COL].astype(str).unique().tolist())
    if not patterns:
        raise ValueError("No patterns found after filtering.")

    # grid layout
    n = len(patterns)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.2 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    # legend labels in fixed order
    legend_labels = ["Predicted Mismatch", "Predicted Match", "Predicted Gap"]
    legend_keys = ["mismatch", "match", "gap"]

    for idx, pat in enumerate(patterns):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        dp = d[d[PATTERN_COL].astype(str) == pat]

        counts = dp["_pred"].value_counts().reindex(legend_keys).fillna(0).astype(int)
        values = counts.values

        # avoid zero-sum pies
        if values.sum() == 0:
            ax.axis("off")
            continue

        ax.pie(
            values,
            startangle=90,
            autopct=autopct_no_labels,
            wedgeprops=dict(edgecolor="white", linewidth=1),
        )
        ax.set_title(str(pat))
        ax.text(0, 1.15, f"n={values.sum()}", ha="center", va="center", fontsize=10)

    # turn off unused axes
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")

    fig.legend(
        legend_labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle("Matches, mismatches, and metadata gaps across integration patterns (GPT-OSS-120B)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -------------------------
# Chart 2: Viewpoint x Pattern distribution (stacked bars)
# -------------------------
def plot_viewpoint_by_pattern_distribution(df: pd.DataFrame, outpath: str):
    field_to_vp = build_field_to_viewpoint_map(VIEWPOINT_FIELDS_RAW)

    d = df.copy()
    d["_pred"] = d[PRED_COL].map(norm_label)
    d["_viewpoint"] = d[FIELD_COL].map(field_to_vp)

    # keep only evaluable viewpoint+pattern and 3-class outcome
    d = d[(d["_viewpoint"].notna()) & (d[PATTERN_COL].notna())].copy()
    d = d[d["_pred"].isin(OUTCOMES_3)].copy()

    viewpoints = list(VIEWPOINT_FIELDS_RAW.keys())
    patterns = sorted(d[PATTERN_COL].astype(str).unique().tolist())
    if not patterns:
        raise ValueError("No patterns found after filtering for viewpoint-pattern distribution.")

    # build counts: (viewpoint, pattern) -> counts for match/mismatch/gap
    rows = []
    for vp in viewpoints:
        dv = d[d["_viewpoint"] == vp]
        for pat in patterns:
            dp = dv[dv[PATTERN_COL].astype(str) == pat]
            vc = dp["_pred"].value_counts().to_dict()
            rows.append({
                "viewpoint": vp,
                "pattern": pat,
                "match": int(vc.get("match", 0)),
                "mismatch": int(vc.get("mismatch", 0)),
                "gap": int(vc.get("gap", 0)),
            })
    counts = pd.DataFrame(rows)

    # plot: one subplot per viewpoint
    n_vp = len(viewpoints)
    fig, axes = plt.subplots(n_vp, 1, figsize=(1.2 * len(patterns) + 4, 2.2 * n_vp), sharex=True)

    if n_vp == 1:
        axes = [axes]

    # stacked order + legend labels
    stack_order = ["mismatch", "match", "gap"]
    legend_labels = ["Predicted Mismatch", "Predicted Match", "Predicted Gap"]

    for ax, vp in zip(axes, viewpoints):
        sub = counts[counts["viewpoint"] == vp].set_index("pattern").reindex(patterns).fillna(0)

        bottom = np.zeros(len(patterns))
        for key in stack_order:
            vals = sub[key].values.astype(float)
            ax.bar(patterns, vals, bottom=bottom, label=key)
            bottom += vals

        ax.set_ylabel(vp)
        ax.grid(axis="y", linestyle="--", alpha=0.25)

    # prettify x labels
    axes[-1].set_xticks(range(len(patterns)))
    axes[-1].set_xticklabels(patterns, rotation=25, ha="right")
    axes[-1].set_xlabel("Integration Pattern")

    # custom legend labels (in same order as stack_order)
    handles, _ = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))

    fig.suptitle("Matches, mismatches, and metadata gaps by RM-ODP viewpoint across integration patterns (GPT-OSS-120B)", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -------------------------
# Chart 3: Bottleneck pie (mismatch types)
# -------------------------
def plot_bottleneck_pie(df: pd.DataFrame, outpath: str, top_k: int = 6):
    if BOTTLENECK_COL not in df.columns:
        raise KeyError(f"Missing '{BOTTLENECK_COL}' column in CSV.")

    d = df.copy()
    d["_pred"] = d[PRED_COL].map(norm_label)
    d = d[d["_pred"] == "mismatch"].copy()

    # normalize bottleneck names a bit
    d["_bottleneck"] = d[BOTTLENECK_COL].fillna("Unknown").astype(str).str.strip()
    vc = d["_bottleneck"].value_counts()

    if vc.empty:
        raise ValueError("No predicted mismatches found to plot bottleneck distribution.")

    # group small categories into "Other"
    top = vc.head(top_k)
    rest = vc.iloc[top_k:].sum()
    if rest > 0:
        top = pd.concat([top, pd.Series({"Other": rest})])

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.5))
    ax.pie(
        top.values,
        startangle=90,
        autopct=lambda p: f"{p:.0f}%" if p >= 5 else "",
        wedgeprops=dict(edgecolor="white", linewidth=1),
    )
    ax.set_title("Distribution of mismatch types (predicted mismatches only, GPT-OSS-120B)")
    ax.legend(top.index.tolist(), loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -------------------------
# Main
# -------------------------
def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    # sanity checks
    required = [PRED_COL, PATTERN_COL, FIELD_COL, BOTTLENECK_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}\nAvailable: {list(df.columns)}")

    out1 = os.path.join(OUT_DIR, "fig9_pie_match_mismatch_gap_by_pattern_nolabels_GPTOSS.png")
    out2 = os.path.join(OUT_DIR, "fig10_viewpoint_by_pattern_distribution_GPTOSS.png")
    out3 = os.path.join(OUT_DIR, "fig11_pie_mismatch_by_bottleneck_other3_GPTOSS.png")

    plot_pie_grid_by_pattern(df, out1)
    plot_viewpoint_by_pattern_distribution(df, out2)
    plot_bottleneck_pie(df, out3, top_k=6)

    print("[OK] Saved:")
    print(" -", out1)
    print(" -", out2)
    print(" -", out3)


if __name__ == "__main__":
    main()