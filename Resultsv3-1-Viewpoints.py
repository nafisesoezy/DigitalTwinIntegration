#!/usr/bin/env python3
"""
Create a LaTeX detection-performance table *per viewpoint* from:
  allLLM_match_report_groundTruth.csv

For each viewpoint, we evaluate each method by comparing its prediction column
to `groundTruth`, using binary "Mismatch detection" metrics:
  positive := mismatch
  negative := everything else (match, missing, gap, etc.)

We drop rows where:
- groundTruth is missing/empty
- field is not in any viewpoint list

Output:
- detection_performance_by_viewpoint.tex
"""

import os
from typing import Dict, List, Tuple
import pandas as pd

#INPUT_CSV  = os.environ.get("GT_REPORT_PATH", "allLLM_match_report_groundTruth.csv")
INPUT_CSV  = os.environ.get("GT_REPORT_PATH", "allLLM_match_report_groundTruth_withoutMissing.csv")


OUTPUT_TEX = os.environ.get("DETECTION_TABLE_TEX", "detection_performance_by_viewpoint.tex")

GT_COL = "groundTruth"
FIELD_COL = "field"

VIEWPOINT_FIELDS_RAW: Dict[str, List[str]] = {
    "domain": [
        "Title",
        "Model Version",
        "Description",
        "Keywords",
        "Model Type",
        "Scope",
        "Purpose & Pattern",
        "Assumptions",
        "Links to Publications & Reports",
        "Authors’ Unique Identifier",
        "Conceptual Model Evaluation",
        "Calibration Tools/Data",
        "Validation Capabilities",
        "Sensitivity Analysis",
        "Uncertainty Analysis",
    ],
    "information": [
        "A.output",
        "A.input vs AB.input",
        "B.output vs AB.output",
        "time_steps_temporal_resolution",
        "temporal_extent_coverage",
        "spatial_resolution",
        "spatial_extent_coverage",
        "dimensionality",
        "communication_mechanism",
        "file_formats",
    ],
    "computational": [
        "error_handling",
    ],
    "engineering": [
        "parallel_execution",
        "execution_constraints",
        "acknowledgment_protocols",
        "latency_expectations",
        "data_synchronization",
    ],
    "technology": [
        "programming_language",
        "availability_of_source_code (A)",
        "availability_of_source_code (B)",
        "implementation_verification (A)",
        "implementation_verification (B)",
        "software_specification_and_requirements",
        "hardware_specification_and_requirements",
        "distribution_version",
        "execution_instructions",
        "license",
        "landing_page (A)",
        "landing_page (B)",
    ],
}

METHODS: List[Tuple[str, str]] = [
    ("Rule-based (Static)", "result"),
    ("OpenAI GPT-OSS-120B", "LLM-result-openai/gpt-oss-120b"),
    ("Mistral Small 3.2 24B Instruct", "LLM-result-mistralai/Mistral-Small-3.2-24B-Instruct-2506"),
    ("default-text-large", "LLM-result-default-text-large"),
]

POS_LABEL = "mismatch"  # positive class keyword


# -----------------------
# Helpers
# -----------------------
def norm_label(x: object) -> str:
    """Normalize labels to: 'mismatch', 'match', 'missing', or 'other'."""
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
    return s  # keep other labels as-is (e.g., 'gap')


def to_binary(label: str) -> int:
    """Binary mapping: 1 if mismatch else 0."""
    return 1 if label == POS_LABEL else 0


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def compute_metrics(y_true, y_pred):
    """Return accuracy, precision, recall, f1 for binary classification."""
    tp = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    tn = sum((yt == 0 and yp == 0) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))

    acc = safe_div(tp + tn, tp + tn + fp + fn)
    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    f1 = safe_div(2 * prec * rec, prec + rec)
    return acc, prec, rec, f1


def latex_escape(s: str) -> str:
    """Escape minimal LaTeX specials for method/viewpoint names."""
    return (s.replace("\\", r"\textbackslash{}")
             .replace("&", r"\&")
             .replace("%", r"\%")
             .replace("$", r"\$")
             .replace("#", r"\#")
             .replace("_", r"\_")
             .replace("{", r"\{")
             .replace("}", r"\}")
             .replace("~", r"\textasciitilde{}")
             .replace("^", r"\textasciicircum{}"))


def build_field_to_viewpoint_map(vp_fields: Dict[str, List[str]]) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for vp, fields in vp_fields.items():
        for f in fields:
            # if duplicates across viewpoints, first one wins (shouldn't happen in your lists)
            m.setdefault(f, vp)
    return m


# -----------------------
# Main
# -----------------------
def main():
    df = pd.read_csv(INPUT_CSV)

    # sanity
    for col in [GT_COL, FIELD_COL]:
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}'. Available: {list(df.columns)}")

    for _, pred_col in METHODS:
        if pred_col not in df.columns:
            raise KeyError(f"Missing prediction column '{pred_col}'. Available: {list(df.columns)}")

    field_to_vp = build_field_to_viewpoint_map(VIEWPOINT_FIELDS_RAW)

    # normalize GT, filter evaluable
    df["_gt_norm"] = df[GT_COL].map(norm_label)
    df["_viewpoint"] = df[FIELD_COL].map(field_to_vp)

    # keep only rows that belong to a viewpoint AND have GT
    df_eval = df[(df["_gt_norm"] != "missing") & (df["_viewpoint"].notna())].copy()

    if df_eval.empty:
        raise ValueError("No evaluable rows after filtering by viewpoint membership and non-missing groundTruth.")

    # compute metrics per (viewpoint, method)
    results = []  # (viewpoint, method_name, acc, prec, rec, f1, n_rows)
    for vp in VIEWPOINT_FIELDS_RAW.keys():
        df_vp = df_eval[df_eval["_viewpoint"] == vp].copy()
        if df_vp.empty:
            # keep viewpoint but with 0 rows? usually better to skip entirely
            continue

        y_true = df_vp["_gt_norm"].map(to_binary).tolist()

        for method_name, pred_col in METHODS:
            y_pred = df_vp[pred_col].map(norm_label).map(to_binary).tolist()
            acc, prec, rec, f1 = compute_metrics(y_true, y_pred)
            results.append((vp, method_name, acc, prec, rec, f1, len(df_vp)))

    # Order rows within each viewpoint (match your sample ordering)
    method_order = {m[0]: i for i, m in enumerate([
        ("Rule-based (Static)", ""),
        ("default-text-large", ""),
        ("Mistral Small 3.2 24B Instruct", ""),
        ("OpenAI GPT-OSS-120B", ""),
    ])}

    def sort_key(r):
        vp, method_name, *_ = r
        return (list(VIEWPOINT_FIELDS_RAW.keys()).index(vp), method_order.get(method_name, 999))

    results.sort(key=sort_key)

    # Build LaTeX
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Detection performance comparison between the rule-based baseline and the LLM-assisted detector (per viewpoint).}")
    lines.append(r"\label{tab:detection_performance_by_viewpoint}")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Method} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-score} \\")
    lines.append(r"\hline")

    current_vp = None
    for vp, method_name, acc, prec, rec, f1, n in results:
        if current_vp is not None and vp != current_vp:
            lines.append(r"\hline")  # separator between viewpoints
        current_vp = vp

        row_name = f"{vp}-{method_name}"
        lines.append(
            f"{latex_escape(row_name)} & {acc:.2f} & {prec:.2f} & {rec:.2f} & {f1:.2f} \\\\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)

    with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
        f.write(latex)

    print(f"Wrote LaTeX table to: {OUTPUT_TEX}")
    print("\n--- LaTeX preview ---\n")
    print(latex)


if __name__ == "__main__":
    main()