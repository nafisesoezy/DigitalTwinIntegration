#!/usr/bin/env python3
"""
Grouped horizontal bar chart: Accuracy per viewpoint per method
(WITHOUT missing).
"""

import os
import sys
from typing import Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt


INPUT_CSV = os.environ.get(
    "GT_REPORT_PATH",
    "allLLM_match_report_groundTruth_withoutMissing.csv"
)
OUTPUT_FIG = os.environ.get(
    "OUT_FIG_ACC_BY_VIEWPOINT",
    "fig_accuracy_by_viewpoint_grouped_bars.png"
)

GT_COL = "groundTruth"
FIELD_COL = "field"

VIEWPOINT_FIELDS_RAW = { ... }  # <-- keep same as your previous code

METHODS = [
    ("Rule-based (Static)", "result"),
    ("OpenAI GPT-OSS-120B", "LLM-result-openai/gpt-oss-120b"),
    ("Mistral Small 3.2 24B Instruct", "LLM-result-mistralai/Mistral-Small-3.2-24B-Instruct-2506"),
    ("default-text-large", "LLM-result-default-text-large"),
]

POS_LABEL = "mismatch"


def norm_label(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "missing"
    s = str(x).strip().lower()
    if "mismatch" in s:
        return "mismatch"
    if "match" in s:
        return "match"
    return "missing"


def to_binary(label):
    return 1 if label == POS_LABEL else 0


def safe_div(a, b):
    return a / b if b else 0.0


def accuracy_score(y_true, y_pred):
    correct = sum(yt == yp for yt, yp in zip(y_true, y_pred))
    return safe_div(correct, len(y_true))


def build_field_map(vp_fields):
    m = {}
    for vp, fields in vp_fields.items():
        for f in fields:
            m.setdefault(f, vp)
    return m


def compute_accuracy(df):
    field_map = build_field_map(VIEWPOINT_FIELDS_RAW)

    df["_gt_norm"] = df[GT_COL].map(norm_label)
    df["_viewpoint"] = df[FIELD_COL].map(field_map)

    df_eval = df[(df["_gt_norm"] != "missing") & (df["_viewpoint"].notna())].copy()

    vp_order = list(VIEWPOINT_FIELDS_RAW.keys())
    method_names = [m[0] for m in METHODS]

    mat = pd.DataFrame(index=vp_order, columns=method_names)

    for vp in vp_order:
        df_vp = df_eval[df_eval["_viewpoint"] == vp]
        if df_vp.empty:
            continue

        y_true = df_vp["_gt_norm"].map(to_binary).tolist()

        for name, col in METHODS:
            y_pred = df_vp[col].map(norm_label).map(to_binary).tolist()
            mat.loc[vp, name] = accuracy_score(y_true, y_pred)

    return mat


def main():
    df = pd.read_csv(INPUT_CSV)
    acc_mat = compute_accuracy(df)

    viewpoints = acc_mat.index.tolist()
    methods = acc_mat.columns.tolist()

    y = list(range(len(viewpoints)))
    bar_h = 0.18
    offsets = [(-((len(methods)-1)/2) + i)*bar_h for i in range(len(methods))]

    plt.figure(figsize=(12,5))
    ax = plt.gca()

    for i, method in enumerate(methods):
        vals = [float(acc_mat.loc[vp, method]) if pd.notna(acc_mat.loc[vp, method]) else 0.0 for vp in viewpoints]
        ax.barh([yy+offsets[i] for yy in y], vals, height=bar_h, label=method)

    ax.set_yticks(y)
    ax.set_yticklabels(viewpoints)
    ax.set_xlim(0,1.05)
    ax.set_xlabel("Accuracy")
    ax.set_title("Accuracy by viewpoint and method (without missing)")
    ax.legend(frameon=False)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=200)
    plt.close()

    print("[OK] Saved:", OUTPUT_FIG)


if __name__ == "__main__":
    main()