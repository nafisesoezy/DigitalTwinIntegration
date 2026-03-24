#!/usr/bin/env python3
"""
Create a LaTeX detection-performance table *per pattern* from:
  allLLM_match_report_groundTruth.csv

For each pattern, compare each method's prediction to `groundTruth` and compute:
Accuracy, Precision, Recall, F1.

IMPORTANT (matches your “with missing” requirement):
- We KEEP rows whose groundTruth is "Missing".
- We treat this as a MULTI-CLASS exact-match evaluation over labels:
    {Match, Mismatch, Missing}
  (anything else is mapped to "Other", but you can disable that).

Precision/Recall/F1 are computed as *macro-averages* across classes.

Outputs:
- detection_performance_by_pattern_with_missing.tex
"""

import os
from typing import List, Tuple
import pandas as pd

INPUT_CSV  = os.environ.get("GT_REPORT_PATH", "allLLM_match_report_groundTruth.csv")
#INPUT_CSV  = os.environ.get("GT_REPORT_PATH", "allLLM_match_report_groundTruth_withoutMissing.csv")


OUTPUT_TEX = os.environ.get("DETECTION_TABLE_TEX", "detection_performance_by_pattern_with_missing.tex")

GT_COL = "groundTruth"
PATTERN_COL = "pattern"

METHODS: List[Tuple[str, str]] = [
    ("Rule-based (Static)", "result"),
    ("OpenAI GPT-OSS-120B", "LLM-result-openai/gpt-oss-120b"),
    ("Mistral Small 3.2 24B Instruct", "LLM-result-mistralai/Mistral-Small-3.2-24B-Instruct-2506"),
    ("default-text-large", "LLM-result-default-text-large"),
]

# If you want ONLY {Match, Mismatch, Missing}, keep as-is.
# If you also want "Gap" to be its own class, add it here and in norm_label().
ALLOWED_LABELS = {"match", "mismatch", "missing"}  # multi-class set used for macro metrics


# -----------------------
# Helpers
# -----------------------
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
    return "other"


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def accuracy(y_true: List[str], y_pred: List[str]) -> float:
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return safe_div(correct, len(y_true))


def macro_precision_recall_f1(y_true: List[str], y_pred: List[str], classes: List[str]) -> Tuple[float, float, float]:
    """
    Macro-averaged precision/recall/f1 across provided classes.
    """
    precisions, recalls, f1s = [], [], []

    for c in classes:
        tp = sum((t == c and p == c) for t, p in zip(y_true, y_pred))
        fp = sum((t != c and p == c) for t, p in zip(y_true, y_pred))
        fn = sum((t == c and p != c) for t, p in zip(y_true, y_pred))

        prec = safe_div(tp, tp + fp)
        rec  = safe_div(tp, tp + fn)
        f1   = safe_div(2 * prec * rec, prec + rec)

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    return (
        safe_div(sum(precisions), len(classes)),
        safe_div(sum(recalls), len(classes)),
        safe_div(sum(f1s), len(classes)),
    )


def latex_escape(s: str) -> str:
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


# -----------------------
# Main
# -----------------------
def main():
    df = pd.read_csv(INPUT_CSV)

    # sanity checks
    for col in [GT_COL, PATTERN_COL]:
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}'. Available: {list(df.columns)}")

    for _, pred_col in METHODS:
        if pred_col not in df.columns:
            raise KeyError(f"Missing prediction column '{pred_col}'. Available: {list(df.columns)}")

    # normalize ground truth and keep rows where GT exists (including Missing)
    df["_gt"] = df[GT_COL].map(norm_label)

    # drop rows where GT is empty/NaN in file (norm_label makes them "missing", which we KEEP)
    # So here we only drop if pattern is missing.
    df = df[df[PATTERN_COL].notna()].copy()

    # normalize predictions
    for _, pred_col in METHODS:
        df[f"_pred__{pred_col}"] = df[pred_col].map(norm_label)

    # Ensure GT and preds are in allowed set; everything else becomes "other"
    # (GT "other" is allowed; but you probably want to exclude it from macro metrics.)
    # We'll compute macro metrics on ALLOWED_LABELS only, but accuracy uses exact match on normalized labels.
    classes_for_macro = sorted(ALLOWED_LABELS)

    patterns = sorted(df[PATTERN_COL].astype(str).unique().tolist())

    rows = []  # (pattern, method, acc, prec, rec, f1)
    for pat in patterns:
        df_pat = df[df[PATTERN_COL].astype(str) == pat].copy()
        if df_pat.empty:
            continue

        y_true = df_pat["_gt"].tolist()

        for method_name, pred_col in METHODS:
            y_pred = df_pat[f"_pred__{pred_col}"].tolist()

            acc = accuracy(y_true, y_pred)
            prec, rec, f1 = macro_precision_recall_f1(y_true, y_pred, classes_for_macro)

            rows.append((pat, method_name, acc, prec, rec, f1))

    # sort: by pattern, then method order as in METHODS
    method_rank = {m[0]: i for i, m in enumerate([m[0] for m in METHODS])}
    rows.sort(key=lambda r: (r[0], method_rank.get(r[1], 999)))

    # build LaTeX
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Detection performance comparison between the rule-based baseline and the LLM-assisted detector(with missing).}")
    lines.append(r"\label{tab:detection_performance}")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Method} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-score} \\")
    lines.append(r"\hline")

    current_pat = None
    for pat, method_name, acc, prec, rec, f1 in rows:
        if current_pat is not None and pat != current_pat:
            lines.append(r"\hline")  # separator between patterns
        current_pat = pat

        method_cell = f"{pat}-{method_name}"
        lines.append(
            f"{latex_escape(method_cell)} & {acc:.2f} & {prec:.2f} & {rec:.2f} & {f1:.2f} \\\\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)
    with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
        f.write(latex)

    print(f"Wrote LaTeX table to: {OUTPUT_TEX}\n")
    print(latex)


if __name__ == "__main__":
    main()