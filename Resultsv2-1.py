#!/usr/bin/env python3
"""
Create a LaTeX detection-performance table from allLLM_match_report_groundTruth.csv

Metrics are computed by comparing each method's predicted label to `groundTruth`.

Assumptions:
- groundTruth and prediction columns contain categorical labels like:
  Match / Mismatch / Missing (case-insensitive)
- We evaluate *Mismatch detection* as the positive class:
  positive := "mismatch"
  negative := anything else ("match" or "missing" etc.)
- Rows with missing groundTruth are dropped.
"""

import os
import re
import pandas as pd


INPUT_CSV  = os.environ.get("GT_REPORT_PATH", "allLLM_match_report_groundTruth.csv")
OUTPUT_TEX = os.environ.get("DETECTION_TABLE_TEX", "detection_performance.tex")

GT_COL = "groundTruth"

METHODS = [
    ("Rule-based (Static)", "result"),
    ("OpenAI GPT-OSS-120B", "LLM-result-openai/gpt-oss-120b"),
    ("Mistral Small 3.2 24B Instruct", "LLM-result-mistralai/Mistral-Small-3.2-24B-Instruct-2506"),
    ("default-text-large", "LLM-result-default-text-large"),
]

POS_LABEL = "mismatch"   # positive class keyword (case-insensitive)


def norm_label(x: object) -> str:
    """Normalize labels to: 'mismatch', 'match', 'missing', or 'other'."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "missing"
    s = str(x).strip().lower()
    if not s or s == "nan":
        return "missing"
    # common variants
    if "mismatch" in s:
        return "mismatch"
    if s == "match" or " match" in s or s.startswith("match"):
        return "match"
    if "missing" in s:
        return "missing"
    # anything else (e.g., "gap", "not applicable", etc.)
    return s


def to_binary(label: str) -> int:
    """Map normalized label to binary class (1=positive mismatch, 0=non-mismatch)."""
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
    """Escape minimal LaTeX specials (for method names/caption if needed)."""
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


def main():
    df = pd.read_csv(INPUT_CSV)

    if GT_COL not in df.columns:
        raise KeyError(f"Missing required column '{GT_COL}'. Available: {list(df.columns)}")

    # Drop rows with missing ground truth (no evaluable target)
    df["_gt_norm"] = df[GT_COL].map(norm_label)
    df_eval = df[df["_gt_norm"] != "missing"].copy()

    if df_eval.empty:
        raise ValueError("No evaluable rows after dropping missing groundTruth.")

    y_true = df_eval["_gt_norm"].map(to_binary).tolist()

    rows = []
    for method_name, col in METHODS:
        if col not in df_eval.columns:
            raise KeyError(f"Missing prediction column '{col}' for method '{method_name}'. "
                           f"Available: {list(df_eval.columns)}")

        df_eval["_pred_norm"] = df_eval[col].map(norm_label)
        y_pred = df_eval["_pred_norm"].map(to_binary).tolist()

        acc, prec, rec, f1 = compute_metrics(y_true, y_pred)
        rows.append((method_name, acc, prec, rec, f1))

    # Build LaTeX table (values to 2 decimals like your example)
    latex_lines = []
    latex_lines.append(r"\begin{table}[h]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Detection performance comparison between the rule-based baseline and the LLM-assisted detector.}")
    latex_lines.append(r"\label{tab:detection_performance}")
    latex_lines.append(r"\footnotesize")
    latex_lines.append(r"\begin{tabular}{lcccc}")
    latex_lines.append(r"\hline")
    latex_lines.append(r"\textbf{Method} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-score} \\")
    latex_lines.append(r"\hline")

    for method_name, acc, prec, rec, f1 in rows:
        latex_lines.append(
            f"{latex_escape(method_name)} & {acc:.2f} & {prec:.2f} & {rec:.2f} & {f1:.2f} \\\\"
        )

    latex_lines.append(r"\hline")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")
    latex = "\n".join(latex_lines)

    with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
        f.write(latex)

    print(f"Wrote LaTeX table to: {OUTPUT_TEX}")
    print("\n--- LaTeX preview ---\n")
    print(latex)


if __name__ == "__main__":
    main()