#!/usr/bin/env python3
"""
Evaluate agreement/detection performance between INTENDED vs INTEGRATED rows
from ONE unified CSV: allLLM_match_report.csv

Expected columns (at least):
  group, bottleneck, field, pattern, required_check, A_value, B_value, AB_value,
  detail, result, ab_kind,
  LLM-result-openai/gpt-oss-120b, LLM-suggestion-openai/gpt-oss-120b,
  LLM-result-mistralai/Mistral-Small-3.2-24B-Instruct-2506, LLM-suggestion-mistralai/Mistral-Small-3.2-24B-Instruct-2506,
  LLM-result-default-text-large, LLM-suggestion-default-text-large

What it computes (binary: issue vs not-issue; issue ∈ {mismatch, missing, gap}):

Rule-based (Static):
  y_true = INTENDED.result
  y_pred = INTEGRATED.LLM-result-openai/gpt-oss-120b

A:
  y_true = INTENDED.LLM-result-openai/gpt-oss-120b
  y_pred = INTEGRATED.LLM-result-openai/gpt-oss-120b

B:
  y_true = INTENDED.LLM-result-default-text-large
  y_pred = INTEGRATED.LLM-result-openai/gpt-oss-120b

C:
  y_true = INTENDED.LLM-suggestion-mistralai/Mistral-Small-3.2-24B-Instruct-2506
  y_pred = INTEGRATED.LLM-result-openai/gpt-oss-120b

Outputs:
  - Prints debug alignment stats
  - Prints LaTeX table for Accuracy/Precision/Recall/F1
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

INPUT_CSV = "allLLM_match_report.csv"

# Labels considered "issue" for binary metrics
ISSUE_LABELS = {"mismatch", "missing", "gap"}

# ab_kind values (normalized)
K_INTENDED = "intended"
K_INTEGRATED = "integrated"


# -----------------------------
# Normalization / binarization
# -----------------------------
def norm_label(x) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower()

    # canonical mapping
    if s in {"match", "matched", "ok", "yes"}:
        return "match"
    if s in {"mismatch", "real_mismatch", "real mismatch"}:
        return "mismatch"
    if s in {"missing", "miss"}:
        return "missing"
    if s in {"gap", "metadata gap", "metadata_gap", "metadata-gap"}:
        return "gap"

    # fuzzy contains
    if "mismatch" in s:
        return "mismatch"
    if "missing" in s:
        return "missing"
    if "gap" in s:
        return "gap"
    if "match" in s:
        return "match"
    return s


def to_binary_issue(label: str) -> int:
    return 1 if norm_label(label) in ISSUE_LABELS else 0


def require_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found. Available columns: {list(df.columns)}")
    return df[col]


# -----------------------------
# Alignment key (robust)
# -----------------------------
KEY_CANDIDATES = [
    ("group", "field", "bottleneck", "pattern"),
    ("group", "field", "bottleneck"),
    ("group", "field", "pattern"),
    ("group", "field"),
    ("field", "bottleneck", "pattern"),
    ("field", "bottleneck"),
    ("field", "pattern"),
    ("field",),
]


def build_key(df: pd.DataFrame) -> Tuple[pd.Series, Tuple[str, ...]]:
    cols_present = set(df.columns)
    for cols in KEY_CANDIDATES:
        if all(c in cols_present for c in cols):
            key = (
                df[list(cols)]
                .astype(str)
                .apply(lambda s: s.str.strip().str.lower())
                .agg("||".join, axis=1)
            )
            return key, cols
    raise KeyError(
        f"Cannot build alignment key. Need one of {KEY_CANDIDATES}. Found columns: {list(df.columns)}"
    )


# -----------------------------
# Confusion / metrics
# -----------------------------
@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def add(self, y_true: int, y_pred: int) -> None:
        if y_true == 1 and y_pred == 1:
            self.tp += 1
        elif y_true == 0 and y_pred == 1:
            self.fp += 1
        elif y_true == 1 and y_pred == 0:
            self.fn += 1
        else:
            self.tn += 1

    def accuracy(self) -> float:
        n = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / n if n else 0.0

    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    def f1(self) -> float:
        p = self.precision()
        r = self.recall()
        return (2 * p * r / (p + r)) if (p + r) else 0.0


def fmt(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}"


# -----------------------------
# Core evaluation
# -----------------------------
def load_and_split(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)

    if "ab_kind" not in df.columns:
        raise KeyError("Input CSV must contain column 'ab_kind'.")

    ab = df["ab_kind"].astype(str).str.strip().str.lower()
    df = df.copy()
    df["_ab_kind_norm"] = ab

    df_intended = df[df["_ab_kind_norm"] == K_INTENDED].copy()
    df_integrated = df[df["_ab_kind_norm"] == K_INTEGRATED].copy()

    if df_intended.empty or df_integrated.empty:
        raise RuntimeError(
            "Could not split INTENDED vs INTEGRATED rows.\n"
            f"Found intended={len(df_intended)} integrated={len(df_integrated)}.\n"
            "Check ab_kind values (expected 'INTENDED'/'INTEGRATED')."
        )

    key_i, _ = build_key(df_intended)
    key_g, _ = build_key(df_integrated)

    df_intended["_key"] = key_i
    df_integrated["_key"] = key_g

    return df_intended, df_integrated


def evaluate_pair(
    df_intended: pd.DataFrame,
    df_integrated: pd.DataFrame,
    intended_col: str,
    integrated_col: str,
) -> Tuple[Confusion, Dict[str, int]]:
    """
    INTENDED is treated as ground truth (y_true).
    INTEGRATED is treated as prediction (y_pred).
    Metrics are computed on binary labels: issue vs not-issue.
    """

    df_true = df_intended[["_key"]].copy()
    df_pred = df_integrated[["_key"]].copy()

    df_true["_true"] = require_col(df_intended, intended_col).map(norm_label)
    df_pred["_pred"] = require_col(df_integrated, integrated_col).map(norm_label)

    merged = df_pred.merge(df_true, on="_key", how="outer", indicator=True)

    debug = {
        "rows_intended_total": int(len(df_intended)),
        "rows_integrated_total": int(len(df_integrated)),
        "rows_aligned": int((merged["_merge"] == "both").sum()),
        "rows_only_integrated": int((merged["_merge"] == "left_only").sum()),
        "rows_only_intended": int((merged["_merge"] == "right_only").sum()),
    }

    conf = Confusion()
    aligned = merged[merged["_merge"] == "both"]

    for _, r in aligned.iterrows():
        conf.add(
            y_true=to_binary_issue(r["_true"]),
            y_pred=to_binary_issue(r["_pred"]),
        )

    return conf, debug


def latex_table(rows: List[Tuple[str, Confusion]]) -> str:
    caption = "Detection performance comparison between the rule-based baseline and the LLM-assisted detector."
    label = "tab:detection_performance"

    out = []
    out.append(r"\begin{table}[h]")
    out.append(r"\centering")
    out.append(rf"\caption{{{caption}}}")
    out.append(rf"\label{{{label}}}")
    out.append(r"\footnotesize")
    out.append(r"\begin{tabular}{lcccc}")
    out.append(r"\hline")
    out.append(r"\textbf{Method} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-score} \\")
    out.append(r"\hline")
    for name, conf in rows:
        out.append(
            f"{name} & {fmt(conf.accuracy())} & {fmt(conf.precision())} & {fmt(conf.recall())} & {fmt(conf.f1())} \\\\"
        )
    out.append(r"\hline")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")
    return "\n".join(out)


def main() -> None:


    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Cannot find input file: {INPUT_CSV}")

    df_intended, df_integrated = load_and_split(INPUT_CSV)

    # methods are (name, intended_col_as_GT, integrated_col_as_pred)
    methods = [
        # ✅ corrected baseline per your definition:
        ("Rule-based (Static)",
         "result",
         "LLM-result-openai/gpt-oss-120b"),

        ("A(LLM-result-openai/gpt-oss-120b)",
         "LLM-result-openai/gpt-oss-120b",
         "LLM-result-openai/gpt-oss-120b"),

        ("B(LLM-result-default-text-large)",
         "LLM-result-default-text-large",
         "LLM-result-openai/gpt-oss-120b"),

        ("C(LLM-suggestion-mistralai/Mistral-Small-3.2-24B-Instruct-2506)",
         "LLM-suggestion-mistralai/Mistral-Small-3.2-24B-Instruct-2506",
         "LLM-result-openai/gpt-oss-120b"),
    ]

    print(f"Loaded unified file: {INPUT_CSV}")
    print(f"INTENDED rows:   {len(df_intended)}")
    print(f"INTEGRATED rows: {len(df_integrated)}\n")

    results: List[Tuple[str, Confusion]] = []

    for name, intended_col, integrated_col in methods:
        conf, debug = evaluate_pair(
            df_intended=df_intended,
            df_integrated=df_integrated,
            intended_col=intended_col,
            integrated_col=integrated_col,
        )

        print(f"=== {name} ===")
        for k, v in debug.items():
            print(f"{k}: {v}")
        print(f"TP={conf.tp} FP={conf.fp} FN={conf.fn} TN={conf.tn}\n")

        results.append((name, conf))

    print(latex_table(results))


if __name__ == "__main__":
    main()