#!/usr/bin/env python3
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import pandas as pd

REPORT_DIR = "mismatch_solutions"

# ✅ FIX: ground truth column is "LLM-Result" (with dash), not "LLM_result"
METHODS = [
    ("Rule-based (Static)", "result", "LLM-Result"),     # pred from INTENDED.result vs GT from INTEGRATED.LLM-Result
    ("LLM-assisted",        "LLM-Result", "LLM-Result"), # pred from INTENDED.LLM-Result vs GT from INTEGRATED.LLM-Result
]

ISSUE_LABELS = {"mismatch", "missing", "gap"}

PATTERN_INTENDED = "annotated_mismatch_report_*_INTENDED*.csv"
PATTERN_INTEGRATED = "annotated_mismatch_report_*_INTEGRATED*.csv"


def norm_label(x) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower()
    if s in {"match", "matched"}:
        return "match"
    if s in {"mismatch", "real_mismatch"}:
        return "mismatch"
    if s in {"missing", "miss"}:
        return "missing"
    if s in {"gap", "metadata gap", "metadata_gap"}:
        return "gap"
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


KEY_CANDIDATES = [
    ("group", "pattern", "field", "bottleneck"),
    ("pattern", "field", "bottleneck"),
    ("field", "bottleneck"),
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


def extract_config_id(path: str) -> str:
    base = os.path.basename(path)
    prefix = "annotated_mismatch_report_"
    if not base.startswith(prefix):
        return os.path.splitext(base)[0]
    tail = base[len(prefix):]
    tail = tail.replace("_INTENDED.csv", "").replace("_INTEGRATED.csv", "")
    tail = tail.split("_INTENDED")[0].split("_INTEGRATED")[0]
    return tail.strip("_")


def list_pairs(report_dir: str) -> List[Tuple[str, str, str]]:
    intended_paths = sorted(glob.glob(os.path.join(report_dir, PATTERN_INTENDED)))
    integrated_paths = sorted(glob.glob(os.path.join(report_dir, PATTERN_INTEGRATED)))
    intended_map: Dict[str, str] = {extract_config_id(p): p for p in intended_paths}
    integrated_map: Dict[str, str] = {extract_config_id(p): p for p in integrated_paths}
    common = sorted(set(intended_map) & set(integrated_map))
    if not common:
        raise RuntimeError(f"No pairs found in '{report_dir}'.")
    return [(cid, intended_map[cid], integrated_map[cid]) for cid in common]


def evaluate_method(pred_col_intended: str, gt_col_integrated: str) -> Tuple[Confusion, Dict[str, int]]:
    pairs = list_pairs(REPORT_DIR)

    conf = Confusion()
    debug = {
        "configs": 0,
        "rows_intended_total": 0,
        "rows_integrated_total": 0,
        "rows_aligned": 0,
        "rows_unaligned_intended": 0,
        "rows_unaligned_integrated": 0,
    }

    for _, p_int, p_gt in pairs:
        df_int = pd.read_csv(p_int)
        df_gt = pd.read_csv(p_gt)

        debug["configs"] += 1
        debug["rows_intended_total"] += len(df_int)
        debug["rows_integrated_total"] += len(df_gt)

        key_int, _ = build_key(df_int)
        key_gt, _ = build_key(df_gt)

        df_int = df_int.copy()
        df_gt = df_gt.copy()
        df_int["_key"] = key_int
        df_gt["_key"] = key_gt

        df_int["_pred"] = require_col(df_int, pred_col_intended).map(norm_label)
        df_gt["_gt"] = require_col(df_gt, gt_col_integrated).map(norm_label)

        merged = df_int[["_key", "_pred"]].merge(df_gt[["_key", "_gt"]], on="_key", how="outer", indicator=True)

        aligned = merged[merged["_merge"] == "both"].copy()
        debug["rows_aligned"] += len(aligned)
        debug["rows_unaligned_intended"] += int((merged["_merge"] == "left_only").sum())
        debug["rows_unaligned_integrated"] += int((merged["_merge"] == "right_only").sum())

        for _, r in aligned.iterrows():
            conf.add(y_true=to_binary_issue(r["_gt"]), y_pred=to_binary_issue(r["_pred"]))

    return conf, debug


def fmt(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}"


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
        out.append(f"{name} & {fmt(conf.accuracy())} & {fmt(conf.precision())} & {fmt(conf.recall())} & {fmt(conf.f1())} \\\\")
    out.append(r"\hline")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")
    return "\n".join(out)


def main() -> None:
    print(f"📁 REPORT_DIR = {REPORT_DIR}")
    pairs = list_pairs(REPORT_DIR)
    print(f"✅ Found {len(pairs)} paired configurations.")

    results: List[Tuple[str, Confusion]] = []
    for method_name, pred_col, gt_col in METHODS:
        conf, debug = evaluate_method(pred_col_intended=pred_col, gt_col_integrated=gt_col)

        print(f"\n=== {method_name} ===")
        for k, v in debug.items():
            print(f"{k}: {v}")
        print(f"TP={conf.tp} FP={conf.fp} FN={conf.fn} TN={conf.tn}")

        results.append((method_name, conf))

    print("\n" + latex_table(results) + "\n")


if __name__ == "__main__":
    main()