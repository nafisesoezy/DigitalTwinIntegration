#!/usr/bin/env python3
"""
Per-viewpoint detection performance (Accuracy/Precision/Recall/F1) for:
  - Rule-based baseline (Static)
  - All LLM models found in the unified CSV (LLM-result-*)

Input (one unified CSV): allLLM_match_report.csv

Expected columns (minimum):
  group, bottleneck, field, pattern, result, ab_kind,
  and one or more columns like: LLM-result-<model_id>

Ground-truth vs prediction (recommended / matches your paper text):
  y_true = INTEGRATED.result        (realized integration outcome, ground truth)
  y_pred = INTENDED.<method column> (predicted outcome from intended setting)

Binary metrics:
  issue = {mismatch, missing, gap} ; not-issue = {match}

Output:
  - prints debug stats
  - prints one LaTeX table where each viewpoint contributes multiple rows:
      <viewpoint>-Rule-based (Static)
      <viewpoint>-<LLM model 1>
      <viewpoint>-<LLM model 2>
      ...

Notes:
  - Viewpoint assignment is done by mapping the 'field' column to your table's
    metadata field names. Rows whose field is not recognized are ignored.
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

import pandas as pd

INPUT_CSV = "allLLM_match_report.csv"

# Labels considered "issue" for binary metrics
ISSUE_LABELS = {"mismatch", "missing", "gap"}

# ab_kind values (normalized)
K_INTENDED = "intended"
K_INTEGRATED = "integrated"

# -----------------------------
# Viewpoint → fields mapping (MATCHES YOUR CSV FIELD NAMES)
# -----------------------------
VIEWPOINT_FIELDS_RAW: Dict[str, List[str]] = {
    "domain": [
        "Title",
        "Model Version",
        "Description",
        "Keywords",
        "Model Type",
        "Scope",
        "Purpose & Pattern",                 # <-- combined in your file
        "Assumptions",
        "Links to Publications & Reports",   # <-- & variant
        "Authors’ Unique Identifier",        # <-- curly apostrophe
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


def _canon(s: str) -> str:
    """Canonicalize a field name for robust matching."""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    # keep basic punctuation because your 'field' column sometimes contains dots/parens/slashes;
    # but for viewpoint mapping we mainly match on plain metadata names.
    s = s.replace("’", "'")
    return s

VIEWPOINT_FIELDS: Dict[str, set] = {
    vp: {_canon(x) for x in xs}
    for vp, xs in VIEWPOINT_FIELDS_RAW.items()
}

def viewpoint_of_field(field_value: str) -> Optional[str]:
    """
    Returns viewpoint key (domain/information/...) if 'field' matches one of the metadata field names.
    Otherwise None.

    If your CSV uses slightly different labels, add aliases here.
    """
    f = _canon(field_value)

    # --- simple aliases that often appear in your mismatch reports ---
    aliases = {
        "acknowledgment protocols": "acknowledgment protocols",
        "acknowledgment_protocols": "acknowledgment protocols",
        "communication mechanism": "data synchronization",  # (if you prefer, add a dedicated mapping)
        "distribution version": "distribution version",
        "programming language": "programming language",
        "license": "license",
        "landing page": "landing page",
        "interface signature": "interface signature",
        "error handling": "error handling",
        "integration pattern": "integration pattern",
        "pattern": "pattern",
        "assumptions": "assumptions",
        "inputs": "input datasets",
        "input": "input datasets",
        "outputs": "output",
        "output": "output",
    }
    f2 = _canon(aliases.get(f, f))

    for vp, fieldset in VIEWPOINT_FIELDS.items():
        if f2 in fieldset:
            return vp

    return None


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
# Load / split
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

    # assign viewpoint from the 'field' column
    if "field" not in df.columns:
        raise KeyError("Input CSV must contain column 'field' for viewpoint assignment.")

    df_intended["_viewpoint"] = df_intended["field"].map(viewpoint_of_field)
    df_integrated["_viewpoint"] = df_integrated["field"].map(viewpoint_of_field)

    return df_intended, df_integrated


# -----------------------------
# Evaluation (GT = integrated, Pred = intended)
# -----------------------------
def evaluate_pair(
    df_intended: pd.DataFrame,
    df_integrated: pd.DataFrame,
    pred_col_intended: str,
    gt_col_integrated: str,
    viewpoint: str,
) -> Tuple[Confusion, Dict[str, int]]:
    """
    Ground-truth (y_true): INTEGRATED.<gt_col_integrated>
    Prediction  (y_pred): INTENDED.<pred_col_intended>

    Only evaluates rows whose viewpoint == viewpoint and which align by _key.
    """
    di = df_intended[df_intended["_viewpoint"] == viewpoint].copy()
    dg = df_integrated[df_integrated["_viewpoint"] == viewpoint].copy()

    df_pred = di[["_key"]].copy()
    df_true = dg[["_key"]].copy()

    df_pred["_pred"] = require_col(di, pred_col_intended).map(norm_label)
    df_true["_true"] = require_col(dg, gt_col_integrated).map(norm_label)

    merged = df_pred.merge(df_true, on="_key", how="outer", indicator=True)

    debug = {
        "viewpoint": viewpoint,
        "rows_intended_total": int(len(di)),
        "rows_integrated_total": int(len(dg)),
        "rows_aligned": int((merged["_merge"] == "both").sum()),
        "rows_only_intended": int((merged["_merge"] == "left_only").sum()),
        "rows_only_integrated": int((merged["_merge"] == "right_only").sum()),
    }

    conf = Confusion()
    aligned = merged[merged["_merge"] == "both"]

    for _, r in aligned.iterrows():
        conf.add(
            y_true=to_binary_issue(r["_true"]),
            y_pred=to_binary_issue(r["_pred"]),
        )

    return conf, debug


# -----------------------------
# Methods discovery
# -----------------------------
def discover_llm_result_cols(df: pd.DataFrame) -> List[str]:
    """Return all LLM-result-* columns."""
    cols = [c for c in df.columns if c.startswith("LLM-result-")]
    # stable / nice ordering
    cols.sort()
    return cols


def pretty_model_name_from_col(col: str) -> str:
    """
    Convert column like:
      LLM-result-openai/gpt-oss-120b
    to:
      OpenAI GPT-OSS-120B
    """
    s = col.replace("LLM-result-", "").strip()

    # a few nice prettifications
    s_lower = s.lower()
    if "openai/" in s_lower and "gpt-oss-120b" in s_lower:
        return "OpenAI GPT-OSS-120B"
    if "llama" in s_lower and "70b" in s_lower:
        return "Llama 3.3 70B Instruct (AWQ)" if "awq" in s_lower else "Llama 3.3 70B Instruct"
    if "mistral" in s_lower and "24b" in s_lower:
        return "Mistral Small 3.2 24B Instruct"
    if "default-text-large" in s_lower:
        return "default-text-large"

    # fallback: keep id but make it readable
    return s.replace("_", " ")


# -----------------------------
# LaTeX table
# -----------------------------
def latex_table_per_viewpoint(rows: List[Tuple[str, Confusion]]) -> str:
    caption = "Detection performance comparison between the rule-based baseline and the LLM-assisted detector (per viewpoint)."
    label = "tab:detection_performance_by_viewpoint"

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

    current_vp = None
    for name, conf in rows:
        vp = name.split("-", 1)[0] if "-" in name else ""
        if current_vp is None:
            current_vp = vp
        elif vp != current_vp:
            out.append(r"\hline")
            current_vp = vp

        out.append(
            f"{name} & {fmt(conf.accuracy())} & {fmt(conf.precision())} & {fmt(conf.recall())} & {fmt(conf.f1())} \\\\"
        )

    out.append(r"\hline")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")
    return "\n".join(out)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Cannot find input file: {INPUT_CSV}")

    df_intended, df_integrated = load_and_split(INPUT_CSV)
    df_all = pd.concat([df_intended, df_integrated], ignore_index=True)

    llm_cols = discover_llm_result_cols(df_all)
    if not llm_cols:
        raise RuntimeError("No columns found matching 'LLM-result-*' in the unified CSV.")

    viewpoints_in_data = sorted(
        {vp for vp in df_all.get("_viewpoint", pd.Series([])).dropna().unique().tolist()}
    )

    # If nothing matched your field list, fail loudly (so you notice mapping problems)
    if not viewpoints_in_data:
        raise RuntimeError(
            "No rows could be assigned to any viewpoint.\n"
            "This usually means your CSV 'field' values do not match the metadata field names.\n"
            "Add aliases in viewpoint_of_field() or adapt the mapping."
        )

    print(f"Loaded unified file: {INPUT_CSV}")
    print(f"INTENDED rows:   {len(df_intended)}")
    print(f"INTEGRATED rows: {len(df_integrated)}")
    print(f"Discovered LLM models: {len(llm_cols)}")
    print(f"Viewpoints found in data: {viewpoints_in_data}\n")

    # Build methods:
    # Baseline: intended.result vs integrated.result (static)
    # LLM: intended.LLM-result-<model> vs integrated.result
    methods: List[Tuple[str, str, str]] = []
    methods.append(("Rule-based (Static)", "result", "result"))
    for col in llm_cols:
        methods.append((pretty_model_name_from_col(col), col, "result"))

    results: List[Tuple[str, Confusion]] = []

    for vp in viewpoints_in_data:
        for method_name, pred_col, gt_col in methods:
            conf, debug = evaluate_pair(
                df_intended=df_intended,
                df_integrated=df_integrated,
                pred_col_intended=pred_col,
                gt_col_integrated=gt_col,
                viewpoint=vp,
            )

            # keep the debug prints, but compact
            print(f"=== {vp.upper()} | {method_name} ===")
            print(
                f"aligned={debug['rows_aligned']} "
                f"only_intended={debug['rows_only_intended']} "
                f"only_integrated={debug['rows_only_integrated']} "
                f"(intended={debug['rows_intended_total']}, integrated={debug['rows_integrated_total']})"
            )
            print(f"TP={conf.tp} FP={conf.fp} FN={conf.fn} TN={conf.tn}\n")

            row_name = f"{vp}-{method_name}"
            results.append((row_name, conf))

    print(latex_table_per_viewpoint(results))


if __name__ == "__main__":
    main()