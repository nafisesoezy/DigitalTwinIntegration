#!/usr/bin/env python3
"""
Resultsv7-fine-grained-errors.py

Table `fine_grained_errors`: TP/FP/FN/Gap counts (not just F1) for the
deterministic rule-based baseline vs. one LLM-assisted method, broken down
by RM-ODP viewpoint, integration pattern, and mismatch type -- showing
WHERE detection errors occur and WHERE the hybrid procedure improves over
the baseline, rather than only aggregate metrics.

Ground truth: the corrected, deterministic ground truth (merge2.py --
`result` applied to the realized/INTEGRATED model, see README.md §8), NOT
the earlier circular LLM-derived one. On the real, already-collected
dataset this gives N_M=97 reference mismatches, N_C=280 reference-
compatible cases, 59 Gap (insufficient-evidence) cases -- confirmed against
allLLM_match_report_groundTruth.csv, NOT the 29/348 split from the invalid
ground truth used in an earlier draft.

Definitions (fixed, computed directly from the raw prediction/ground-truth
strings -- never derived from rounded metrics):
  TP  = groundTruth==Mismatch AND prediction==Mismatch
  FN  = groundTruth==Mismatch AND prediction==Match
  FP  = groundTruth==Match    AND prediction==Mismatch
  Gap = prediction is Missing/Gap, counted regardless of ground-truth class
        (both Gap-when-Mismatch and Gap-when-Match contribute to this one
        column, matching a single "Gap" column per method). Gap is NEVER
        folded into FN or FP: it is an abstention (insufficient evidence),
        not a compatibility judgment -- see README.md §8.
  (TN is not tabulated; it is implied by N_C - FP - Gap-when-Match.)

Mismatch-type grouping (raw `bottleneck` values -> display label; see
constraint_templates.py for the corresponding ConstraintTemplate names):
  Semantic                     <- Semantic Mismatch
  Schema                       <- Data Schema Mismatch
  Dimensionality                <- Dimensionality Mismatch
  Data synchronization           <- Data Synchronization
  Execution constraints           <- Execution Constraint Mismatch,
                                      Latency Expectation Mismatch
  Error handling                   <- Error Handling Mismatch
  Software environment              <- Software Environment Mismatch,
                                        Hardware Resource Mismatch,
                                        Programming Language Incompatibility,
                                        Distribution Version Mismatch
  License                            <- License Incompatibility
  Spatial/Temporal Alignment          <- Temporal Resolution Mismatch,
                                          Temporal Coverage Mismatch,
                                          Spatial Resolution Mismatch,
                                          Spatial Coverage Mismatch
("Execution constraints" additionally absorbs Latency Expectation Mismatch
so the nine rows above sum to exactly 97 -- Latency has no natural row of
its own in the requested nine and is a sibling Runtime Coordination concern
to Execution Constraint Compatibility in constraint_templates.py.)

IMPORTANT SCOPE NOTE: this grouping is applied over ALL rows with a
reference groundTruth (the same row set Table `detection_performance`
scores), NOT restricted to constraint_templates.in_paper_appendix==True
rows. So "Semantic" also includes the legacy Title/Description/Keywords/
Model Type/Model Version auxiliary fields (grouped under the same
"Semantic Mismatch" bottleneck as the paper's Purpose/Scope/Assumption
templates), and "Spatial/Temporal Alignment" also includes the legacy
Coverage fields alongside the paper's Resolution templates. This keeps
this table's Overall row identical to Table `detection_performance`'s N
(97/280/59). A stricter Appendix-only variant would show smaller counts
for both rows -- ask if that variant is wanted instead.

Run:
  python Resultsv7-fine-grained-errors.py

Optional env vars:
  GT_REPORT_PATH   (default: allLLM_match_report_groundTruth.csv)
  ERROR_TABLE_TEX  (default: fine_grained_errors.tex)
  COMPARE_MODEL    (default: LLM-result-openai/gpt-oss-120b)
  COMPARE_LABEL    (default: GPT-OSS-120B)
"""

import os
from typing import Dict, List, Tuple

import pandas as pd

from constraint_templates import classify

INPUT_CSV = os.environ.get("GT_REPORT_PATH", "allLLM_match_report_groundTruth.csv")
OUTPUT_TEX = os.environ.get("ERROR_TABLE_TEX", "fine_grained_errors.tex")
COMPARE_MODEL_COL = os.environ.get("COMPARE_MODEL", "LLM-result-openai/gpt-oss-120b")
COMPARE_MODEL_LABEL = os.environ.get("COMPARE_LABEL", "GPT-OSS-120B")

BASELINE_COL = "result"
BASELINE_LABEL = "Deterministic baseline"
GT_COL = "groundTruth"
VIEWPOINT_COL = "rm_odp_viewpoint"
PATTERN_COL = "pattern"

VIEWPOINTS = ["Domain", "Information", "Computational", "Engineering", "Technology"]
PATTERNS = ["One-Way", "Loose", "Shared", "Integrated", "Embedded"]

TYPE_GROUPS: Dict[str, List[str]] = {
    "Semantic": ["Semantic Mismatch"],
    "Schema": ["Data Schema Mismatch"],
    "Dimensionality": ["Dimensionality Mismatch"],
    "Data synchronization": ["Data Synchronization"],
    "Execution constraints": ["Execution Constraint Mismatch", "Latency Expectation Mismatch"],
    "Error handling": ["Error Handling Mismatch"],
    "Software environment": ["Software Environment Mismatch", "Hardware Resource Mismatch",
                              "Programming Language Incompatibility", "Distribution Version Mismatch"],
    "License": ["License Incompatibility"],
    "Spatial/Temporal Alignment": ["Temporal Resolution Mismatch", "Temporal Coverage Mismatch",
                                    "Spatial Resolution Mismatch", "Spatial Coverage Mismatch"],
}


def norm3(x: object) -> str:
    """Match / Mismatch / Gap -- used for BOTH ground truth and predictions
    (predictions additionally fold "Missing"/"Error"/anything unrecognized
    into "gap", since an abstention or a failed call is not a compatibility
    judgment either)."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "gap"
    s = str(x).strip().lower()
    if "mismatch" in s:
        return "mismatch"
    if s == "match" or s.startswith("match") or " match" in s:
        return "match"
    return "gap"  # "missing", "gap", "error", anything else


def counts_for(df_sub: pd.DataFrame, pred_col: str) -> Dict[str, int]:
    gt = df_sub["_gt"]
    pred = df_sub[pred_col].map(norm3)
    n_m = int((gt == "mismatch").sum())
    n_c = int((gt == "match").sum())
    tp = int(((gt == "mismatch") & (pred == "mismatch")).sum())
    fn = int(((gt == "mismatch") & (pred == "match")).sum())
    fp = int(((gt == "match") & (pred == "mismatch")).sum())
    gap = int((pred == "gap").sum())
    return {"N_M": n_m, "N_C": n_c, "TP": tp, "FP": fp, "FN": fn, "Gap": gap}


def build_type_column(df: pd.DataFrame) -> pd.Series:
    raw_to_label: Dict[str, str] = {}
    for label, raws in TYPE_GROUPS.items():
        for r in raws:
            raw_to_label[r] = label
    return df["bottleneck"].map(raw_to_label)


def latex_escape(s: str) -> str:
    return (s.replace("\\", r"\textbackslash{}").replace("&", r"\&")
             .replace("%", r"\%").replace("_", r"\_")
             .replace("#", r"\#"))


def dashfmt(v: int) -> str:
    return str(v)


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    for c in (GT_COL, BASELINE_COL, COMPARE_MODEL_COL, PATTERN_COL, "bottleneck", "field"):
        if c not in df.columns:
            raise KeyError(f"Missing column '{c}'. Available: {list(df.columns)}")

    tmpl = df.apply(lambda r: classify(r["bottleneck"], r["field"]), axis=1)
    df[VIEWPOINT_COL] = tmpl.apply(lambda t: t.viewpoint if t else None)
    df["_mismatch_type"] = build_type_column(df)
    df["_gt"] = df[GT_COL].map(norm3)

    unmapped_type_mismatches = int(
        ((df["_gt"] == "mismatch") & (df["_mismatch_type"].isna())).sum()
    )

    subgroups: List[Tuple[str, str, pd.DataFrame]] = [("Overall", "All constraints", df)]
    for vp in VIEWPOINTS:
        subgroups.append(("Viewpoint", vp, df[df[VIEWPOINT_COL] == vp]))
    for pat in PATTERNS:
        subgroups.append(("Pattern", pat, df[df[PATTERN_COL] == pat]))
    for t in TYPE_GROUPS:
        subgroups.append(("Mismatch type", t, df[df["_mismatch_type"] == t]))

    results = []
    for analysis, name, sub in subgroups:
        base = counts_for(sub, BASELINE_COL)
        comp = counts_for(sub, COMPARE_MODEL_COL)
        results.append((analysis, name, base, comp))

    # --- sanity checks against the Overall row (printed, not enforced) ---
    overall_nm = results[0][2]["N_M"]
    vp_nm_sum = sum(r[2]["N_M"] for r in results if r[0] == "Viewpoint")
    pat_nm_sum = sum(r[2]["N_M"] for r in results if r[0] == "Pattern")
    type_nm_sum = sum(r[2]["N_M"] for r in results if r[0] == "Mismatch type")

    # --- console table ---
    hdr = f"{'Analysis':<14} {'Subgroup':<28} {'N_M':>4} {'N_C':>4} | " \
          f"{'baseTP':>6} {'baseFP':>6} {'baseFN':>6} {'baseGap':>7} | " \
          f"{'cmpTP':>5} {'cmpFP':>5} {'cmpFN':>5} {'cmpGap':>6}"
    print(hdr)
    print("-" * len(hdr))
    for analysis, name, base, comp in results:
        print(f"{analysis:<14} {name:<28} {base['N_M']:>4} {base['N_C']:>4} | "
              f"{base['TP']:>6} {base['FP']:>6} {base['FN']:>6} {base['Gap']:>7} | "
              f"{comp['TP']:>5} {comp['FP']:>5} {comp['FN']:>5} {comp['Gap']:>6}")

    print()
    print(f"Sanity check -- Overall N_M = {overall_nm}")
    print(f"  Sum of Viewpoint N_M     = {vp_nm_sum}  ({'OK' if vp_nm_sum == overall_nm else 'MISMATCH -- some rows have no mapped viewpoint'})")
    print(f"  Sum of Pattern N_M       = {pat_nm_sum}  ({'OK' if pat_nm_sum == overall_nm else 'MISMATCH -- some rows have no pattern'})")
    print(f"  Sum of Mismatch-type N_M = {type_nm_sum}  ({'OK' if type_nm_sum == overall_nm else 'MISMATCH -- unmapped bottleneck category exists'})")
    if unmapped_type_mismatches:
        print(f"  [warn] {unmapped_type_mismatches} reference-mismatch rows have a bottleneck "
              f"not covered by TYPE_GROUPS -- extend the mapping.")

    # --- LaTeX ---
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Fine-grained error analysis for the deterministic baseline and "
        rf"{latex_escape(COMPARE_MODEL_LABEL)} hybrid variant. $N_M$ and $N_C$ denote the "
        r"numbers of reference mismatch and compatible cases, respectively. TP, FP, and FN "
        r"denote true detections, false detections, and missed mismatches. Gap reports cases "
        r"for which the available evidence was insufficient to determine compatibility.}"
    )
    lines.append(r"\label{tab:fine_grained_errors}")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(r"\begin{tabular}{llrr|rrrr|rrrr}")
    lines.append(r"\hline")
    lines.append(r" & & \multicolumn{2}{c|}{\textbf{Reference}} & "
                 r"\multicolumn{4}{c|}{\textbf{" + latex_escape(BASELINE_LABEL) + r"}} & "
                 r"\multicolumn{4}{c}{\textbf{" + latex_escape(COMPARE_MODEL_LABEL) + r"}} \\")
    lines.append(r"\textbf{Analysis} & \textbf{Subgroup} & "
                 r"$\mathbf{N_M}$ & $\mathbf{N_C}$ & "
                 r"\textbf{TP} & \textbf{FP} & \textbf{FN} & \textbf{Gap} & "
                 r"\textbf{TP} & \textbf{FP} & \textbf{FN} & \textbf{Gap} \\")
    lines.append(r"\hline")

    def row_line(analysis_label: str, name: str, base: Dict[str, int], comp: Dict[str, int], first: bool) -> str:
        a = latex_escape(analysis_label) if first else ""
        return (f"{a} & {latex_escape(name)} & {base['N_M']} & {base['N_C']} & "
                f"{dashfmt(base['TP'])} & {dashfmt(base['FP'])} & {dashfmt(base['FN'])} & {dashfmt(base['Gap'])} & "
                f"{dashfmt(comp['TP'])} & {dashfmt(comp['FP'])} & {dashfmt(comp['FN'])} & {dashfmt(comp['Gap'])} \\\\")

    # Overall
    analysis, name, base, comp = results[0]
    lines.append(row_line(r"\textbf{Overall}", name, base, comp, True))
    lines.append(r"\hline")

    # Viewpoint
    vp_rows = [r for r in results if r[0] == "Viewpoint"]
    lines.append(r"\multirow{" + str(len(vp_rows)) + r"}{*}{\textbf{Viewpoint}}")
    for i, (analysis, name, base, comp) in enumerate(vp_rows):
        lines.append("& " + row_line("", name, base, comp, False).lstrip("& ").strip())
    lines.append(r"\hline")

    # Pattern
    pat_rows = [r for r in results if r[0] == "Pattern"]
    lines.append(r"\multirow{" + str(len(pat_rows)) + r"}{*}{\textbf{Pattern}}")
    for i, (analysis, name, base, comp) in enumerate(pat_rows):
        lines.append("& " + row_line("", name, base, comp, False).lstrip("& ").strip())
    lines.append(r"\hline")

    # Mismatch type
    type_rows = [r for r in results if r[0] == "Mismatch type"]
    lines.append(r"\multirow{" + str(len(type_rows)) + r"}{*}{\textbf{Mismatch type}}")
    for i, (analysis, name, base, comp) in enumerate(type_rows):
        lines.append("& " + row_line("", name, base, comp, False).lstrip("& ").strip())
    lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    latex = "\n".join(lines)

    with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"\nWrote LaTeX table to: {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
