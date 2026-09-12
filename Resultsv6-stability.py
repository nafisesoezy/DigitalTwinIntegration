#!/usr/bin/env python3
"""
Resultsv6-stability.py

Extends Table `detection_performance` (Resultsv2-2-withoutMissing.py) to
also answer a second, DIFFERENT question: given identical evidence and
settings, how stable is an LLM-assisted verdict across repeated calls?

Requires allLLM_match_report_groundTruth.csv to have been produced by a run
of hybrid_evaluate.py with LLM_NUM_RUNS>1 (columns
"LLM-result-<model>-run1" .. "-run{N}"). If only a single run exists
(LLM_NUM_RUNS=1, the default), every method's "Stable verdicts" column is
trivially 100% -- there is nothing to compare against yet -- and this script
still runs, but the stability column carries no information until you
actually re-run hybrid_evaluate.py with LLM_NUM_RUNS>1.

Methodology (do not change without also changing the paper text that cites
it):
  - Each of the N runs is scored independently against `groundTruth` using
    the SAME binary "mismatch is positive" methodology as
    Resultsv2-2-withoutMissing.py. We do NOT majority-vote the N runs into
    one verdict and score that -- that would silently turn repeated
    sampling into a new ensemble method and change what is being evaluated.
  - Accuracy/Precision/Recall/F1 are reported as mean +/- SD across the N
    per-run scores.
  - "Stable verdicts" is computed ONLY over LLMAssisted-mode constraint
    instances (Deterministic-mode rows cannot vary between runs by
    construction, so including them would inflate the number without
    telling you anything about the LLM). It is the fraction of those rows
    where all N run-columns hold the identical verdict string.
  - The deterministic rule-based baseline's "Stable verdicts" is reported as
    exactly 100% by definition (no repeated calls involved), not computed
    from data.

Run:
  python Resultsv6-stability.py

Optional env vars:
  GT_REPORT_PATH        (default: allLLM_match_report_groundTruth.csv)
  STABILITY_TABLE_TEX   (default: detection_performance_stability.tex)
"""

import os
import re
import statistics
from typing import Dict, List, Tuple

import pandas as pd


INPUT_CSV = os.environ.get("GT_REPORT_PATH", "allLLM_match_report_groundTruth.csv")
OUTPUT_TEX = os.environ.get("STABILITY_TABLE_TEX", "detection_performance_stability.tex")

GT_COL = "groundTruth"
EVAL_MODE_COL = "evaluation_mode"

# (display name, base "LLM-result-<...>" column, or "result" for the
# deterministic baseline)
METHODS: List[Tuple[str, str]] = [
    ("Rule-based (Static)", "result"),
    ("OpenAI GPT-OSS-120B", "LLM-result-openai/gpt-oss-120b"),
    ("Mistral Small 3.2 24B Instruct", "LLM-result-mistralai/Mistral-Small-3.2-24B-Instruct-2506"),
    ("default-text-large", "LLM-result-default-text-large"),
]

POS_LABEL = "mismatch"


def norm_label(x: object) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "missing"
    s = str(x).strip().lower()
    if not s or s == "nan":
        return "missing"
    if "mismatch" in s:
        return "mismatch"
    if s == "match" or " match" in s or s.startswith("match"):
        return "match"
    if "missing" in s or "gap" in s:
        return "missing"
    return s


def to_binary(label: str) -> int:
    return 1 if label == POS_LABEL else 0


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Tuple[float, float, float, float]:
    tp = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    tn = sum((yt == 0 and yp == 0) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))
    acc = safe_div(tp + tn, tp + tn + fp + fn)
    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    f1 = safe_div(2 * prec * rec, prec + rec)
    return acc, prec, rec, f1


def mean_sd(values: List[float]) -> Tuple[float, float]:
    m = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return m, sd


def find_run_columns(df: pd.DataFrame, base_col: str) -> List[str]:
    """All "<base_col>-run{i}" columns present, sorted by run index."""
    pattern = re.compile(re.escape(base_col) + r"-run(\d+)$")
    hits = []
    for c in df.columns:
        m = pattern.match(c)
        if m:
            hits.append((int(m.group(1)), c))
    return [c for _, c in sorted(hits)]


def latex_escape(s: str) -> str:
    return (s.replace("\\", r"\textbackslash{}")
             .replace("&", r"\&").replace("%", r"\%").replace("$", r"\$")
             .replace("#", r"\#").replace("_", r"\_")
             .replace("{", r"\{").replace("}", r"\}")
             .replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}"))


def score_method(df_eval: pd.DataFrame, base_col: str) -> Dict[str, object]:
    y_true = df_eval["_gt_bin"].tolist()
    run_cols = find_run_columns(df_eval, base_col)

    if base_col == "result" or not run_cols:
        # Deterministic baseline (or an LLM column with no repeated runs
        # recorded yet -- fall back to the single base column, N=1).
        cols_to_score = [base_col] if base_col in df_eval.columns else []
        stability_pct = 100.0  # deterministic by definition / not yet measured
    else:
        cols_to_score = run_cols
        # Stability: only over LLMAssisted rows, only when we actually have
        # more than one run to compare.
        llm_mask = df_eval[EVAL_MODE_COL] == "LLMAssisted"
        sub = df_eval.loc[llm_mask, run_cols]
        if len(run_cols) > 1 and not sub.empty:
            agree = (sub.nunique(axis=1) == 1)
            stability_pct = 100.0 * agree.mean()
        else:
            stability_pct = 100.0  # only one run recorded -- nothing to compare

    accs, precs, recs, f1s = [], [], [], []
    for col in cols_to_score:
        y_pred = df_eval[col].map(norm_label).map(to_binary).tolist()
        acc, prec, rec, f1 = compute_metrics(y_true, y_pred)
        accs.append(acc); precs.append(prec); recs.append(rec); f1s.append(f1)

    if not accs:
        raise KeyError(f"No usable column(s) found for base '{base_col}' "
                        f"(looked for '{base_col}' and '{base_col}-run*').")

    return {
        "n_runs": len(accs),
        "accuracy": mean_sd(accs),
        "precision": mean_sd(precs),
        "recall": mean_sd(recs),
        "f1": mean_sd(f1s),
        "stability_pct": stability_pct,
    }


def fmt_mean_sd(m: float, sd: float, n_runs: int) -> str:
    if n_runs <= 1:
        return f"{m:.2f}"
    return f"{m:.2f} $\\pm$ {sd:.2f}"


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    if GT_COL not in df.columns:
        raise KeyError(f"Missing required column '{GT_COL}'. Available: {list(df.columns)}")
    if EVAL_MODE_COL not in df.columns:
        raise KeyError(
            f"Missing '{EVAL_MODE_COL}' column -- this script needs the enriched "
            f"match_report.csv schema (integration_bottleneckv6.enrich_with_templates())."
        )

    df["_gt_norm"] = df[GT_COL].map(norm_label)
    df_eval = df[df["_gt_norm"] != "missing"].copy()
    if df_eval.empty:
        raise ValueError("No evaluable rows after dropping missing groundTruth.")
    df_eval["_gt_bin"] = df_eval["_gt_norm"].map(to_binary)

    rows = []
    for method_name, base_col in METHODS:
        try:
            res = score_method(df_eval, base_col)
        except KeyError as e:
            print(f"[warn] Skipping '{method_name}': {e}")
            continue
        rows.append((method_name, res))

    # ---- LaTeX table ----
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Detection performance with repeated-run stability. "
                  r"For LLM-assisted methods, each LLM-assisted constraint was evaluated "
                  r"$N$ times under identical evidence and inference settings; values are "
                  r"mean $\pm$ SD across runs. \emph{Stable verdicts} is the fraction of "
                  r"LLM-assisted constraint instances for which all $N$ runs returned the "
                  r"same verdict. The rule-based baseline involves no stochastic inference, "
                  r"so its stability is 100\% by construction.}")
    lines.append(r"\label{tab:detection_performance_stability}")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Method} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & "
                 r"\textbf{F1-score} & \textbf{Stable verdicts} \\")
    lines.append(r"\hline")

    print(f"{'Method':<32} {'N runs':>6}  {'Accuracy':>14}  {'Precision':>14}  "
          f"{'Recall':>14}  {'F1':>14}  {'Stable %':>9}")
    for method_name, res in rows:
        acc_s = fmt_mean_sd(*res["accuracy"], res["n_runs"])
        prec_s = fmt_mean_sd(*res["precision"], res["n_runs"])
        rec_s = fmt_mean_sd(*res["recall"], res["n_runs"])
        f1_s = fmt_mean_sd(*res["f1"], res["n_runs"])
        stab_s = f"{res['stability_pct']:.0f}\\%"
        lines.append(f"{latex_escape(method_name)} & {acc_s} & {prec_s} & {rec_s} & {f1_s} & {stab_s} \\\\")

        print(f"{method_name:<32} {res['n_runs']:>6}  "
              f"{fmt_mean_sd(*res['accuracy'], res['n_runs']):>14}  "
              f"{fmt_mean_sd(*res['precision'], res['n_runs']):>14}  "
              f"{fmt_mean_sd(*res['recall'], res['n_runs']):>14}  "
              f"{fmt_mean_sd(*res['f1'], res['n_runs']):>14}  "
              f"{res['stability_pct']:>8.1f}%")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    latex = "\n".join(lines)

    with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"\nWrote LaTeX table to: {OUTPUT_TEX}")

    max_runs = max((res["n_runs"] for _, res in rows), default=1)
    if max_runs <= 1:
        print(
            "\n[note] Every method above had only 1 run available, so every 'Stable "
            "verdicts' value is a trivial 100% (nothing to compare against). Re-run "
            "hybrid_evaluate.py with LLM_NUM_RUNS>1 (e.g. 5) to get a real stability "
            "measurement before quoting these numbers as evidence of anything."
        )
    else:
        print(f"\nSuggested paper sentence, filled in with the numbers above "
              f"(N={max_runs} runs):")
        for method_name, res in rows:
            if res["n_runs"] > 1:
                print(f'  "{method_name} produced the same Match/Mismatch/Gap verdict across '
                      f'all {res["n_runs"]} runs for {res["stability_pct"]:.0f}% of '
                      f'LLM-assisted constraint instances."')


if __name__ == "__main__":
    main()
