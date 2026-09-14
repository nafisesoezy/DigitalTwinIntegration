#!/usr/bin/env python3
"""
hybrid_evaluate.py

Replaces `llm_mismatch_solver_basedonMismatchReport_v3.py` (aka
`all_llm_triage.py` in the README). Implements Stage 2 of Algorithm 1
("Pattern-aware and LLM-assisted compatibility assessment", Section 4.3)
as the two GENUINELY SEPARATE LLM calls shown in Fig. `lst:compatibility_prompt`
("Two-stage prompt templates for LLM-assisted compatibility evaluation,
explanation, and adaptation recommendation"):

  * Every instantiated constraint's EvaluationMode is PREDEFINED by its
    ConstraintTemplate (constraint_templates.py) -- Deterministic or
    LLMAssisted. This script never lets an LLM override a Deterministic
    verdict, and it never asks the rule engine to pre-judge an LLMAssisted
    row before the LLM sees it.

  * Deterministic-mode rows: the verdict is copied from the rule engine's
    own `verdict` column (integration_bottleneckv6.py / enrich_with_templates).
    No LLM call is needed to determine it. By default the local, rule-
    generated explanation/adaptation are kept (cheap, always available);
    set REASON_DETERMINISTIC_VIA_LLM=1 to additionally run these rows
    through the SAME Stage 2 prompt used below, for a more natural-language
    explanation -- this can never change the verdict (Stage 2 is only ever
    given a fixed verdict to explain, and has no way to return a different
    one).

  * LLMAssisted-mode rows go through two SEPARATE calls, matching the
    figure exactly:
      Stage 1 - Compatibility Evaluation: given constraint, criterion,
        model_a_evidence, model_b_evidence, is_requirements, returns ONLY
        a verdict (Match/Mismatch/Gap). Nothing else is asked for or
        returned at this stage.
      Stage 2 - Explanation and Adaptation: given the SAME evidence PLUS
        the fixed verdict from Stage 1, returns an explanation and (for
        Mismatch) an adaptation / (for Gap) the missing information
        required / (for Match) "no adaptation required".
    Stage 2 cannot change the verdict Stage 1 produced -- it is not even
    given the option to return one. This is the key behavioral difference
    from the earlier combined-call design (and from the original
    all_llm_triage.py, which only ever confirmed-or-overturned rows the
    rule engine had already labeled "Mismatch", never touched rule-labeled
    "Match" rows, and never produced "Gap" at all).

  * "Execution Constraint Compatibility" (Appendix Table, RuntimeLevel,
    LLMAssisted) and "Execution Ordering Compatibility" (Deterministic) both
    read the same `execution_constraints` evidence but ask different
    questions. The rule engine only emits one row for that field (mapped to
    the deterministic Ordering template). This script SYNTHESIZES the
    additional LLMAssisted "Execution Constraint Compatibility" row from the
    same evidence, per constraint_templates.EXTRA_LLM_TEMPLATES below.

Field names sent to the LLM match the figure literally: `constraint`,
`criterion`, `model_a_evidence`, `model_b_evidence`, `is_requirements` (Stage
2 additionally receives `verdict`). `pattern` is also included as extra
context (the figure omits it, but Section 4's pattern-aware instantiation
depends on it) -- it does not appear in either stage's OUTPUT.

Env vars (same names/semantics as the script this replaces):
  LLM_BASE_URL, LLM_API_KEY, LLM_CHAT_ENDPOINT
  MATCH_REPORT_PATH            (default: match_report.csv -- the enriched
                                 output of integration_bottleneckv6.py)
  ALL_LLM_MATCH_REPORT_PATH    (default: allLLM_match_report.csv)
  LLM_CHUNK_SIZE, LLM_TIMEOUT_S, LLM_MAX_RETRIES
  LLM_MODELS                   (optional comma-separated override)
  REASON_DETERMINISTIC_VIA_LLM (default "0"; set "1" to also run
                                 Deterministic-mode rows through the Stage 2
                                 prompt for a natural-language explanation)

Repeated-run stability (opt-in, off by default):
  Set LLM_NUM_RUNS=N (N>1) to run Stage 1 -- the verdict-determining call --
  N independent times per LLM-assisted constraint, with identical evidence
  and settings each time (no shared state, no caching between runs). This
  answers a DIFFERENT question than accuracy: how stable is the verdict when
  nothing about the input changes? It does NOT majority-vote the N runs into
  a single "improved" verdict -- that would silently turn repeated sampling
  into a new ensemble method and change what is being evaluated. Each run is
  kept as its own independent replication in its own column
  (LLM-result-<m>-run1 .. -run{N}); Resultsv6-stability.py scores each run
  separately against ground truth and reports mean +/- SD across runs, plus
  the fraction of constraint instances where all N runs agreed (see that
  script and README.md for the exact methodology). Stage 2 (explanation/
  adaptation) is run once, using run 1's verdict -- this feature targets
  verdict stability, not explanation-text stability. Deterministic-mode
  rows are copied into every run column unchanged (by construction they
  cannot vary between runs).

Output columns added per model <m>:
  LLM-result-<m>            Stage 1 verdict (run 1): Match | Mismatch | Gap | Error
  LLM-result-<m>-run{i}     Stage 1 verdict for repetition i (1..LLM_NUM_RUNS;
                            with the default LLM_NUM_RUNS=1 this is just -run1,
                            identical to LLM-result-<m>)
  LLM-explanation-<m>       Stage 2 output (why), from run 1's verdict
  LLM-suggestion-<m>        Stage 2 output (adaptation / required information) --
                            kept under this name for backward compatibility with
                            scripts that already read "LLM-suggestion-*"
"""

import os
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from constraint_templates import classify, ConstraintTemplate

# =========================
# Config (same env vars as the script this replaces)
# =========================
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "").rstrip("/")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
CHAT_ENDPOINT = os.environ.get("LLM_CHAT_ENDPOINT", f"{LLM_BASE_URL}/api/v0/chat/completions")

MATCH_REPORT_PATH = os.environ.get("MATCH_REPORT_PATH", "match_report.csv")
ALL_LLM_MATCH_REPORT_PATH = os.environ.get("ALL_LLM_MATCH_REPORT_PATH", "allLLM_match_report.csv")

LLM_CHUNK_SIZE = int(os.environ.get("LLM_CHUNK_SIZE", "20"))
LLM_TIMEOUT_S = int(os.environ.get("LLM_TIMEOUT_S", "600"))
LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "4"))
REASON_DETERMINISTIC_VIA_LLM = os.environ.get("REASON_DETERMINISTIC_VIA_LLM", "0") == "1"
# Repeated-run stability (opt-in): see module docstring. 1 = current/default
# behavior (single Stage 1 call per constraint, unchanged output schema
# other than the always-present "-run1" column).
LLM_NUM_RUNS = max(1, int(os.environ.get("LLM_NUM_RUNS", "1")))

DEFAULT_MODELS: List[str] = [
    "openai/gpt-oss-120b",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "default-text-large",
]
_env_models = os.environ.get("LLM_MODELS", "").strip()
LLM_MODELS: List[str] = [m.strip() for m in _env_models.split(",") if m.strip()] if _env_models else DEFAULT_MODELS

# (bottleneck, field) -> extra ConstraintTemplate NAME to synthesize on top
# of the row the rule engine already produced. Evidence is shared; only the
# question (and hence the template/evaluation mode) differs.
EXTRA_LLM_TEMPLATES: List[Tuple[str, str, str]] = [
    ("Execution Constraint Mismatch", "execution_constraints", "Execution Constraint Compatibility"),
]


# =========================
# Prompts: Stage 1 (evaluation) and Stage 2 (explanation + adaptation),
# genuinely separate calls, matching Fig. lst:compatibility_prompt.
# =========================
STAGE1_PROMPT_TEMPLATE = """Stage 1 - Compatibility Evaluation

You are assessing the compatibility of two models for an intended Digital Twin integration.

Each item below gives:
  constraint: the compatibility condition to assess
  criterion: the evaluation criterion
  model_a_evidence: the relevant evidence extracted from Model A's metadata
  model_b_evidence: the relevant evidence extracted from Model B's metadata
  is_requirements: the relevant requirement from the integration specification / realized model
  pattern: the selected integration pattern (context only; not part of the verdict rule)

Assess the compatibility condition using only the supplied evidence.
Do not infer or assume model properties that are not provided. If the evidence
needed to decide is missing or insufficient, you MUST return "Gap" -- never
guess a Match or Mismatch to fill the gap.

Return:
- Match: sufficient evidence shows the condition is satisfied.
- Mismatch: sufficient evidence shows the condition is not satisfied.
- Gap: evidence is missing or insufficient.

Return ONLY valid JSON, no markdown, no prose outside the JSON:
{
  "results": [
    {"row_ref": "<copy the row_ref field exactly as given>", "verdict": "Match|Mismatch|Gap"}
  ]
}

Items (JSON array):
<<ROWS_JSON>>
"""

STAGE2_PROMPT_TEMPLATE = """Stage 2 - Explanation and Adaptation

You are explaining the result of a compatibility assessment. The verdict for each item was
ALREADY DETERMINED in Stage 1 and is given to you fixed -- you may NOT change it here.

Each item below gives:
  constraint: the compatibility condition that was assessed
  criterion: the evaluation criterion
  model_a_evidence: the relevant evidence extracted from Model A's metadata
  model_b_evidence: the relevant evidence extracted from Model B's metadata
  is_requirements: the relevant requirement from the integration specification / realized model
  verdict: the fixed Stage 1 assessment verdict (Match | Mismatch | Gap)

Using only the supplied evidence, explain why the given verdict applies.
Do not infer or assume model properties that are not provided.

If Mismatch, recommend an adaptation to address the incompatibility.
If Gap, identify the additional information required.
If Match, no adaptation is required.

Return ONLY valid JSON:
{
  "results": [
    {"row_ref": "...", "explanation": "...", "adaptation": "..."}
  ]
}

Items (JSON array, each includes its fixed verdict):
<<ROWS_JSON>>
"""


# =========================
# HTTP helpers (identical pattern to llm_mismatch_solver_basedonMismatchReport_v3.py)
# =========================
def _strip_code_fences(s: str) -> str:
    t = (s or "").strip()
    if not t.startswith("```"):
        return t
    lines = t.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def call_llm_json(model: str, prompt: str, timeout_s: int, max_retries: int) -> Dict[str, Any]:
    if not LLM_BASE_URL:
        raise RuntimeError("LLM_BASE_URL is empty (set env var, e.g., https://willma.surf.nl)")
    if not LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY is empty (set env var to the FULL key)")

    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Output strictly valid JSON only. No markdown, no prose."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
    }

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(CHAT_ENDPOINT, headers=headers, json=payload, timeout=timeout_s)
            if resp.status_code != 200:
                raise RuntimeError(f"API error {resp.status_code}: {resp.text}")
            data = resp.json()
            content = _strip_code_fences(data["choices"][0]["message"]["content"])
            return json.loads(content)
        except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout) as e:
            last_err = e
            sleep_s = min(30, 2 ** attempt)
            print(f"[warn] [{model}] Timeout attempt {attempt}/{max_retries}. Retrying in {sleep_s}s...")
            time.sleep(sleep_s)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"[{model}] Model returned non-JSON: {e}")
        except Exception as e:
            last_err = e
            sleep_s = min(15, attempt * 3)
            print(f"[warn] [{model}] Error attempt {attempt}/{max_retries}: {e}. Retrying in {sleep_s}s...")
            time.sleep(sleep_s)

    raise RuntimeError(f"[{model}] Failed after {max_retries} attempts. Last error: {last_err}")


# =========================
# Row preparation -- maps this pipeline's internal column names onto the
# figure's field names (constraint / criterion / model_a_evidence /
# model_b_evidence / is_requirements) for everything sent to the LLM.
# =========================
def _row_ref(idx: int) -> str:
    return f"row-{idx}"


def _clip(v: Any) -> str:
    return "" if pd.isna(v) else str(v)[:400]


def _stage1_payload(idx: int, r: pd.Series) -> Dict[str, Any]:
    return {
        "row_ref": _row_ref(idx),
        "constraint": _clip(r.get("constraint_template", "")),
        "criterion": _clip(r.get("required_check", "")),
        "model_a_evidence": _clip(r.get("A_value", "")),
        "model_b_evidence": _clip(r.get("B_value", "")),
        "is_requirements": _clip(r.get("AB_value", "")),
        "pattern": _clip(r.get("pattern", "")),
    }


def _stage2_payload(idx: int, r: pd.Series, verdict: str) -> Dict[str, Any]:
    d = _stage1_payload(idx, r)
    d.pop("pattern", None)
    d["verdict"] = verdict
    return d


def synthesize_extra_llm_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the extra LLMAssisted rows described in EXTRA_LLM_TEMPLATES (e.g.
    "Execution Constraint Compatibility" alongside the rule engine's
    deterministic "Execution Ordering Compatibility" row for the same
    execution_constraints evidence). Returns a NEW DataFrame with the extra
    rows appended (df itself is not mutated).
    """
    extra_rows = []
    for bottleneck, field, extra_template_name in EXTRA_LLM_TEMPLATES:
        tmpl = next((t for t in _all_templates_by_name().values() if t.name == extra_template_name), None)
        if tmpl is None:
            continue
        src = df[(df["bottleneck"] == bottleneck) & (df["field"] == field)]
        for _, r in src.iterrows():
            new_r = r.copy()
            new_r["constraint_template"] = tmpl.name
            new_r["constraint_category"] = tmpl.category
            new_r["rm_odp_viewpoint"] = tmpl.viewpoint
            new_r["evaluation_mode"] = tmpl.evaluation_mode
            new_r["in_paper_appendix"] = True
            new_r["verdict"] = ""       # to be determined by Stage 1
            new_r["explanation"] = ""
            new_r["adaptation"] = ""
            extra_rows.append(new_r)
    if not extra_rows:
        return df
    return pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)


def _all_templates_by_name():
    from constraint_templates import all_templates
    return all_templates()


# =========================
# Stage 1 / Stage 2 call helpers (batched, chunked, retried)
# =========================
def _run_stage1(df_out: pd.DataFrame, model: str, idx: List[int], col_result: str) -> None:
    """Compatibility Evaluation: returns ONLY a verdict per row."""
    for start in range(0, len(idx), LLM_CHUNK_SIZE):
        chunk = idx[start:start + LLM_CHUNK_SIZE]
        payload_rows = [_stage1_payload(i, df_out.loc[i]) for i in chunk]
        prompt = STAGE1_PROMPT_TEMPLATE.replace("<<ROWS_JSON>>", json.dumps(payload_rows, ensure_ascii=False))

        try:
            solution = call_llm_json(model, prompt, LLM_TIMEOUT_S, LLM_MAX_RETRIES) or {}
        except Exception as e:
            for i in chunk:
                df_out.at[i, col_result] = "Error"
            print(f"[error] [{model}] Stage 1 chunk starting at {start} failed: {e}")
            continue

        by_ref = {str(item.get("row_ref", "")).strip(): item for item in (solution.get("results") or [])}
        for i in chunk:
            item = by_ref.get(_row_ref(i))
            if not item:
                df_out.at[i, col_result] = "Gap"
                continue
            verdict = str(item.get("verdict", "")).strip() or "Gap"
            if verdict not in ("Match", "Mismatch", "Gap"):
                verdict = "Gap"
            df_out.at[i, col_result] = verdict


def _run_stage2(df_out: pd.DataFrame, model: str, idx: List[int], verdict_col: str,
                col_expl: str, col_sugg: str) -> None:
    """Explanation and Adaptation: given a FIXED verdict, returns explanation + adaptation."""
    idx = [i for i in idx if df_out.at[i, verdict_col] not in ("", "Error")]
    for start in range(0, len(idx), LLM_CHUNK_SIZE):
        chunk = idx[start:start + LLM_CHUNK_SIZE]
        payload_rows = [_stage2_payload(i, df_out.loc[i], df_out.at[i, verdict_col]) for i in chunk]
        prompt = STAGE2_PROMPT_TEMPLATE.replace("<<ROWS_JSON>>", json.dumps(payload_rows, ensure_ascii=False))

        try:
            solution = call_llm_json(model, prompt, LLM_TIMEOUT_S, LLM_MAX_RETRIES) or {}
        except Exception as e:
            for i in chunk:
                df_out.at[i, col_expl] = f"Stage 2 call failed: {e}"
            print(f"[warn] [{model}] Stage 2 chunk starting at {start} failed: {e} (keeping any existing explanation)")
            continue

        by_ref = {str(item.get("row_ref", "")).strip(): item for item in (solution.get("results") or [])}
        for i in chunk:
            item = by_ref.get(_row_ref(i))
            if not item:
                df_out.at[i, col_expl] = df_out.at[i, col_expl] or "No Stage 2 response returned for this row."
                continue
            df_out.at[i, col_expl] = str(item.get("explanation", "")).strip()
            df_out.at[i, col_sugg] = str(item.get("adaptation", "")).strip()


# =========================
# Per-model evaluation
# =========================
def ensure_model_columns(df_out: pd.DataFrame, model: str) -> Tuple[str, str, str, List[str]]:
    suffix = model.replace(" ", "_")
    col_result = f"LLM-result-{suffix}"
    col_expl = f"LLM-explanation-{suffix}"
    col_sugg = f"LLM-suggestion-{suffix}"   # kept for backward compatibility
    run_cols = [f"{col_result}-run{i}" for i in range(1, LLM_NUM_RUNS + 1)]
    for c in [col_result, col_expl, col_sugg] + run_cols:
        if c not in df_out.columns:
            df_out[c] = ""
    return col_result, col_expl, col_sugg, run_cols


def run_one_model(df_out: pd.DataFrame, model: str) -> None:
    col_result, col_expl, col_sugg, run_cols = ensure_model_columns(df_out, model)

    det_idx = [i for i, r in df_out.iterrows() if r.get("evaluation_mode") == "Deterministic"]
    llm_idx = [i for i, r in df_out.iterrows() if r.get("evaluation_mode") == "LLMAssisted"]

    # --- Deterministic rows: verdict is fixed by the rule engine, not Stage 1/2.
    #     Copied into every run column unchanged -- by construction a
    #     deterministic check cannot vary between repeated runs. ---
    for i in det_idx:
        r = df_out.loc[i]
        v = r.get("verdict", "")
        df_out.at[i, col_result] = v
        for rc in run_cols:
            df_out.at[i, rc] = v
        df_out.at[i, col_expl] = r.get("explanation", "")
        df_out.at[i, col_sugg] = r.get("adaptation", "")

    if REASON_DETERMINISTIC_VIA_LLM and det_idx:
        _run_stage2(df_out, model, det_idx, col_result, col_expl, col_sugg)

    # --- LLMAssisted rows: Stage 1 run LLM_NUM_RUNS independent times (each
    #     an unrelated call with the same evidence/settings -- no majority
    #     voting; see module docstring), then Stage 2 once using run 1's
    #     verdict. ---
    if not llm_idx:
        print(f"[info] [{model}] No LLM-assisted rows to evaluate.")
        return

    for run_i, rc in enumerate(run_cols, start=1):
        _run_stage1(df_out, model, llm_idx, rc)
        print(f"[info] [{model}] Stage 1 run {run_i}/{LLM_NUM_RUNS} complete.")

    # Primary result column mirrors run 1, so every existing script that
    # reads a single "LLM-result-<model>" column keeps working unchanged.
    for i in llm_idx:
        df_out.at[i, col_result] = df_out.at[i, run_cols[0]]

    _run_stage2(df_out, model, llm_idx, col_result, col_expl, col_sugg)


# =========================
# Main
# =========================
def main() -> None:
    if not os.path.exists(MATCH_REPORT_PATH):
        raise RuntimeError(f"Input file not found: {MATCH_REPORT_PATH}")
    if not LLM_BASE_URL:
        raise RuntimeError("LLM_BASE_URL is empty.")
    if not LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY is empty.")

    df = pd.read_csv(MATCH_REPORT_PATH)
    for c in ("constraint_template", "constraint_category", "rm_odp_viewpoint",
              "evaluation_mode", "verdict", "explanation", "adaptation"):
        if c not in df.columns:
            raise KeyError(
                f"'{c}' column missing from {MATCH_REPORT_PATH}. "
                f"Run the updated integration_bottleneckv6.py first (it calls "
                f"enrich_with_templates() before writing this file)."
            )

    df_out = synthesize_extra_llm_rows(df).copy()

    print("Models to run:")
    for m in LLM_MODELS:
        print(f"  - {m}")
    print(f"Deterministic rows: {(df_out['evaluation_mode'] == 'Deterministic').sum()}")
    print(f"LLM-assisted rows:  {(df_out['evaluation_mode'] == 'LLMAssisted').sum()}")
    if LLM_NUM_RUNS > 1:
        print(f"Repeated-run stability mode: LLM_NUM_RUNS={LLM_NUM_RUNS} independent Stage 1 "
              f"calls per LLM-assisted constraint (see Resultsv6-stability.py).")

    for model in LLM_MODELS:
        try:
            print(f"\nRunning model: {model}")
            run_one_model(df_out, model)
            print(f"Done: {model}")
        except Exception as e:
            col_result, col_expl, col_sugg, run_cols = ensure_model_columns(df_out, model)
            df_out[col_result] = "Error"
            df_out[col_expl] = f"Model run failed: {e}"
            df_out[col_sugg] = ""
            for rc in run_cols:
                df_out[rc] = "Error"
            print(f"[error] Model failed: {model}: {e}")

    df_out.to_csv(ALL_LLM_MATCH_REPORT_PATH, index=False)
    print(f"\nWrote combined report: {ALL_LLM_MATCH_REPORT_PATH}")


if __name__ == "__main__":
    main()
