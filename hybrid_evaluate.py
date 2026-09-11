#!/usr/bin/env python3
# Updated for the RM-ODP constraint-template pipeline (see README.md).
"""
hybrid_evaluate.py

Replaces `llm_mismatch_solver_basedonMismatchReport_v3.py` (aka
`all_llm_triage.py` in the README). Implements Stage 2 of Algorithm 1
("Pattern-aware and LLM-assisted compatibility assessment", Section 4.3)
properly:

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
    through the LLM-Assisted Reasoner for a more natural-language
    explanation -- this can never change the verdict (Algorithm 1, line
    "LLM_AssistedReason(r, ctx_r, v)" has no verdict field to return).

  * LLMAssisted-mode rows: the LLM independently determines Match / Mismatch
    / Gap FROM THE EVIDENCE (constraint template + A/B/AB values), not by
    re-triaging a rule-based guess. This is the key behavioral difference
    from the old all_llm_triage.py, which only ever confirmed-or-overturned
    rows the rule engine had already labeled "Mismatch", never touched
    rule-labeled "Match" rows, and never produced "Gap" at all.
    The evaluator call and the reasoner call are combined into ONE JSON
    request per row for efficiency (same retry/chunking machinery as the
    original script), but the response schema keeps `verdict` (evaluator)
    and `explanation`/`adaptation` (reasoner) as separate fields so the
    conceptual separation in Algorithm 1 is still visible in the output.

  * "Execution Constraint Compatibility" (Appendix Table, RuntimeLevel,
    LLMAssisted) and "Execution Ordering Compatibility" (Deterministic) both
    read the same `execution_constraints` evidence but ask different
    questions. The rule engine only emits one row for that field (mapped to
    the deterministic Ordering template). This script SYNTHESIZES the
    additional LLMAssisted "Execution Constraint Compatibility" row from the
    same evidence, per constraint_templates.EXTRA_LLM_TEMPLATES below.

Env vars (same names/semantics as the script this replaces):
  LLM_BASE_URL, LLM_API_KEY, LLM_CHAT_ENDPOINT
  MATCH_REPORT_PATH            (default: match_report.csv -- the enriched
                                 output of integration_bottleneckv6.py)
  ALL_LLM_MATCH_REPORT_PATH    (default: allLLM_match_report.csv)
  LLM_CHUNK_SIZE, LLM_TIMEOUT_S, LLM_MAX_RETRIES
  LLM_MODELS                   (optional comma-separated override)
  REASON_DETERMINISTIC_VIA_LLM (default "0"; set "1" to also run
                                 Deterministic-mode rows through the LLM
                                 reasoner for a natural-language explanation)

Output columns added per model <m>:
  LLM-result-<m>        verdict:      Match | Mismatch | Gap | Error
  LLM-explanation-<m>   reasoner output (why)
  LLM-suggestion-<m>    reasoner output (candidate adaptation) -- kept under
                        this name for backward compatibility with scripts
                        that already read "LLM-suggestion-*"
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
# Prompt: LLM-Assisted Evaluator + Reasoner, combined per row
# =========================
EVAL_PROMPT_TEMPLATE = """You are a Digital Twin integration engineer applying ONE predefined \
compatibility constraint template to evidence extracted from two candidate models (A, B) and an \
integration specification/realized model (AB). The constraint's evaluation mode has ALREADY been \
decided to be LLM-assisted (semantic/contextual interpretation required) -- you are not choosing \
whether to use a rule or an LLM, you ARE the evaluator for this constraint.

For each row you are given:
  - constraint_template: the name of the compatibility condition to assess
  - field / required_check: what the condition requires
  - A_value / B_value / AB_value: the relevant evidence extracted from each model's metadata
  - pattern: the selected integration pattern (One-Way, Loose, Shared, Integrated, Embedded)

Task, for EACH row:
1) Determine a verdict:
   - "Match": the evidence is sufficient and shows the condition IS satisfied.
   - "Mismatch": the evidence is sufficient and shows the condition is NOT satisfied.
   - "Gap": the evidence is absent or insufficient to decide either way.
   Only classify "Mismatch" when the values are genuinely incompatible for the stated
   constraint (not merely differently worded). If A_value or B_value is empty/missing,
   that is normally a "Gap", not a "Mismatch".
2) Give a short (1-2 sentence) explanation for the verdict.
3) If (and only if) the verdict is "Mismatch", give a concrete candidate adaptation
   (a mediation/conversion/orchestration step that would resolve it). For "Match" say
   "No adaptation required." For "Gap" say what missing metadata is needed.

The verdict you return is authoritative for this constraint (this is the LLM-Assisted
Evaluator step). The explanation/adaptation are the LLM-Assisted Reasoner step, reported
alongside it -- they never change a verdict that was fixed elsewhere; here there is no
other verdict since this row's evaluation mode is LLM-assisted.

Return ONLY valid JSON, no markdown, no prose outside the JSON:
{
  "results": [
    {
      "row_ref": "<copy the row_ref field exactly as given>",
      "verdict": "Match|Mismatch|Gap",
      "explanation": "...",
      "adaptation": "..."
    }
  ]
}

Rows (JSON array):
<<ROWS_JSON>>
"""

# Lighter prompt used only when REASON_DETERMINISTIC_VIA_LLM=1: verdict is
# fixed and given; the LLM only rewrites the explanation/adaptation in
# natural language. It is explicitly told it cannot change the verdict.
REASON_ONLY_PROMPT_TEMPLATE = """You are writing the human-readable explanation for compatibility \
verdicts that were ALREADY DETERMINED by a deterministic rule engine. You may NOT change any verdict. \
For each row, given constraint_template, the evidence, and the fixed verdict, write:
  - "explanation": a short (1-2 sentence) explanation of why that verdict follows from the evidence.
  - "adaptation": if verdict=="Mismatch", a concrete candidate adaptation; if verdict=="Match", \
"No adaptation required."; if verdict=="Gap", what missing metadata is needed.

Return ONLY valid JSON:
{
  "results": [
    {"row_ref": "...", "explanation": "...", "adaptation": "..."}
  ]
}

Rows (JSON array, each already includes its fixed "verdict"):
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
# Row preparation
# =========================
EVIDENCE_COLS = ["constraint_template", "field", "required_check", "pattern",
                  "A_value", "B_value", "AB_value", "detail"]


def _row_ref(idx: int) -> str:
    return f"row-{idx}"


def _row_payload(idx: int, r: pd.Series, fixed_verdict: Optional[str] = None) -> Dict[str, Any]:
    d = {"row_ref": _row_ref(idx)}
    for c in EVIDENCE_COLS:
        v = r.get(c, "")
        d[c] = "" if pd.isna(v) else str(v)[:400]
    if fixed_verdict is not None:
        d["verdict"] = fixed_verdict
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
            new_r["verdict"] = ""       # to be determined by the LLM evaluator
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
# Per-model evaluation
# =========================
def ensure_model_columns(df_out: pd.DataFrame, model: str) -> Tuple[str, str, str]:
    suffix = model.replace(" ", "_")
    col_result = f"LLM-result-{suffix}"
    col_expl = f"LLM-explanation-{suffix}"
    col_sugg = f"LLM-suggestion-{suffix}"   # kept for backward compatibility
    for c in (col_result, col_expl, col_sugg):
        if c not in df_out.columns:
            df_out[c] = ""
    return col_result, col_expl, col_sugg


def run_one_model(df_out: pd.DataFrame, model: str) -> None:
    col_result, col_expl, col_sugg = ensure_model_columns(df_out, model)

    det_idx = [i for i, r in df_out.iterrows() if r.get("evaluation_mode") == "Deterministic"]
    llm_idx = [i for i, r in df_out.iterrows() if r.get("evaluation_mode") == "LLMAssisted"]

    # --- Deterministic rows: verdict is NOT re-decided by the LLM. ---
    for i in det_idx:
        r = df_out.loc[i]
        df_out.at[i, col_result] = r.get("verdict", "")
        df_out.at[i, col_expl] = r.get("explanation", "")
        df_out.at[i, col_sugg] = r.get("adaptation", "")

    if REASON_DETERMINISTIC_VIA_LLM and det_idx:
        _run_reason_only(df_out, model, det_idx, col_expl, col_sugg)

    # --- LLMAssisted rows: verdict IS determined by the LLM, from evidence. ---
    if not llm_idx:
        print(f"[info] [{model}] No LLM-assisted rows to evaluate.")
        return

    for start in range(0, len(llm_idx), LLM_CHUNK_SIZE):
        chunk = llm_idx[start:start + LLM_CHUNK_SIZE]
        payload_rows = [_row_payload(i, df_out.loc[i]) for i in chunk]
        prompt = EVAL_PROMPT_TEMPLATE.replace("<<ROWS_JSON>>", json.dumps(payload_rows, ensure_ascii=False))

        try:
            solution = call_llm_json(model, prompt, LLM_TIMEOUT_S, LLM_MAX_RETRIES) or {}
        except Exception as e:
            for i in chunk:
                df_out.at[i, col_result] = "Error"
                df_out.at[i, col_expl] = f"LLM call failed: {e}"
                df_out.at[i, col_sugg] = ""
            print(f"[error] [{model}] chunk starting at {start} failed: {e}")
            continue

        by_ref = {str(item.get("row_ref", "")).strip(): item for item in (solution.get("results") or [])}
        for i in chunk:
            ref = _row_ref(i)
            item = by_ref.get(ref)
            if not item:
                df_out.at[i, col_result] = "Gap"
                df_out.at[i, col_expl] = "No LLM response returned for this row."
                df_out.at[i, col_sugg] = ""
                continue
            verdict = str(item.get("verdict", "")).strip() or "Gap"
            if verdict not in ("Match", "Mismatch", "Gap"):
                verdict = "Gap"
            df_out.at[i, col_result] = verdict
            df_out.at[i, col_expl] = str(item.get("explanation", "")).strip()
            df_out.at[i, col_sugg] = str(item.get("adaptation", "")).strip()


def _run_reason_only(df_out: pd.DataFrame, model: str, det_idx: List[int], col_expl: str, col_sugg: str) -> None:
    for start in range(0, len(det_idx), LLM_CHUNK_SIZE):
        chunk = det_idx[start:start + LLM_CHUNK_SIZE]
        payload_rows = [_row_payload(i, df_out.loc[i], fixed_verdict=df_out.loc[i, "verdict"]) for i in chunk]
        prompt = REASON_ONLY_PROMPT_TEMPLATE.replace("<<ROWS_JSON>>", json.dumps(payload_rows, ensure_ascii=False))
        try:
            solution = call_llm_json(model, prompt, LLM_TIMEOUT_S, LLM_MAX_RETRIES) or {}
        except Exception as e:
            print(f"[warn] [{model}] reasoner pass failed for chunk at {start}: {e} (keeping local explanation)")
            continue
        by_ref = {str(item.get("row_ref", "")).strip(): item for item in (solution.get("results") or [])}
        for i in chunk:
            item = by_ref.get(_row_ref(i))
            if item:
                df_out.at[i, col_expl] = str(item.get("explanation", "")).strip() or df_out.at[i, col_expl]
                df_out.at[i, col_sugg] = str(item.get("adaptation", "")).strip() or df_out.at[i, col_sugg]


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

    for model in LLM_MODELS:
        try:
            print(f"\nRunning model: {model}")
            run_one_model(df_out, model)
            print(f"Done: {model}")
        except Exception as e:
            col_result, col_expl, col_sugg = ensure_model_columns(df_out, model)
            df_out[col_result] = "Error"
            df_out[col_expl] = f"Model run failed: {e}"
            df_out[col_sugg] = ""
            print(f"[error] Model failed: {model}: {e}")

    df_out.to_csv(ALL_LLM_MATCH_REPORT_PATH, index=False)
    print(f"\nWrote combined report: {ALL_LLM_MATCH_REPORT_PATH}")


if __name__ == "__main__":
    main()
