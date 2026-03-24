#!/usr/bin/env python3
"""
all_llm_triage.py

Runs the same mismatch-triage prompt across MULTIPLE chat models (same API key/base URL),
and writes ONE combined CSV report: allLLM_match_report.csv

For each model, two columns are added:
  - LLM-result-<MODEL_NAME>
  - LLM-suggestion-<MODEL_NAME>

Default behavior keeps your original deterministic handling for:
  - result=missing  -> model columns filled with "missing" + missing sides
  - result=match    -> model columns filled with "Match"
Then for result=mismatch with A/B present (and field not excluded), each model is called (chunked).

Env vars (same as your script, plus optional model list):
  LLM_BASE_URL                (required)
  LLM_API_KEY                 (required)
  LLM_CHAT_ENDPOINT           (optional; default: {LLM_BASE_URL}/api/v0/chat/completions)

  MATCH_REPORT_PATH           (default: match_report.csv)
  ALL_LLM_MATCH_REPORT_PATH   (default: allLLM_match_report.csv)

  LLM_CHUNK_SIZE              (default: 60)
  LLM_TIMEOUT_S               (default: 600)
  LLM_MAX_RETRIES             (default: 4)

  LLM_MODELS                  (optional; comma-separated list of model ids to run)
                              If not set, uses the hardcoded RUNNING list below.

Notes:
- Column names keep your requested exact model string.
- If a model errors, rows in that model's columns are filled with "Error" and a short message.
"""

import os
import json
import time
from typing import Any, Dict, Optional, List, Tuple

import pandas as pd
import requests


# =========================
# Config via environment vars
# =========================
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "").rstrip("/")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
CHAT_ENDPOINT = os.environ.get("LLM_CHAT_ENDPOINT", f"{LLM_BASE_URL}/api/v0/chat/completions")

MATCH_REPORT_PATH = os.environ.get("MATCH_REPORT_PATH", "match_report.csv")
ALL_LLM_MATCH_REPORT_PATH = os.environ.get("ALL_LLM_MATCH_REPORT_PATH", "allLLM_match_report.csv")

LLM_CHUNK_SIZE = int(os.environ.get("LLM_CHUNK_SIZE", "60"))
LLM_TIMEOUT_S = int(os.environ.get("LLM_TIMEOUT_S", "600"))
LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "4"))

# Optional: avoid sending some fields to the LLM
SKIP_FIELDS = {"model_version", "distribution_version"}


# =========================
# Models to run
# =========================
DEFAULT_MODELS: List[str] = [
    "openai/gpt-oss-120b",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "default-text-large",
]
# Allow override
_env_models = os.environ.get("LLM_MODELS", "").strip()
LLM_MODELS: List[str] = [m.strip() for m in _env_models.split(",") if m.strip()] if _env_models else DEFAULT_MODELS


# =========================
# Prompt
# =========================
PROMPT_TEMPLATE = """You are a Digital Twin integration engineer.

You are given CSV rows from a mismatch report table describing Model A, Model B, and an integration specification AB.

IMPORTANT RULES (must follow):
1) You will ONLY see rows where A_value and B_value are BOTH PRESENT.
2) AB is an integration specification; it is allowed to be more abstract and does NOT need to repeat A/B metadata.
   Therefore: DO NOT report "missing AB fields" as problems.
3) Each input row has a 'result' column. For rows with result=Mismatch, you must RE-CHECK semantically whether it is a REAL mismatch or not.
   Many rows are false positives due to simple similarity thresholds.
4) REAL mismatch rule:
   A REAL mismatch exists ONLY when A_value and B_value are incompatible for integration
   (format/unit/timestep/schema/handshake/runtime pattern, etc.)
   If real mismatch: provide reasoning + a concrete fix (converter/mapping/resampling/orchestration).
5) If row is Match, do not include it.

Your task:
- Produce a JSON report with:
  (A) mismatch_triage: a decision for each mismatch row (REAL_MISMATCH or NOT_A_MISMATCH)
  (B) mismatches: only REAL mismatches that require fixes

CRITICAL OUTPUT KEYING:
- For every triage item, set row_ref EXACTLY as: "<field>||<bottleneck>"
  (lowercase ok). This must match the CSV field/bottleneck columns.

Return ONLY valid JSON following this schema:
{
  "summary": {
    "num_rows_seen": 0,
    "num_real_mismatches": 0,
    "num_not_a_mismatch": 0,
    "risk_level": "low|medium|high"
  },
  "mismatch_triage": [
    {
      "row_ref": "field||bottleneck",
      "decision": "REAL_MISMATCH|NOT_A_MISMATCH",
      "justification": "..."
    }
  ],
  "mismatches": [
    {
      "id": "F1",
      "field": "...",
      "bottleneck": "...",
      "root_cause": "...",
      "evidence": { "A_value": "...", "B_value": "..." },
      "why_it_blocks_integration": "...",
      "proposed_fix": "..."
    }
  ]
}

CSV rows:
<<REPORT_CSV>>
"""


# =========================
# Helpers
# =========================
def _is_blank(x: Any) -> bool:
    if x is None:
        return True
    s = str(x).strip().lower()
    return s in ("", "none", "(none)", "nan", "null")


def _norm(x: Any) -> str:
    return str(x).strip().lower()


def _row_key(field: Any, bottleneck: Any) -> str:
    return f"{str(field).strip()}||{str(bottleneck).strip()}".lower()


def missing_sides_aba(row: pd.Series) -> str:
    miss: List[str] = []
    if _is_blank(row.get("A_value")):
        miss.append("A")
    if _is_blank(row.get("B_value")):
        miss.append("B")
    if _is_blank(row.get("AB_value")):
        miss.append("AB")
    return "+".join(miss)


def should_send_to_llm(row: pd.Series) -> bool:
    if _norm(row.get("result", "")) != "mismatch":
        return False
    field = _norm(row.get("field", ""))
    if field in SKIP_FIELDS:
        return False
    if _is_blank(row.get("A_value")) or _is_blank(row.get("B_value")):
        return False
    return True


def dataframe_to_compact_csv(df: pd.DataFrame) -> str:
    keep_cols = [
        "group", "ab_kind",
        "bottleneck", "field", "pattern", "required_check",
        "A_value", "B_value", "AB_value",
        "detail", "result",
    ]
    cols = [c for c in keep_cols if c in df.columns]
    df2 = df[cols].copy()

    for col in ["required_check", "A_value", "B_value", "AB_value", "detail"]:
        if col in df2.columns:
            df2[col] = df2[col].astype(str).str.slice(0, 250)

    return df2.to_csv(index=False)


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


def _sanitize_col_suffix(model_name: str) -> str:
    # Keep it readable and stable: replace spaces with underscores.
    # (Your requirement is "use the model name"; CSV headers work fine with spaces,
    # but underscores are safer in many tools. If you truly want spaces preserved,
    # change this to: return model_name
    return model_name.replace(" ", "_")


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
            print(f"⚠️ [{model}] Timeout attempt {attempt}/{max_retries}. Retrying in {sleep_s}s...")
            time.sleep(sleep_s)

        except json.JSONDecodeError as e:
            raise RuntimeError(f"[{model}] Model returned non-JSON: {e}")

        except Exception as e:
            last_err = e
            sleep_s = min(15, attempt * 3)
            print(f"⚠️ [{model}] Error attempt {attempt}/{max_retries}: {e}. Retrying in {sleep_s}s...")
            time.sleep(sleep_s)

    raise RuntimeError(f"[{model}] Failed after {max_retries} attempts. Last error: {last_err}")


def build_llm_annotations(solution: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}

    for t in solution.get("mismatch_triage", []) or []:
        row_ref = str(t.get("row_ref", "")).strip().lower()
        decision = str(t.get("decision", "")).strip()
        justification = str(t.get("justification", "")).strip()
        if "||" not in row_ref:
            continue
        out.setdefault(row_ref, {})
        out[row_ref]["triage"] = decision
        out[row_ref]["justification"] = justification

    for m in solution.get("mismatches", []) or []:
        field = str(m.get("field", "")).strip()
        bottleneck = str(m.get("bottleneck", "")).strip()
        key = _row_key(field, bottleneck)
        fix = str(m.get("proposed_fix", "")).strip() or str(m.get("root_cause", "")).strip()
        out.setdefault(key, {})
        out[key]["fix"] = fix

    return out


def ensure_model_columns(df_out: pd.DataFrame, model: str) -> Tuple[str, str]:
    suffix = _sanitize_col_suffix(model)
    col_result = f"LLM-result-{suffix}"
    col_sugg = f"LLM-suggestion-{suffix}"
    if col_result not in df_out.columns:
        df_out[col_result] = ""
    if col_sugg not in df_out.columns:
        df_out[col_sugg] = ""
    return col_result, col_sugg


def fill_deterministic_rows(df_out: pd.DataFrame, col_result: str, col_sugg: str) -> None:
    # Apply the same deterministic logic you had, but per model columns.
    for i, row in df_out.iterrows():
        r = _norm(row.get("result", ""))

        if r == "missing":
            df_out.at[i, col_result] = "missing"
            miss = missing_sides_aba(row)
            df_out.at[i, col_sugg] = f"Missing fields: {miss}" if miss else "Missing fields: (undetected)"
        elif r == "match":
            df_out.at[i, col_result] = "Match"
        # mismatches are handled later per model


def apply_skipped_mismatch_rows(df_out: pd.DataFrame, col_result: str, col_sugg: str) -> None:
    for i, row in df_out.iterrows():
        if _norm(row.get("result", "")) == "mismatch" and not should_send_to_llm(row):
            df_out.at[i, col_result] = "Mismatch"
            df_out.at[i, col_sugg] = "LLM skipped (A/B missing or field excluded)."


# =========================
# Main pipeline
# =========================
def run_one_model(df_out: pd.DataFrame, model: str) -> None:
    col_result, col_sugg = ensure_model_columns(df_out, model)

    # Deterministic base fill
    fill_deterministic_rows(df_out, col_result, col_sugg)
    apply_skipped_mismatch_rows(df_out, col_result, col_sugg)

    mismatch_idx = [i for i, row in df_out.iterrows() if should_send_to_llm(row)]

    if not mismatch_idx:
        print(f"ℹ️ [{model}] No eligible mismatch rows to send.")
        return

    all_annotations: Dict[str, Dict[str, str]] = {}

    for start in range(0, len(mismatch_idx), LLM_CHUNK_SIZE):
        chunk_ids = mismatch_idx[start:start + LLM_CHUNK_SIZE]
        df_chunk = df_out.loc[chunk_ids].copy()

        csv_blob = dataframe_to_compact_csv(df_chunk)
        prompt = PROMPT_TEMPLATE.replace("<<REPORT_CSV>>", csv_blob)

        solution = call_llm_json(
            model=model,
            prompt=prompt,
            timeout_s=LLM_TIMEOUT_S,
            max_retries=LLM_MAX_RETRIES,
        )
        ann = build_llm_annotations(solution or {})
        all_annotations.update(ann)

    # Apply LLM outputs
    for i in mismatch_idx:
        row = df_out.loc[i]
        key = _row_key(row.get("field", ""), row.get("bottleneck", ""))
        info = all_annotations.get(key, {})
        triage = (info.get("triage", "") or "").strip()

        if triage == "NOT_A_MISMATCH":
            df_out.at[i, col_result] = "Match"
            df_out.at[i, col_sugg] = info.get("justification", "")
        elif triage == "REAL_MISMATCH":
            df_out.at[i, col_result] = "Mismatch"
            df_out.at[i, col_sugg] = info.get("fix", "") or info.get("justification", "")
        else:
            df_out.at[i, col_result] = "Mismatch"
            df_out.at[i, col_sugg] = "No LLM triage returned for this row."


def main() -> None:
    if not os.path.exists(MATCH_REPORT_PATH):
        raise RuntimeError(f"Input file not found: {MATCH_REPORT_PATH}")
    if not LLM_BASE_URL:
        raise RuntimeError("LLM_BASE_URL is empty.")
    if not LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY is empty.")

    df = pd.read_csv(MATCH_REPORT_PATH)
    df_out = df.copy()

    print("Models to run:")
    for m in LLM_MODELS:
        print(f"  - {m}")

    for model in LLM_MODELS:
        try:
            print(f"\n▶ Running model: {model}")
            run_one_model(df_out, model)
            print(f"✅ Done: {model}")
        except Exception as e:
            # If a model fails, still add columns and mark error across all rows.
            col_result, col_sugg = ensure_model_columns(df_out, model)
            df_out[col_result] = "Error"
            df_out[col_sugg] = f"Model run failed: {e}"
            print(f"❌ Model failed: {model}: {e}")

    df_out.to_csv(ALL_LLM_MATCH_REPORT_PATH, index=False)
    print(f"\n✅ Wrote combined report: {ALL_LLM_MATCH_REPORT_PATH}")


if __name__ == "__main__":
    main()