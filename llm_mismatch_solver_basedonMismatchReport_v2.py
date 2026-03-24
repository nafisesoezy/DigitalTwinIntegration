#!/usr/bin/env python3
import os
import json
import time
from typing import Any, Dict, Optional, List

import pandas as pd
import requests


# =========================
# Config via environment vars
# =========================
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "").rstrip("/")     # e.g., https://willma.surf.nl
LLM_API_KEY  = os.environ.get("LLM_API_KEY", "")
LLM_MODEL    = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b") # must match /api/v0/models id exactly
CHAT_ENDPOINT = os.environ.get("LLM_CHAT_ENDPOINT", f"{LLM_BASE_URL}/api/v0/chat/completions")

# Input/Output
MATCH_REPORT_PATH = os.environ.get("MATCH_REPORT_PATH", "match_report.csv")
LLM_MATCH_REPORT_PATH = os.environ.get("LLM_MATCH_REPORT_PATH", "LLM_match_report.csv")

# LLM batching
LLM_CHUNK_SIZE = int(os.environ.get("LLM_CHUNK_SIZE", "60"))  # safe default

# Optional: avoid sending some fields to the LLM
SKIP_FIELDS = {"model_version", "distribution_version"}


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
    """Which of A_value / B_value / AB_value are missing (any combination)."""
    miss: List[str] = []
    if _is_blank(row.get("A_value")):
        miss.append("A")
    if _is_blank(row.get("B_value")):
        miss.append("B")
    if _is_blank(row.get("AB_value")):
        miss.append("AB")
    return "+".join(miss)


def should_send_to_llm(row: pd.Series) -> bool:
    """Send only mismatch rows where A and B exist and field not excluded."""
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
    t = s.strip()
    if not t.startswith("```"):
        return t
    lines = t.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def call_llm_json(prompt: str, timeout_s: int = 600, max_retries: int = 4) -> Dict[str, Any]:
    if not LLM_BASE_URL:
        raise RuntimeError("LLM_BASE_URL is empty (set env var, e.g., https://willma.surf.nl)")
    if not LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY is empty (set env var to the FULL key)")

    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": LLM_MODEL,
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
            print(f"⚠️ Timeout attempt {attempt}/{max_retries}. Retrying in {sleep_s}s...")
            time.sleep(sleep_s)

        except json.JSONDecodeError as e:
            raise RuntimeError(f"Model returned non-JSON: {e}")

        except Exception as e:
            last_err = e
            sleep_s = min(15, attempt * 3)
            print(f"⚠️ Error attempt {attempt}/{max_retries}: {e}. Retrying in {sleep_s}s...")
            time.sleep(sleep_s)

    raise RuntimeError(f"Failed after {max_retries} attempts. Last error: {last_err}")


def build_llm_annotations(solution: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """Map row_ref -> triage/justification/fix."""
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


# =========================
# Main pipeline
# =========================
def main() -> None:
    if not os.path.exists(MATCH_REPORT_PATH):
        raise RuntimeError(f"Input file not found: {MATCH_REPORT_PATH}")

    df = pd.read_csv(MATCH_REPORT_PATH)
    df_out = df.copy()
    df_out["LLM_result"] = ""
    df_out["LLM-suggestion"] = ""

    # ---- Rule 1 + 2: local, per row ----
    for i, row in df_out.iterrows():
        r = _norm(row.get("result", ""))

        if r == "missing":
            df_out.at[i, "LLM_result"] = "missing"
            miss = missing_sides_aba(row)
            df_out.at[i, "LLM-suggestion"] = f"Missing fields: {miss}" if miss else "Missing fields: (undetected)"
            continue

        if r == "match":
            df_out.at[i, "LLM_result"] = "Match"
            continue

    # ---- Rule 3: mismatch => LLM (chunked) ----
    mismatch_idx = [i for i, row in df_out.iterrows() if should_send_to_llm(row)]

    # If there are mismatch rows we can't send (e.g., excluded/blank A/B), still fill deterministically
    for i, row in df_out.iterrows():
        if _norm(row.get("result", "")) == "mismatch" and not should_send_to_llm(row):
            df_out.at[i, "LLM_result"] = "Mismatch"
            df_out.at[i, "LLM-suggestion"] = "LLM skipped (A/B missing or field excluded)."

    if mismatch_idx:
        all_annotations: Dict[str, Dict[str, str]] = {}

        # chunk by chunk
        for start in range(0, len(mismatch_idx), LLM_CHUNK_SIZE):
            chunk_ids = mismatch_idx[start:start + LLM_CHUNK_SIZE]
            df_chunk = df_out.loc[chunk_ids].copy()

            csv_blob = dataframe_to_compact_csv(df_chunk)
            prompt = PROMPT_TEMPLATE.replace("<<REPORT_CSV>>", csv_blob)
            solution = call_llm_json(prompt)
            ann = build_llm_annotations(solution or {})

            # merge annotations
            all_annotations.update(ann)

        # apply LLM results back to rows
        for i in mismatch_idx:
            row = df_out.loc[i]
            key = _row_key(row.get("field", ""), row.get("bottleneck", ""))
            info = all_annotations.get(key, {})
            triage = (info.get("triage", "") or "").strip()

            if triage == "NOT_A_MISMATCH":
                df_out.at[i, "LLM_result"] = "Match"
                df_out.at[i, "LLM-suggestion"] = info.get("justification", "")
            elif triage == "REAL_MISMATCH":
                df_out.at[i, "LLM_result"] = "Mismatch"
                df_out.at[i, "LLM-suggestion"] = info.get("fix", "") or info.get("justification", "")
            else:
                df_out.at[i, "LLM_result"] = "Mismatch"
                df_out.at[i, "LLM-suggestion"] = "No LLM triage returned for this row."

    df_out.to_csv(LLM_MATCH_REPORT_PATH, index=False)
    print(f"✅ Wrote: {LLM_MATCH_REPORT_PATH}")


if __name__ == "__main__":
    main()