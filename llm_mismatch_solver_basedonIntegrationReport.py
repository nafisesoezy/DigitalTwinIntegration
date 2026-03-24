#!/usr/bin/env python3
import os
import json
import glob
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

# =========================
# Config via environment vars
# =========================
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "").rstrip("/")     # e.g., https://willma.surf.nl
LLM_API_KEY  = os.environ.get("LLM_API_KEY", "")
LLM_MODEL    = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
CHAT_ENDPOINT = os.environ.get("LLM_CHAT_ENDPOINT", f"{LLM_BASE_URL}/api/v0/chat/completions")

REPORT_DIR = os.environ.get("INTEGRATION_REPORT_DIR", "integration_reports")
OUT_DIR    = os.environ.get("LLM_ANNOTATED_DIR", "integration_solutions")

os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE = int(os.environ.get("LLM_BATCH_SIZE", "20"))          # rows per LLM call
SLEEP_S    = float(os.environ.get("LLM_SLEEP_S", "0.2"))          # between calls
TIMEOUT_S  = int(os.environ.get("LLM_TIMEOUT_S", "600"))
MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "4"))


# =========================
# Prompt (your detection logic)
# =========================
PROMPT_TEMPLATE = """You are a Digital Twin integration engineer.

You are given rows from an "integration report" table. Each row describes:
- bottleneck + field
- integration pattern
- A_value, B_value, AB_value (AB can be missing)
Your job is to RE-INFER the correct status and propose fixes.

Use these rules (MUST follow):

GENERAL (all viewpoints):
1) If AB_value is not missing:
   - Check (A vs B) compatibility AND check ((A+B) vs AB) compatibility.
2) If AB_value is missing:
   - Check only (A vs B) compatibility.

DOMAIN viewpoint (Enterprise):
- For Title/Description/Keywords/Scope/Purpose&Pattern/Assumptions:
  If AB_value is missing -> return Missing.
  Else check whether the goal/intent/assumptions of A+B can satisfy AB (semantic alignment).

INFORMATION viewpoint:
- Determine which edge(s) are activated by the IntegrationPattern:
  One-Way: only A -> B edge.
  Loose/Shared: A -> B and B -> A edges.
  Integrated/Embedded: A->B and/or B->A only if enabled by the pattern/IS (use AB_value and pattern text if present).
- Then decide mismatch based on the relevant edge(s):
  E.g., for One-Way check A.output compatible with B.input (and AB if present).

OTHER viewpoints (Computational/Engineering/Technology):
- Decide mismatch based on A and B runtime/technical compatibility (and AB if present).

IMPORTANT:
- Missing metadata means A_value or B_value is missing when needed for judging compatibility.
- AB_value missing: do NOT automatically mark as problem except for DOMAIN viewpoint as specified above.

Output format:
Return ONLY valid JSON, as:
{
  "results": [
    {
      "row_id": <int>,
      "LLM-Result": "Match|Mismatch|Missing|Unsure",
      "LLM-suggestion": "<short fix if mismatch else empty>"
    }
  ]
}

Rows:
<<ROWS_JSON>>
"""


# =========================
# Helpers
# =========================
def _is_blank(x: Any) -> bool:
    if x is None:
        return True
    s = str(x).strip().lower()
    return s in ("", "none", "(none)", "nan", "null")

def _strip_code_fences(s: str) -> str:
    t = s.strip()
    if not t.startswith("```"):
        return t
    lines = t.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()

def call_llm_json(prompt: str) -> Dict[str, Any]:
    if not LLM_BASE_URL:
        raise RuntimeError("LLM_BASE_URL is empty")
    if not LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY is empty")

    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "Return strictly valid JSON only. No markdown."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
    }

    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(CHAT_ENDPOINT, headers=headers, json=payload, timeout=TIMEOUT_S)
            if resp.status_code != 200:
                raise RuntimeError(f"API error {resp.status_code}: {resp.text}")
            data = resp.json()
            content = _strip_code_fences(data["choices"][0]["message"]["content"])
            return json.loads(content)
        except Exception as e:
            last_err = e
            time.sleep(min(15, attempt * 2))
    raise RuntimeError(f"LLM call failed after retries. Last error: {last_err}")

def chunk_indices(n: int, k: int) -> List[List[int]]:
    out = []
    i = 0
    while i < n:
        out.append(list(range(i, min(n, i + k))))
        i += k
    return out

def build_rows_payload(df: pd.DataFrame, idxs: List[int]) -> List[Dict[str, Any]]:
    rows = []
    for i in idxs:
        r = df.iloc[i]
        rows.append({
            "row_id": int(i),
            "group": str(r.get("group", "")),
            "ab_kind": str(r.get("ab_kind", "")),
            "bottleneck": str(r.get("bottleneck", "")),
            "field": str(r.get("field", "")),
            "pattern": str(r.get("pattern", "")),
            "required_check": str(r.get("required_check", "")),
            "A_value": "" if _is_blank(r.get("A_value")) else str(r.get("A_value")),
            "B_value": "" if _is_blank(r.get("B_value")) else str(r.get("B_value")),
            "AB_value": "" if _is_blank(r.get("AB_value")) else str(r.get("AB_value")),
            "detail": str(r.get("detail", "")),
        })
    return rows

def annotate_one_file(csv_path: str) -> str:
    df = pd.read_csv(csv_path)

    # Ensure columns exist
    if "LLM-Result" not in df.columns:
        df["LLM-Result"] = ""
    if "LLM-suggestion" not in df.columns:
        df["LLM-suggestion"] = ""

    # Only annotate rows that are not already filled (resume-friendly)
    pending = [i for i in range(len(df)) if not str(df.at[i, "LLM-Result"]).strip()]
    if not pending:
        out_path = os.path.join(OUT_DIR, os.path.basename(csv_path))
        df.to_csv(out_path, index=False)
        return out_path

    # Batch calls
    for batch in chunk_indices(len(pending), BATCH_SIZE):
        idxs = [pending[j] for j in batch]
        rows_payload = build_rows_payload(df, idxs)
        prompt = PROMPT_TEMPLATE.replace("<<ROWS_JSON>>", json.dumps(rows_payload, ensure_ascii=False))

        resp = call_llm_json(prompt)
        results = resp.get("results", [])

        # Write back
        by_id = {int(x["row_id"]): x for x in results if "row_id" in x}
        for i in idxs:
            x = by_id.get(int(i))
            if not x:
                df.at[i, "LLM-Result"] = "Unsure"
                df.at[i, "LLM-suggestion"] = ""
                continue
            df.at[i, "LLM-Result"] = str(x.get("LLM-Result", "Unsure")).strip()
            sugg = str(x.get("LLM-suggestion", "")).strip()
            df.at[i, "LLM-suggestion"] = sugg

        time.sleep(SLEEP_S)

    out_path = os.path.join(OUT_DIR, os.path.basename(csv_path))
    df.to_csv(out_path, index=False)
    return out_path

def main():
    files = sorted(glob.glob(os.path.join(REPORT_DIR, "integration_report_*.csv")))
    if not files:
        raise RuntimeError(f"No integration reports found in {REPORT_DIR} (integration_report_*.csv)")

    print(f"Using LLM: {LLM_MODEL}")
    print(f"Reports: {len(files)}")
    for p in files:
        try:
            out = annotate_one_file(p)
            print(f"✅ annotated: {p} -> {out}")
        except Exception as e:
            print(f"❌ failed: {p}: {e}")

if __name__ == "__main__":
    main()