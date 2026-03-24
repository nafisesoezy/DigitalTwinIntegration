#!/usr/bin/env python3
import os
import json
import glob
import time
from typing import Any, Dict, Optional

import pandas as pd
import requests


# =========================
# Config via environment vars
# =========================
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "").rstrip("/")     # e.g., https://willma.surf.nl
LLM_API_KEY  = os.environ.get("LLM_API_KEY", "")
LLM_MODEL    = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b") # must match /api/v0/models id exactly
OUT_DIR      = os.environ.get("LLM_SOLUTIONS_DIR", "mismatch_solutions")
REPORT_DIR   = os.environ.get("MISMATCH_REPORT_DIR", "mismatch_reports")

CHAT_ENDPOINT = os.environ.get("LLM_CHAT_ENDPOINT", f"{LLM_BASE_URL}/api/v0/chat/completions")

os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# Fields we never send to LLM
# =========================
SKIP_FIELDS = {"model_version", "distribution_version"}


# =========================
# Prompt: triage ONLY mismatches where A and B exist
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
    "group": "...",
    "ab_kind": "...",
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
      "proposed_fix": "...",
      "data_transform": {
        "rules": ["..."],
        "example": "..."
      },
      "runtime_orchestration": {
        "pattern": "...",
        "steps": ["..."]
      },
      "yaml_patches": { "A": "...", "B": "...", "AB": "..." },
      "validation": ["..."]
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


def _norm_field(x: Any) -> str:
    return str(x).strip().lower()


def _ab_missing_side(row: pd.Series) -> str:
    """
    Returns: "A", "B", "A+B", or "" (no missing in A/B).
    AB is not considered here.
    """
    a_blank = _is_blank(row.get("A_value"))
    b_blank = _is_blank(row.get("B_value"))

    if a_blank and b_blank:
        return "A+B"
    if a_blank:
        return "A"
    if b_blank:
        return "B"
    return ""


def _should_skip_llm(row: pd.Series) -> bool:
    """
    Don't send to LLM when:
      - field is model_version or distribution_version
      - A or B is missing
    """
    field = _norm_field(row.get("field", ""))
    if field in SKIP_FIELDS:
        return True
    if _ab_missing_side(row) != "":
        return True
    return False


def preprocess_for_llm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Send to LLM:
      - everything EXCEPT:
        * model_version / distribution_version rows
        * rows where A or B missing
        * rows that are already Match (no need)
    """
    df2 = df.copy()

    # drop Match rows (no need to send)
    if "result" in df2.columns:
        df2 = df2[df2["result"].astype(str).str.strip().str.lower() != "match"].copy()

    # drop skip rows (versions + missing A/B)
    df2 = df2[~df2.apply(_should_skip_llm, axis=1)].copy()

    return df2


def dataframe_to_compact_csv(df: pd.DataFrame, max_rows: int = 60) -> str:
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

    if len(df2) > max_rows:
        df2 = df2.head(max_rows)

    return df2.to_csv(index=False)


def _row_key(field: Any, bottleneck: Any) -> str:
    return f"{str(field).strip()}||{str(bottleneck).strip()}".lower()


# =========================
# LLM call (timeout + retries)
# =========================
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

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }

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
            content = data["choices"][0]["message"]["content"]
            content = _strip_code_fences(content)
            return json.loads(content)

        except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout) as e:
            last_err = e
            sleep_s = min(30, 2 ** attempt)
            print(f"⚠️ Timeout attempt {attempt}/{max_retries}. Retrying in {sleep_s}s...")
            time.sleep(sleep_s)

        except json.JSONDecodeError as e:
            raise RuntimeError(f"Model returned non-JSON (JSONDecodeError): {e}")

        except Exception as e:
            last_err = e
            sleep_s = min(15, attempt * 3)
            print(f"⚠️ Error attempt {attempt}/{max_retries}: {e}. Retrying in {sleep_s}s...")
            time.sleep(sleep_s)

    raise RuntimeError(f"Failed after {max_retries} attempts. Last error: {last_err}")


# =========================
# Build annotations from LLM solution (only for mismatch triage + fixes)
# =========================
def build_llm_annotations(solution: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """
    Returns: dict[key] = {"triage": "...", "justification": "...", "fix": "..."}
    key is "<field>||<bottleneck>" lowercase.
    """
    out: Dict[str, Dict[str, str]] = {}

    # triage decisions
    for t in solution.get("mismatch_triage", []) or []:
        row_ref = str(t.get("row_ref", "")).strip().lower()
        decision = str(t.get("decision", "")).strip()
        justification = str(t.get("justification", "")).strip()
        if "||" not in row_ref:
            continue
        out.setdefault(row_ref, {})
        out[row_ref]["triage"] = decision
        out[row_ref]["justification"] = justification

    # fixes for real mismatches
    for m in solution.get("mismatches", []) or []:
        field = str(m.get("field", "")).strip()
        bottleneck = str(m.get("bottleneck", "")).strip()
        key = _row_key(field, bottleneck)
        fix = str(m.get("proposed_fix", "")).strip() or str(m.get("root_cause", "")).strip()
        out.setdefault(key, {})
        out[key]["fix"] = fix

    return out

def _normalize_field_name(x: Any) -> str:
    return str(x).strip().lower().replace(" ", "_")

def annotate_report(df: pd.DataFrame, solution: Optional[Dict[str, Any]]) -> pd.DataFrame:
    """
    Local rules (NO LLM):
      - field == model_version: LLM-Result=Match, suggestion="version can vary but ..."
      - field == distribution_version: LLM-Result=Match, suggestion="version can vary but ..."
      - missing in A or B: LLM-Result=Gap, suggestion="field A/B/A+B missing"
    LLM rules:
      - For mismatch rows that were sent to LLM:
          * NOT_A_MISMATCH => result=LLM-match + LLM-Result=Match
          * REAL_MISMATCH  => LLM-Result=Mismatch + suggestion=fix/justification
    """
    ann = build_llm_annotations(solution or {})

    df_out = df.copy()
    df_out["LLM-Result"] = ""
    df_out["LLM-suggestion"] = ""

    if not {"field", "bottleneck", "result"}.issubset(set(df_out.columns)):
        return df_out

    for i, row in df_out.iterrows():
        field = _norm_field(row.get("field", ""))
        original_result = str(row.get("result", "")).strip().lower()

        # -------------------------
        # Local: version fields
        # -------------------------

        field = _normalize_field_name(row.get("field", ""))
        if field == "model_version":
            df_out.at[i, "LLM-Result"] = "Match"
            df_out.at[i, "LLM-suggestion"] = (
                "Version can vary across models/distributions; this is typically not an integration blocker "
                "as long as interfaces, schemas, and runtime contracts remain compatible."
            )
            continue

        if field == "distribution_version":
            df_out.at[i, "LLM-Result"] = "Match"
            df_out.at[i, "LLM-suggestion"] = (
                "Version can vary across distributions/releases; treat as informational metadata unless it "
                "changes input/output schema, units, timestep, or execution dependencies."
            )
            continue

        # -------------------------
        # Local: A/B missing => Gap
        # -------------------------
        missing_side = _ab_missing_side(row)
        if missing_side != "":
            df_out.at[i, "LLM-Result"] = "Gap"
            df_out.at[i, "LLM-suggestion"] = f"field {missing_side} missing"
            continue

        # -------------------------
        # If row is already Match, ignore
        # -------------------------
        if original_result == "match":
            continue

        # -------------------------
        # LLM outputs for remaining rows
        # -------------------------
        key = _row_key(row.get("field", ""), row.get("bottleneck", ""))
        info = ann.get(key, {})

        if original_result == "mismatch":
            triage = info.get("triage", "").strip()

            if triage == "NOT_A_MISMATCH":
                df_out.at[i, "LLM-Result"] = "Match"
                df_out.at[i, "LLM-suggestion"] = info.get("justification", "")
                continue

            if triage == "REAL_MISMATCH":
                df_out.at[i, "LLM-Result"] = "Mismatch"
                df_out.at[i, "LLM-suggestion"] = info.get("fix", "") or info.get("justification", "")
                continue

            # If no triage found (e.g., prompt truncation), leave blank
            continue

        # Other statuses (if any): leave blank
        continue

    return df_out


# =========================
# Pipeline per report
# =========================
def solve_one_report(csv_path: str) -> None:
    df = pd.read_csv(csv_path)

    group = str(df["group"].iloc[0]) if "group" in df.columns and len(df) else "unknown"
    ab_kind = str(df["ab_kind"].iloc[0]) if "ab_kind" in df.columns and len(df) else "unknown"

    # Prepare rows that should be sent to LLM (everything except skip rules)
    df_llm = preprocess_for_llm(df)

    solution: Optional[Dict[str, Any]] = None

    # If nothing to send, skip LLM call entirely
    if len(df_llm) > 0:
        csv_blob = dataframe_to_compact_csv(df_llm, max_rows=60)
        prompt = PROMPT_TEMPLATE.replace("<<REPORT_CSV>>", csv_blob)
        solution = call_llm_json(prompt)

    base = os.path.splitext(os.path.basename(csv_path))[0]
    out_json = os.path.join(OUT_DIR, f"solution_{base}.json")

    # Always write a JSON (empty skeleton if LLM skipped)
    if solution is None:
        solution = {
            "summary": {
                "group": group,
                "ab_kind": ab_kind,
                "num_rows_seen": int(len(df_llm)),
                "num_real_mismatches": 0,
                "num_not_a_mismatch": 0,
                "risk_level": "low",
            },
            "mismatch_triage": [],
            "mismatches": [],
        }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(solution, f, indent=2, ensure_ascii=False)

    # Optional readable .md
    out_md = os.path.join(OUT_DIR, f"solution_{base}.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(f"# Solution for {base}\n\n")
        f.write("```json\n")
        f.write(json.dumps(solution, indent=2, ensure_ascii=False))
        f.write("\n```\n")

    # Annotated CSV (FULL original df + local rules + LLM triage)
    annotated = annotate_report(df, solution)
    out_csv = os.path.join(OUT_DIR, f"annotated_{base}.csv")
    annotated.to_csv(out_csv, index=False)

    print(f"✅ {group} / {ab_kind} -> {out_json}")
    print(f"✅ annotated csv -> {out_csv}")



# =========================
# Merge all annotated reports into one master CSV
# =========================
def merge_all_annotated_reports(
    in_dir: str = OUT_DIR,
    out_path: str = os.path.join(OUT_DIR, "ALL_annotated_reports.csv"),
) -> str:
    """
    Merges all per-report annotated CSVs into one master report.

    Expected columns (kept if present):
      group, bottleneck, field, pattern, required_check,
      A_value, B_value, AB_value, detail, result, ab_kind,
      LLM-Result, LLM-suggestion
    """
    files = sorted(glob.glob(os.path.join(in_dir, "annotated_mismatch_report_*_*.csv")))
    if not files:
        raise RuntimeError(f"No annotated CSV files found in: {in_dir}")

    keep_cols = [
        "group", "bottleneck", "field", "pattern", "required_check",
        "A_value", "B_value", "AB_value", "detail", "result", "ab_kind",
        "LLM-Result", "LLM-suggestion",
    ]

    merged = []
    for fp in files:
        df = pd.read_csv(fp)

        # keep only expected columns (if some are missing, ignore)
        cols = [c for c in keep_cols if c in df.columns]
        df2 = df[cols].copy()

        # ensure all keep_cols exist in final output (fill missing with "")
        for c in keep_cols:
            if c not in df2.columns:
                df2[c] = ""

        df2 = df2[keep_cols]
        merged.append(df2)

    out_df = pd.concat(merged, ignore_index=True)
    out_df.to_csv(out_path, index=False)
    print(f"✅ merged master report -> {out_path}")
    return out_path

def main() -> None:
    reports = sorted(glob.glob(os.path.join(REPORT_DIR, "mismatch_report_*_*.csv")))
    if not reports:
        raise RuntimeError(f"No mismatch reports found in: {REPORT_DIR}")

    print(f"Using base URL: {LLM_BASE_URL}")
    print(f"Using model: {LLM_MODEL}")
    print(f"Reports: {len(reports)}")

    for p in reports:
        try:
            solve_one_report(p)
        except Exception as e:
            print(f"❌ Failed for {p}: {e}")

    # After processing all reports, create one merged master report
    merge_all_annotated_reports(in_dir=OUT_DIR)

if __name__ == "__main__":
    main()