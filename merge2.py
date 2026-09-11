#!/usr/bin/env python3
"""
Create allLLM_match_report_groundTruth.csv from allLLM_match_report.csv

Ground-truth definition (Section 6.2.1 / 6.2.2 in the paper):
  "These predictions are then compared with the metadata of the realized
   integrated model (AB), which serves as the ground truth for integration
   decisions."

GROUND TRUTH MUST BE INDEPENDENT OF ANY LLM. Concretely: for each key =
(group, field, bottleneck, pattern), the ground truth is the DETERMINISTIC
rule engine's own `result` column (integration_bottleneckv6.py), evaluated
on the row where ab_kind == 'INTEGRATED' -- i.e. the rule engine applied
directly to the realized model's own declared metadata, not any model's
opinion of it.

IMPORTANT (previous version of this script): an earlier revision used
'LLM-result-openai/gpt-oss-120b' as GT_SOURCE_COL. That made GPT-OSS-120B
simultaneously (a) one of the methods being scored in Table
detection_performance and (b) the source of the ground truth every method
-- including itself -- was scored against. That is a validity bug, not a
methodology choice: it mechanically biases Table 1 in GPT-OSS-120B's favor
and makes the rule-based baseline and the other LLMs' scores partly a
measure of agreement with GPT-OSS-120B rather than with the realized
integration. GT_SOURCE_COL is fixed below to the deterministic 'result'
column.

Known residual limitation (read before trusting Table 1 numbers):
Not every deterministic check in integration_bottleneckv6.py actually
looks at the `ab` argument -- the `info_simple` loop inside
check_information_viewpoint (Temporal Resolution / Spatial Resolution /
Dimensionality / their Coverage counterparts) compares A directly to B and
never touches `ab`. For those specific templates, `result` is IDENTICAL
whether ab_kind is INTENDED or INTEGRATED, so using it as ground truth is
tautological with the rule-based baseline's own prediction (the rule-based
method will score ~100% on them by construction). Fixing this properly
requires changing those checks to compare AB's own declared field value
against the requiring side (B) plus its declared
Resampling/Conversion Policy, matching the paper's Information-viewpoint
schema (Table env_viewpoint_fields). See constraint_templates.py /
"Known coverage gaps" and the accompanying write-up for the concrete
per-template ground-truth rule this should become. Until that lands,
treat scores on Temporal Resolution / Spatial Resolution / Dimensionality
Compatibility as provisional.

1) Add column 'groundTruth' using, for each key =
   (group, field, bottleneck, pattern),
   the value of GT_SOURCE_COL where ab_kind == 'INTEGRATED'.

2) Remove all rows where ab_kind == 'INTEGRATED'.

Notes:
- If a key has no INTEGRATED row, groundTruth will be empty (NaN).
- If a key has multiple INTEGRATED rows, the first one is used (and a warning is printed).
"""

import sys
import pandas as pd

IN_PATH = "allLLM_match_report.csv"
OUT_PATH = "allLLM_match_report_groundTruth.csv"

KEY_COLS = ["group", "field", "bottleneck", "pattern"]
AB_KIND_COL = "ab_kind"

# Deterministic, LLM-independent ground-truth source: the rule engine's own
# verdict when applied to the realized (INTEGRATED) model's metadata.
GT_SOURCE_COL = "result"
GT_COL = "groundTruth"


def main() -> int:
    df = pd.read_csv(IN_PATH)

    # Basic column checks
    missing = [c for c in (KEY_COLS + [AB_KIND_COL, GT_SOURCE_COL]) if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Build ground truth lookup from INTEGRATED rows
    df_int = df[df[AB_KIND_COL].astype(str).str.strip().str.upper() == "INTEGRATED"].copy()

    if df_int.empty:
        print("WARNING: No INTEGRATED rows found. groundTruth will be empty for all rows.", file=sys.stderr)
        gt_map = pd.DataFrame(columns=KEY_COLS + [GT_COL])
    else:
        # Check duplicate integrated keys
        dup_mask = df_int.duplicated(subset=KEY_COLS, keep=False)
        if dup_mask.any():
            dups = df_int.loc[dup_mask, KEY_COLS].drop_duplicates()
            print(
                f"WARNING: Found {len(dups)} duplicated INTEGRATED keys. "
                f"Using the first occurrence for each duplicated key.",
                file=sys.stderr,
            )

        gt_map = (
            df_int
            .sort_values(KEY_COLS)
            .drop_duplicates(subset=KEY_COLS, keep="first")[KEY_COLS + [GT_SOURCE_COL]]
            .rename(columns={GT_SOURCE_COL: GT_COL})
        )

    # Merge groundTruth back onto all rows
    df = df.merge(gt_map, on=KEY_COLS, how="left")

    # Remove INTEGRATED rows
    df_out = df[df[AB_KIND_COL].astype(str).str.strip().str.upper() != "INTEGRATED"].copy()

    # Write output
    df_out.to_csv(OUT_PATH, index=False)
    print(f"Wrote: {OUT_PATH} (rows: {len(df_out)})")
    print(f"Ground-truth source column: '{GT_SOURCE_COL}' on INTEGRATED rows (deterministic, LLM-independent).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
