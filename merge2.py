#!/usr/bin/env python3
"""
Create allLLM_match_report_groundTruth.csv from allLLM_match_report.csv

Ground truth (Section 6.2.1 / 6.2.2): for each key = (group, field,
bottleneck, pattern), groundTruth is the deterministic rule engine's own
`result` column (integration_bottleneckv6.py), evaluated on the row where
ab_kind == 'INTEGRATED' -- the rule engine applied to the realized model's
own declared metadata. GT_SOURCE_COL must stay independent of every LLM
column scored against it in Table detection_performance.

Residual limitation: for Temporal Resolution / Spatial Resolution /
Dimensionality, `info_simple` treats a difference between A and B as a
Match only when `ab` declares a resampling/conversion policy for that
field. Where the corpus does not document that policy at all (currently
the common case), this reduces to comparing A directly against B, so
scores on these templates are provisional until policy coverage improves.

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
