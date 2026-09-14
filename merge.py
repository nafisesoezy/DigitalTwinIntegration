#!/usr/bin/env python3

import os
import glob
import pandas as pd


# =========================
# Configuration
# =========================
# Folder containing annotated CSVs
INPUT_DIR = os.environ.get("LLM_SOLUTIONS_DIR", "mismatch_solutions")

# Output merged file
OUTPUT_FILE = os.path.join(INPUT_DIR, "ALL_annotated_reports.csv")


# =========================
# Columns to preserve (final schema)
# =========================
FINAL_COLUMNS = [
    "group",
    "bottleneck",
    "field",
    "pattern",
    "required_check",
    "A_value",
    "B_value",
    "AB_value",
    "detail",
    "result",
    "ab_kind",
    "LLM-Result",
    "LLM-suggestion",
]


# =========================
# Merge Function
# =========================
def merge_all_reports(input_dir: str, output_file: str) -> None:
    """
    Merge all annotated CSV reports into a single master CSV.
    """

    pattern = os.path.join(input_dir, "annotated_mismatch_report_*_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise RuntimeError(f"No annotated files found in: {input_dir}")

    print(f"Found {len(files)} annotated reports.")
    merged_frames = []

    for file_path in files:
        print(f"Reading: {file_path}")
        df = pd.read_csv(file_path)

        # Keep only expected columns (if they exist)
        existing_cols = [c for c in FINAL_COLUMNS if c in df.columns]
        df2 = df[existing_cols].copy()

        # Add missing columns if necessary
        for col in FINAL_COLUMNS:
            if col not in df2.columns:
                df2[col] = ""

        # Reorder columns consistently
        df2 = df2[FINAL_COLUMNS]

        merged_frames.append(df2)

    merged_df = pd.concat(merged_frames, ignore_index=True)

    merged_df.to_csv(output_file, index=False)

    print("===================================")
    print(f"✅ Master report created:")
    print(f"{output_file}")
    print(f"Total rows: {len(merged_df)}")
    print("===================================")


# =========================
# Main
# =========================
if __name__ == "__main__":
    merge_all_reports(INPUT_DIR, OUTPUT_FILE)