#!/usr/bin/env python3

import pandas as pd


INPUT_PATH = "allLLM_match_report_groundTruth.csv"
OUTPUT_PATH = "allLLM_match_report_groundTruth_withoutMissing.csv"


def main():
    # Load CSV
    df = pd.read_csv(INPUT_PATH)

    # Check if required column exists
    if "result" not in df.columns:
        raise KeyError(
            f"'result' column not found. Available columns: {list(df.columns)}"
        )

    # Remove rows where result == "Missing" (case-insensitive, trimmed)
    df_filtered = df[
        ~df["result"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("missing")
    ].copy()

    # Save filtered file
    df_filtered.to_csv(OUTPUT_PATH, index=False)

    print(f"Original rows: {len(df)}")
    print(f"Filtered rows: {len(df_filtered)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()