# Digital Twin Integration Compatibility Analysis

This repository provides a **rule-based and LLM-assisted framework** for analyzing compatibility and integration feasibility between heterogeneous Digital Twin models.

## What it does
- Extracts metadata from YAML model descriptions
- Detects mismatches across multiple viewpoints:
  - conceptual
  - informational
  - computational
  - engineering
  - technological
- Supports multiple integration patterns:
  - One-Way, Loose, Shared, Integrated, Embedded
- Uses LLMs to re-check mismatches, reduce false positives, and suggest fixes

## Main scripts
- `model_schema_matcher.py`  
  Runs deterministic mismatch detection and generates `match_report.csv`

- `all_llm_triage.py`  
  Runs multiple LLMs on mismatch rows and generates `allLLM_match_report.csv`

## Input
YAML metadata files for:
- Model A
- Model B
- Intended / integrated AB specifications

## Output
- `match_report.csv` — rule-based results
- `allLLM_match_report.csv` — LLM-enhanced results
- `summary_report.csv` — overall summary

## Run
```bash
python model_schema_matcher.py modelsMetadataFull/
python all_llm_triage.py
