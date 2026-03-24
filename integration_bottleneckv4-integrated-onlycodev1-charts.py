# Usage:
#   python model_schema_matcher.py /path/to/dir_or_files... [--debug]
# Scans YAMLs, groups by "<N>-<ROLE>-*.yaml", extracts metadata, infers AB pattern,
# evaluates mismatches (aligned with your table for Phase 1 & 2), writes "match_report.csv".

import os
import re
import sys
import yaml
import traceback
import pandas as pd

import numpy as np
import matplotlib
import os
import numpy as np
import pandas as pd
import string
import json  # ✅ add this line

import string
import matplotlib.patches as mpatches


matplotlib.use("Agg")  # safe backend
import matplotlib.pyplot as plt

from typing import List, Dict, Any, Tuple, Optional, Set, Iterable
from dataclasses import dataclass, field
from collections import defaultdict

# ============================================================
# Text/Token utilities
# ============================================================

STOPWORDS = {
    "a","an","and","as","at","beneath","by","canonical","component","components","coupled",
    "data","dataset","datasets","description","exchange","exchanges","expected","field","fields",
    "flux","fluxes","for","from","ice","in","input","inputs","intended","into","measurement",
    "measurements","model","models","ocean","of","on","or","output","outputs","rate","rates",
    "schema","sea","shelf","surface","the","to","under","variable","variables","with"
}

def normalize_phrase(s: str) -> str:
    if not s:
        return ""
    s = s.replace("–", "-")
    s = re.sub(r"\s+and\s+", ", ", s, flags=re.IGNORECASE)
    s = s.replace("/", " / ")
    s = re.sub(r"[\(\)\[\]]", " ", s)  # drop bracket content (units) for "name" comparison
    s = re.sub(r"[:,;|]+", ",", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()

def split_variables(text: str) -> List[str]:
    text = normalize_phrase(text)
    if not text:
        return []
    parts = [p.strip() for p in text.split(",") if p.strip()]
    out: List[str] = []
    for p in parts:
        p2 = re.sub(r"\s+", " ", p).strip()
        if p2:
            out.append(p2)
    return out

def tokenize(s: str) -> Set[str]:
    s = normalize_phrase(s)
    tokens = re.findall(r"[a-zA-Z0-9\-\+°²/]+", s)
    return {t for t in tokens if t not in STOPWORDS and len(t) > 1}

def jaccard_token_similarity(a: str, b: str) -> float:
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0

def phrase_similarity(a: str, b: str) -> float:
    # light-weight for variable strings; falls back to token Jaccard
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.9
    return jaccard_token_similarity(a, b)

def list_best_pairwise(src: List[str], dst: List[str]) -> Tuple[float, List[Tuple[str, str, float]]]:
    """Average of best matches from src to dst + list of (src, best_dst, score)."""
    if not src or not dst:
        return (0.0, [])
    matches = []
    scs: List[float] = []
    for s in src:
        best_b, best = "", 0.0
        for d in dst:
            sim = phrase_similarity(s, d)
            if sim > best:
                best, best_b = sim, d
        matches.append((s, best_b, best))
        scs.append(best)
    return ((sum(scs)/len(scs)) if scs else 0.0, matches)

def any_semantic_overlap(src_vars: List[str], dst_vars: List[str], min_sim: float = 0.50) -> Tuple[bool, List[Tuple[str,str,float]]]:
    score, pairs = list_best_pairwise(src_vars, dst_vars)
    good = [(s,d,sc) for (s,d,sc) in pairs if sc >= min_sim]
    return (len(good) > 0, good)

def dedupe(items: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

# ============================================================
# Dataclasses
# ============================================================

@dataclass
class IOSchema:
    variables: List[str] = field(default_factory=list)
    units: List[str] = field(default_factory=list)

@dataclass
class ModelMeta:
    file: str
    group: str
    role: str
    name: str
    root: Dict[str, Any]
    outputs: IOSchema = field(default_factory=IOSchema)
    inputs: IOSchema = field(default_factory=IOSchema)
    ab_pattern: str = ""
    ab_in: List[str] = field(default_factory=list)
    ab_out: List[str] = field(default_factory=list)
    canonical_exchange: List[str] = field(default_factory=list)
    fields: Dict[str, Any] = field(default_factory=dict)  # bag of extracted fields

# ============================================================
# Filename grouping
# ============================================================

def guess_group_role_from_filename(path: str) -> Tuple[str, str]:
    base = os.path.basename(path)
    m = re.match(r"^(\d+)-(A|B|AB-Intended|AB-Integrated)(?:[-_.]|$)", base, flags=re.IGNORECASE)
    group = m.group(1) if m else "unknown"
    role = m.group(2).upper() if m else "unknown"
    print(f"[DEBUG] {base} -> group={group}, role={role}")
    return group, role

# ============================================================
# YAML loading
# ============================================================

def load_yaml_models(paths: List[str]) -> List[ModelMeta]:
    models: List[ModelMeta] = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: failed to parse {p}: {e}")
            continue
        if isinstance(data, dict) and data:
            name = list(data.keys())[0]
            root = data[name]
        else:
            name = os.path.splitext(os.path.basename(p))[0]
            root = data if isinstance(data, dict) else {}
        group, role = guess_group_role_from_filename(p)
        models.append(ModelMeta(file=p, group=group, role=role, name=name, root=root or {}))
    return models

# ============================================================
# Generic recursive field extraction
# ============================================================

def normkey(k: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", k.strip().lower())

FIELD_ALIASES: Dict[str, List[str]] = {
    # Domain/Enterprise
    "title": ["title","name"],
    "model_version": ["model_version","version","design_version"],
    "description": ["description","summary","abstract"],
    "keywords": ["keywords","tags"],
    "model_type": ["model_type","type"],
    "scope": ["scope"],
    "purpose_pattern": ["purpose_pattern","purpose_and_pattern","purpose","pattern"],
    "assumptions": ["assumptions"],
    "links_to_publications_and_reports": ["links","publications","reports","references","links_to_publications_and_reports"],
    "conceptual_model_evaluation": ["conceptual_model_evaluation","conceptual_evaluation"],
    "calibration_tools_data": ["calibration","calibration_tools","calibration_tools_data","calibration_data"],
    "validation_capabilities": ["validation_capabilities","validation"],
    "sensitivity_analysis": ["sensitivity_analysis","sensitivity"],
    "uncertainty_analysis": ["uncertainty_analysis","uncertainty"],
    "authors_unique_identifier": ["authors_unique_identifier","authors","contributors","author_ids","orcid"],
    "contributor_role": ["contributor_role","roles"],
    # Information (model-level)
    "unique_identifier": ["unique_identifier","uuid","doi","id"],
    "parameters": ["parameters","params"],
    "datasets": ["datasets","data_sources"],
    "data": ["data","outputs","output_description"],
    "dimensionality": ["dimensionality","dims"],
    "spatial_extent_coverage": ["spatial_extent","spatial_coverage","extent","coverage"],
    "spatial_resolution": ["spatial_resolution","grid_resolution","grid_size"],
    "variable_spatial_resolution": ["variable_spatial_resolution","adaptive_spatial"],
    "temporal_extent_coverage": ["temporal_extent","temporal_coverage","time_period","time_extent"],
    "time_steps_temporal_resolution": ["time_steps","temporal_resolution","time_step","timestep"],
    "variable_temporal_resolution": ["variable_temporal_resolution","adaptive_temporal"],
    "resampling_policies": ["resampling","conversion_policies","resampling_policies","per_variable_resampling"],
    "input": ["input"],
    "output": ["output"],
    "integrated_input": ["integrated_input"],
    # Computational
    "error_handling": ["error_handling","errors","exceptions"],
    "integration_pattern": ["integration.pattern","integration_pattern","pattern"],
    "communication_mechanism": ["communication","io_mechanism","interaction_style","binding","interface"],
    "execution_instructions": ["execution_instructions","run_instructions","how_to_run","runbook"],
    "acknowledgment_protocols": ["acknowledgment_protocols","ack_protocols","acknowledgement_protocols"],
    "execution_constraints": ["execution_constraints","ordering","timing_constraints"],
    # Engineering
    "parallel_execution": ["parallel_execution","concurrency","parallelism"],
    "latency_expectations": ["latency_expectations","latency","sla"],
    "data_synchronization": ["data_synchronization","sync_strategy","synchronization"],
    # Technology
    "programming_language": ["programming_language","language","lang"],
    "availability_of_source_code": ["availability_of_source_code","source_code","repo","repository"],
    "implementation_verification": ["implementation_verification","tests","verification"],
    "software_specification_and_requirements": ["software_specification_and_requirements","software_requirements","software_stack","dependencies"],
    "hardware_specification_and_requirements": ["hardware_specification_and_requirements","hardware_requirements","hardware","resources"],
    "license": ["license","licence"],
    "landing_page": ["landing_page","homepage","url"],
    "distribution_version": ["distribution_version","package_version","release_version"],
    "file_formats": ["file_formats","formats","file_format"],

}

def dig_for_keypaths(d: Any, path: str) -> List[Any]:
    parts = path.split(".")
    cur = d
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return []
    return [cur]

def collect_values(root: Dict[str, Any], alias_list: List[str]) -> List[Any]:
    vals: List[Any] = []
    root = root or {}
    for ali in alias_list:
        if "." in ali:
            vals.extend(dig_for_keypaths(root, ali))
        elif ali in root:
            vals.append(root[ali])
        nk_target = normkey(ali)
        def rec(x: Any):
            if x is None:
                return
            if isinstance(x, dict):
                for k, v in x.items():
                    if normkey(k) == nk_target:
                        vals.append(v)
                    rec(v)
            elif isinstance(x, list):
                for it in x:
                    rec(it)
        rec(root)
    return vals

def extract_field(model: ModelMeta, key: str) -> List[str]:
    alis = FIELD_ALIASES.get(key, [key])
    vals = collect_values(model.root, alis)
    flat: List[str] = []
    for v in vals:
        if v is None:
            continue
        if key in ("input","output","integrated_input"):
            continue  # IO handled elsewhere
        if isinstance(v, dict):
            for vv in v.values():
                if isinstance(vv, (str,int,float)):
                    flat.append(str(vv))
        elif isinstance(v, list):
            for it in v:
                if it is None:
                    continue
                if isinstance(it, (str,int,float)):
                    flat.append(str(it))
                elif isinstance(it, dict):
                    for vv in it.values():
                        if isinstance(vv, (str,int,float)):
                            flat.append(str(vv))
        elif isinstance(v, (str,int,float)):
            flat.append(str(v))
    return dedupe([normalize_phrase(x) for x in flat if str(x).strip()])

# ============================================================
# IO extraction
# ============================================================

def coerce_var_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return split_variables(v)
    if isinstance(v, list):
        out: List[str] = []
        for item in v:
            if item is None:
                continue
            if isinstance(item, str):
                out.extend(split_variables(item))
            elif isinstance(item, dict):
                for k in ("name","variable","var","id","description","label"):
                    if isinstance(item.get(k), str):
                        out.extend(split_variables(item[k]))
        return dedupe(out)
    if isinstance(v, dict):
        out: List[str] = []
        for k in ("variables","names","list","description"):
            if k in v and isinstance(v[k], (str,list)):
                out.extend(coerce_var_list(v[k]))
        return dedupe(out)
    return []

def coerce_units_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [u.strip() for u in re.split(r"[;,]", v) if u.strip()]
    if isinstance(v, list):
        return [str(u).strip() for u in v if (u is not None and str(u).strip())]
    if isinstance(v, dict):
        out = []
        for k in ("units","unit"):
            if k in v:
                out.extend(coerce_units_list(v[k]))
        return out
    return []

def parse_ab_pattern(r: Dict[str, Any]) -> str:
    integ = r.get("integration")
    if isinstance(integ, dict):
        pat = integ.get("pattern")
        if isinstance(pat, str) and pat.strip():
            txt = pat.lower()
            if "embedded" in txt: return "Embedded"
            if "integrated" in txt or "tight" in txt: return "Integrated"
            if "shared" in txt: return "Shared"
            if "loose" in txt: return "Loose"
            if "one-way" in txt or "one way" in txt: return "One-Way"
            return pat.strip()
    # Fallback keyword inference if not explicitly set
    text = normalize_phrase(" ".join([str(v) for v in r.values() if isinstance(v, str)]))
    if "in memory" in text or "in-memory" in text or "shared memory" in text: return "Embedded"
    if "synchronous" in text or "every step" in text or "same timestep" in text: return "Integrated"
    if "shared repository" in text or "database" in text or "object store" in text: return "Shared"
    if "netcdf" in text or "csv" in text or "file" in text: return "Loose"
    return ""

def extract_io(model: ModelMeta, debug: bool = False) -> None:
    r = model.root if isinstance(model.root, dict) else {}
    # Outputs
    out_vars, out_units = [], []
    if isinstance(r.get("output"), dict):
        oo = r["output"]
        out_vars = coerce_var_list(oo.get("variables"))
        out_units = coerce_units_list(oo.get("units"))
    if not out_vars and isinstance(r.get("data"), dict):
        d = r["data"]
        out_vars = coerce_var_list(d.get("variables"))
        out_units = coerce_units_list(d.get("units"))

    # Inputs
    in_vars, in_units = [], []
    if isinstance(r.get("input"), dict):
        ii = r["input"]
        in_vars.extend(coerce_var_list(ii.get("variables")))
        in_units.extend(coerce_units_list(ii.get("units")))
        if isinstance(ii.get("description"), str):
            in_vars.extend(split_variables(ii["description"]))

    if isinstance(r.get("integrated_input"), list):
        for item in r["integrated_input"]:
            if isinstance(item, dict):
                in_vars.extend(split_variables(item.get("description", "")))

    if isinstance(r.get("datasets"), list):
        for ds in r["datasets"]:
            if isinstance(ds, dict) and isinstance(ds.get("dependencies"), list):
                for dep in ds["dependencies"]:
                    if isinstance(dep, str):
                        in_vars.extend(split_variables(dep))

    model.outputs = IOSchema(variables=dedupe(out_vars), units=dedupe(out_units))
    model.inputs  = IOSchema(variables=dedupe(in_vars), units=dedupe(in_units))

    # AB extras
    if model.role in {"AB-INTENDED", "AB-INTEGRATED"}:
        model.ab_pattern = parse_ab_pattern(r)
        if isinstance(r.get("input"), dict):
            model.ab_in = dedupe(coerce_var_list(r["input"].get("variables")))
        if isinstance(r.get("output"), dict):
            model.ab_out = dedupe(coerce_var_list(r["output"].get("variables")))
        canon: List[str] = []
        if isinstance(r.get("integrated_input"), list):
            for item in r["integrated_input"]:
                if isinstance(item, dict):
                    canon.extend(split_variables(item.get("description", "")))
        canon.extend(model.ab_in)
        canon.extend(model.ab_out)
        if not canon:
            for k in ("description", "purpose_pattern", "integration"):
                val = r.get(k, "")
                if isinstance(val, dict):
                    for vv in val.values():
                        if isinstance(vv, str):
                            canon.extend(split_variables(vv))
                elif isinstance(val, str):
                    canon.extend(split_variables(val))
        model.canonical_exchange = dedupe(canon)

    # Populate generic fields bag
    for k in FIELD_ALIASES.keys():
        if k in ("input","output","integrated_input"):
            continue
        model.fields[k] = extract_field(model, k)

    if debug:
        print(f"\n[DEBUG] {model.file} ({model.role})")
        print(f"  outputs: {model.outputs.variables or '(none)'}")
        print(f"  inputs:  {model.inputs.variables or '(none)'}")
        if model.role in {"AB-INTENDED", "AB-INTEGRATED"}:
            print(f"  AB in:   {model.ab_in or '(none)'}")
            print(f"  AB out:  {model.ab_out or '(none)'}")
            print(f"  AB pat:  {model.ab_pattern or '(unspecified)'}")
            print(f"  AB canon:{model.canonical_exchange or '(none)'}")

# ============================================================
# Pattern inference (simple heuristics; AB overrides)
# ============================================================

INMEMORY_KEYS = ["in-memory", "in memory", "shared memory"]
TOOL_KEYS = ["fabm", "esmf", "oasis", "oasis-mct", "mct", "framework", "mediator", "coupler"]
SYNC_KEYS = ["every physics step", "every step", "synchronous", "same timestep", "same time step"]
FILE_KEYS = ["netcdf", "csv", "file", "files", "post-run", "offline", "batch"]
REPO_KEYS = ["shared repository", "common data store", "database", "object store"]

def text_from_fields(*vals: Any) -> str:
    parts: List[str] = []
    for v in vals:
        if v is None:
            continue
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, list):
            for it in v:
                if it is None:
                    continue
                if isinstance(it, dict):
                    parts.extend([str(x) for x in it.values() if isinstance(x, str)])
                elif isinstance(it, str):
                    parts.append(it)
        elif isinstance(v, dict):
            for x in v.values():
                if isinstance(x, str):
                    parts.append(x)
    return normalize_phrase(" ".join(parts))

def contains_any(text: str, keys: List[str]) -> bool:
    t = text.lower()
    return any(k in t for k in keys)

def detect_integration_pattern(a: ModelMeta, b: ModelMeta, ab: Optional[ModelMeta]) -> Tuple[str, str]:
    if ab and ab.ab_pattern:
        if ab.ab_pattern.lower() not in {"one-way","loose","shared","integrated","embedded"}:
            return "Shared", f"AB declared '{ab.ab_pattern}' → mapped to 'Shared'."
        return ab.ab_pattern, f"AB declares pattern '{ab.ab_pattern}'."
    a_text = text_from_fields(a.root)
    b_text = text_from_fields(b.root)
    ab_text = text_from_fields(ab.root if ab else {})
    union_text = " ".join([a_text, b_text, ab_text])

    has_inmem = contains_any(union_text, INMEMORY_KEYS)
    has_tool  = contains_any(union_text, TOOL_KEYS)
    has_sync  = contains_any(union_text, SYNC_KEYS)
    has_file  = contains_any(union_text, FILE_KEYS)
    has_repo  = contains_any(union_text, REPO_KEYS)

    if has_inmem:
        return "Embedded", "Detected in-memory / shared memory phrasing."
    if has_sync and (has_repo or has_tool):
        return "Shared", "Detected synchronous cadence with shared store/coupler."
    if has_file or has_repo or has_tool:
        return "Loose", "Detected file/repo/coupler exchange without in-memory."
    return "One-Way", "Defaulted to One-Way."

# ============================================================
# Helpers for row building
# ============================================================

def row(group: str, bottleneck: str, field: str, pattern: str, required_check: str,
        av: str, bv: str, abv: str, detail: str, result: str) -> Dict[str,Any]:
    return {
        "group": group,
        "bottleneck": bottleneck,
        "field": field,
        "pattern": pattern or "(unspecified)",
        "required_check": required_check,
        "A_value": av or "",
        "B_value": bv or "",
        "AB_value": abv or "",
        "detail": detail or "",
        "result": result
    }

def join_field_values(model: Optional[ModelMeta], key: str) -> str:
    if not model:
        return ""
    return "; ".join(model.fields.get(key, []))

def missing(values: List[str]) -> bool:
    return not any(v.strip() for v in values)

def pretty_pairs(pairs: List[Tuple[str,str,float]]) -> str:
    if not pairs:
        return ""
    return "; ".join([f"{s} ↔ {d} ({sc:.2f})" for (s,d,sc) in pairs])

# ============================================================
# --------- PHASE 1: Conceptual & Legal (table-aligned) -------
# ============================================================

def check_phase1_conceptual_and_legal(
    group: str, a: ModelMeta, b: ModelMeta, ab: Optional[ModelMeta], pattern: str
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    # ---- tunables ----
    CC_MIN_PRESENT_RATIO = 0.30   # require at least 50% of fields jointly present (A,B,AB)
    CC_SIM_THRESHOLD     = 0   # average Jaccard threshold for Match

    def add_row(check: str, subtype: str, av: str, bv: str, abv: str, result: str, reason: str):
        rows.append({
            "phase": "Phase 1",
            "check": check,
            "subtype": subtype,
            "pattern": pattern,
            "group": group,
            "model_a": a.name,
            "model_b": b.name,
            "tokens_a": av,
            "tokens_b": bv,
            "tokens_ab": abv,
            "result": result,
            "reason": reason,
        })

    # ------------------------------------------------------------------
    # 1) Conceptual Consistency Mismatch — majority-present + averaged sim
    # ------------------------------------------------------------------
    fields_cc = [
        ("Scope", "scope"),
        ("Purpose", "purpose_pattern"),
        ("Assumptions", "assumptions"),
        ("Validation Capabilities", "validation_capabilities"),
        ("Conceptual Model Evaluation", "conceptual_model_evaluation"),
    ]

    per_field_sims = []         # list of (label, sim or None if missing)
    present_count  = 0
    needed_count   = len(fields_cc)
    all_av, all_bv, all_abv = [], [], []
    missing_fields = []         # list of (label, which_missing) e.g. ('Scope', 'AB')

    for label, key in fields_cc:
        av = join_field_values(a, key)
        bv = join_field_values(b, key)
        abv = join_field_values(ab, key) if ab else ""
        all_av.append(av); all_bv.append(bv); all_abv.append(abv)

        # Track which model(s) are missing per field for better diagnostics
        missing_who = []
        if not av: missing_who.append("A")
        if not bv: missing_who.append("B")
        if not abv: missing_who.append("AB")
        if missing_who:
            per_field_sims.append((label, None))
            missing_fields.append((label, ",".join(missing_who)))
            continue

        # All three present → compute similarity of (A ∪ B) vs AB
        sim_union = jaccard_token_similarity(f"{av} {bv}", abv)
        per_field_sims.append((label, sim_union))
        present_count += 1

    present_ratio = (present_count / needed_count) if needed_count else 0.0
    if present_ratio < CC_MIN_PRESENT_RATIO:
        # Not enough jointly present fields → Metadata Gap
        reason_bits = [f"{present_count}/{needed_count} fields jointly present (< {CC_MIN_PRESENT_RATIO:.0%})"]
        if missing_fields:
            # include a compact list of which fields are missing where
            missing_str = "; ".join([f"{lbl}→{who}" for (lbl, who) in missing_fields])
            reason_bits.append(f"Missing: {missing_str}")
        add_row(
            "Conceptual Consistency Mismatch",
            "Scope, Purpose, Assumptions, Validation, Evaluation",
            "; ".join(all_av), "; ".join(all_bv), "; ".join(all_abv),
            "Metadata Gap", " | ".join(reason_bits)
        )
    else:
        # Enough present → average similarity across present fields
        sims_present = [s for (_, s) in per_field_sims if s is not None]
        avg_sim = (sum(sims_present) / len(sims_present)) if sims_present else 0.0
        result = "Match" if avg_sim >= CC_SIM_THRESHOLD else "Mismatch"
        # optional: include the per-field sims in reason (compact)
        per_field_txt = "; ".join([f"{lbl}={s:.2f}" for (lbl, s) in per_field_sims if s is not None])
        reason = f"Avg conceptual similarity = {avg_sim:.2f} (threshold {CC_SIM_THRESHOLD:.2f}); fields used: {present_count}/{needed_count}. " \
                 + (f"Sims: {per_field_txt}" if per_field_txt else "")
        add_row(
            "Conceptual Consistency Mismatch",
            "Scope, Purpose, Assumptions, Validation, Evaluation",
            "; ".join(all_av), "; ".join(all_bv), "; ".join(all_abv),
            result, reason
        )

    # ------------------------------------------------------------------
    # 2) Provenance and Accessibility Gap — presence-based (kept)
    # ------------------------------------------------------------------
    prov_fields = [
        ("Model ID", "model_identifier"),
        ("Landing Page", "landing_page"),
        ("Source Code Availability", "source_code_availability"),
    ]
    has_gap = False
    all_av, all_bv, all_abv = [], [], []
    for _, key in prov_fields:
        av = join_field_values(a, key); all_av.append(av)
        bv = join_field_values(b, key); all_bv.append(bv)
        abv = join_field_values(ab, key) if ab else ""; all_abv.append(abv)
        if not av or not bv or not abv:
            has_gap = True
    add_row(
        "Provenance and Accessibility Gap",
        "Model ID, Landing Page, Source Code Availability",
        "; ".join(all_av), "; ".join(all_bv), "; ".join(all_abv),
        "Metadata Gap" if has_gap else "Match",
        "Missing provenance info in one or more models." if has_gap else "All models provide provenance metadata."
    )

    # ------------------------------------------------------------------
    # 3) Conceptual Quality Gap — presence-based (kept)
    # ------------------------------------------------------------------
    fields_q = [
        ("Publications & Reports", "links_to_publications_and_reports"),
        ("Authors’ ID", "authors_unique_identifier"),
        ("Conceptual Model Evaluation", "conceptual_model_evaluation"),
        ("Calibration Data", "calibration_tools_data"),
        ("Validation Capabilities", "validation_capabilities"),
        ("Sensitivity Analysis", "sensitivity_analysis"),
        ("Uncertainty Analysis", "uncertainty_analysis"),
    ]
    has_gap = False
    all_av, all_bv, all_abv = [], [], []
    for _, key in fields_q:
        av = join_field_values(a, key); all_av.append(av)
        bv = join_field_values(b, key); all_bv.append(bv)
        abv = join_field_values(ab, key) if ab else ""; all_abv.append(abv)
        if not av or not bv or not abv:
            has_gap = True
    add_row(
        "Conceptual Quality Gap",
        "Publications, IDs, Evaluation, Calibration, Validation, Sensitivity, Uncertainty",
        "; ".join(all_av), "; ".join(all_bv), "; ".join(all_abv),
        "Metadata Gap" if has_gap else "Match",
        "One or more models missing quality/validation evidence." if has_gap else "All models provide quality/validation metadata."
    )

    # ------------------------------------------------------------------
    # 4) License Incompatibility
    # ------------------------------------------------------------------
    LICENSE_ALIASES = {
        "mit": "Permissive",
        "bsd": "Permissive",
        "apache": "Permissive",
        "mpl": "Weak Copyleft",
        "lgpl": "Weak Copyleft",
        "gpl": "Strong Copyleft",
        "agpl": "Strong Copyleft",
        "proprietary": "Proprietary",
        "commercial": "Proprietary",
        "unknown": "Unknown",
    }

    LICENSE_COMPATIBILITY = {
        "Permissive": ["Permissive", "Weak Copyleft", "Strong Copyleft", "Proprietary"],
        "Weak Copyleft": ["Permissive", "Weak Copyleft"],
        "Strong Copyleft": ["Strong Copyleft"],
        "Proprietary": ["Proprietary"],
    }

    def normalize_license(text: str) -> str:
        if not text:
            return "Unknown"
        t = text.lower()
        for k, v in LICENSE_ALIASES.items():
            if k in t:
                return v
        return "Unknown"

    av = join_field_values(a, "license")
    bv = join_field_values(b, "license")
    abv = join_field_values(ab, "license") if ab else ""

    if not av or not bv or not abv:
        add_row(
            "License Incompatibility",
            "License",
            av, bv, abv,
            "Metadata Gap",
            "Missing license information in one or more models."
        )
    else:
        cat_a = normalize_license(av)
        cat_b = normalize_license(bv)
        cat_ab = normalize_license(abv)

        compatible = (
                cat_b in LICENSE_COMPATIBILITY.get(cat_a, []) and
                cat_a in LICENSE_COMPATIBILITY.get(cat_b, []) and
                cat_a in LICENSE_COMPATIBILITY.get(cat_ab, []) and
                cat_b in LICENSE_COMPATIBILITY.get(cat_ab, [])
        )

        if compatible:
            result = "Match"
            reason = f"Licenses are compatible ({cat_a}, {cat_b} → {cat_ab})."
        else:
            result = "Mismatch"
            reason = f"Incompatible license families ({cat_a}, {cat_b} → {cat_ab})."

        add_row("License Incompatibility", "License", av, bv, abv, result, reason)

    return rows


def check_phase1_conceptual_and_legal_weighted(
    group: str, a: ModelMeta, b: ModelMeta, ab: Optional[ModelMeta], pattern: str
) -> List[Dict[str, Any]]:
    """
    Phase 1 — Conceptual and Legal Consistency (Weighted Ratios)
    ------------------------------------------------------------
    Instead of categorical results (Match/Mismatch/Gap), returns proportional
    ratios for each type, normalized to sum = 1.0.

    For each check:
      - match_ratio, mismatch_ratio, gap_ratio ∈ [0,1]
      - match_ratio + mismatch_ratio + gap_ratio = 1
    """

    rows: List[Dict[str, Any]] = []

    # --- helper to normalize ratios ---
    def normalize(match, mismatch, gap):
        total = match + mismatch + gap
        if total == 0:
            return (0.0, 0.0, 1.0)
        match /= total
        mismatch /= total
        gap /= total
        return (round(match, 3), round(mismatch, 3), round(gap, 3))

    # --- helper to add a row ---
    def add_row(check, subtype, ratios, reason):
        rows.append({
            "phase": "Phase 1",
            "check": check,
            "subtype": subtype,
            "pattern": pattern,
            "group": group,
            "model_a": a.name,
            "model_b": b.name,
            "match_ratio": ratios[0],
            "mismatch_ratio": ratios[1],
            "gap_ratio": ratios[2],
            "reason": reason,
        })

    # ------------------------------------------------------------------
    # 1️⃣ Conceptual Consistency Mismatch — majority-present + averaged sim
    # ------------------------------------------------------------------
    fields_cc = [
        ("Scope", "scope"),
        ("Purpose", "purpose_pattern"),
        ("Assumptions", "assumptions"),
        #("Validation Capabilities", "validation_capabilities"),
        #("Conceptual Model Evaluation", "conceptual_model_evaluation"),
    ]

    sims, missing = [], 0
    for label, key in fields_cc:
        av, bv = join_field_values(a, key), join_field_values(b, key)
        abv = join_field_values(ab, key) if ab else ""
        if not av or not bv or not abv:
            missing += 1
            continue
        sim_union = jaccard_token_similarity(f"{av} {bv}", abv)
        sims.append(sim_union)

    total = len(fields_cc)
    present = total - missing
    avg_sim = sum(sims) / len(sims) if sims else 0.0

    # weighted proportions
    gap_ratio = missing / total
    match_ratio = avg_sim * (present / total)
    mismatch_ratio = max(0.0, 1.0 - (gap_ratio + match_ratio))
    ratios = normalize(match_ratio, mismatch_ratio, gap_ratio)

    reason = (
        f"Fields={total}, Present={present}, Missing={missing}. "
        f"Avg sim={avg_sim:.2f}. → Match={ratios[0]:.2f}, "
        f"Mismatch={ratios[1]:.2f}, Gap={ratios[2]:.2f}"
    )
    add_row("Conceptual Consistency Mismatch",
            "Scope, Purpose, Assumptions, Validation, Evaluation",
            ratios, reason)

    # ------------------------------------------------------------------
    # 2️⃣ Provenance and Accessibility Gap — presence-based
    # ------------------------------------------------------------------
    prov_fields = [
        #("Model ID", "model_identifier"),
        ("Landing Page", "landing_page"),
        ("Source Code Availability", "source_code_availability"),
    ]

    total = len(prov_fields)
    missing = 0
    for _, key in prov_fields:
        av, bv, abv = join_field_values(a, key), join_field_values(b, key), join_field_values(ab, key) if ab else ""
        if not av or not bv or not abv:
            missing += 1

    present = total - missing
    gap_ratio = missing / total
    match_ratio = present / total
    mismatch_ratio = 0.0
    ratios = normalize(match_ratio, mismatch_ratio, gap_ratio)
    reason = f"Fields={total}, Present={present}, Missing={missing}. " \
             f"→ Match={ratios[0]:.2f}, Gap={ratios[2]:.2f}"

    add_row("Provenance and Accessibility Gap",
            "Model ID, Landing Page, Source Code Availability",
            ratios, reason)

    # ------------------------------------------------------------------
    # 3️⃣ Conceptual Quality Gap — presence-based
    # ------------------------------------------------------------------
    fields_q = [
        ("Publications & Reports", "links_to_publications_and_reports"),
        #("Authors’ ID", "authors_unique_identifier"),
        #("Conceptual Model Evaluation", "conceptual_model_evaluation"),
        #("Calibration Data", "calibration_tools_data"),
        #("Validation Capabilities", "validation_capabilities"),
        #("Sensitivity Analysis", "sensitivity_analysis"),
        #("Uncertainty Analysis", "uncertainty_analysis"),
    ]

    total = len(fields_q)
    missing = 0
    for _, key in fields_q:
        av, bv, abv = join_field_values(a, key), join_field_values(b, key), join_field_values(ab, key) if ab else ""
        if not av or not bv or not abv:
            missing += 1

    present = total - missing
    gap_ratio = missing / total
    match_ratio = present / total
    mismatch_ratio = 0.0
    ratios = normalize(match_ratio, mismatch_ratio, gap_ratio)
    reason = f"Fields={total}, Present={present}, Missing={missing}. " \
             f"→ Match={ratios[0]:.2f}, Gap={ratios[2]:.2f}"

    add_row("Conceptual Quality Gap",
            "Publications, IDs, Evaluation, Calibration, Validation, Sensitivity, Uncertainty",
            ratios, reason)
    # ============================================================
    # 4️⃣ License Incompatibility (Weighted version)
    # ============================================================
    LICENSE_ALIASES = {
        "mit": "Permissive",
        "bsd": "Permissive",
        "apache": "Permissive",
        "mpl": "Weak Copyleft",
        "lgpl": "Weak Copyleft",
        "gpl": "Strong Copyleft",
        "agpl": "Strong Copyleft",
        "proprietary": "Proprietary",
        "commercial": "Proprietary",
        "unknown": "Unknown",
    }

    LICENSE_COMPATIBILITY = {
        "Permissive": ["Permissive", "Weak Copyleft", "Strong Copyleft", "Proprietary"],
        "Weak Copyleft": ["Permissive", "Weak Copyleft"],
        "Strong Copyleft": ["Strong Copyleft"],
        "Proprietary": ["Proprietary"],
    }

    def normalize_license(text: str) -> str:
        if not text:
            return "Unknown"
        t = text.lower()
        for k, v in LICENSE_ALIASES.items():
            if k in t:
                return v
        return "Unknown"

    av = join_field_values(a, "license")
    bv = join_field_values(b, "license")
    abv = join_field_values(ab, "license") if ab else ""

    if not av or not bv or not abv:
        ratios = normalize(0.0, 0.0, 1.0)
        reason = "Missing license information in one or more models."
    else:
        cat_a = normalize_license(av)
        cat_b = normalize_license(bv)
        cat_ab = normalize_license(abv)

        compatible = (
                cat_b in LICENSE_COMPATIBILITY.get(cat_a, []) and
                cat_a in LICENSE_COMPATIBILITY.get(cat_b, []) and
                cat_a in LICENSE_COMPATIBILITY.get(cat_ab, []) and
                cat_b in LICENSE_COMPATIBILITY.get(cat_ab, [])
        )

        if compatible:
            ratios = normalize(1.0, 0.0, 0.0)
            reason = f"Licenses compatible ({cat_a}, {cat_b} → {cat_ab})."
        else:
            ratios = normalize(0.0, 1.0, 0.0)
            reason = f"Incompatible license families ({cat_a}, {cat_b} → {cat_ab})."

    add_row("License Incompatibility", "License", ratios, reason)

    # ============================================================



    return rows


# ============================================================
# --------- PHASE 2: Information Alignment (table-aligned) ----
# ============================================================


import json

import json

def check_phase2_information_alignment(
    group: str, a: ModelMeta, b: ModelMeta, ab: Optional[ModelMeta], pattern: str
) -> List[Dict[str, Any]]:
    """
    Phase 2 — Information Alignment Check
    -------------------------------------
    1. Active-edge (I/O) checks: based on pattern-specific data flow
       • Compares file format, semantics, units, temporal/spatial resolution.
       • Reports Gap, Match, or Mismatch per active edge (e.g., A→B).
    2. Whole-model metadata checks: model-level information consistency.
       • Checks A, B, AB presence and alignment.

    Returns a list of standardized result dictionaries.
    """
    rows: List[Dict[str, Any]] = []

    def add_row(check, subtype, av, bv, abv, result, reason):
        rows.append({
            "phase": "Phase 2",
            "check": check,
            "subtype": subtype,
            "pattern": pattern,
            "group": group,
            "model_a": a.name,
            "model_b": b.name,
            "tokens_a": av,
            "tokens_b": bv,
            "tokens_ab": abv,
            "result": result,
            "reason": reason,
        })

    # ------------------------------------------------------------------
    # 🧩 1️⃣ ACTIVE EDGE CHECKS — depend on pattern
    # ------------------------------------------------------------------
    patt = (pattern or "").strip()
    edges = []
    if patt == "One-Way":
        edges.append(("A→B", a.outputs.variables, b.inputs.variables))
    elif patt in ("Loose", "Shared"):
        edges.extend([
            ("A→B", a.outputs.variables, b.inputs.variables),
            ("B→A", b.outputs.variables, a.inputs.variables),
        ])
    else:  # Integrated / Embedded
        edges.extend([
            ("A→B", a.outputs.variables, b.inputs.variables),
            ("B→A", b.outputs.variables, a.inputs.variables),
        ])

    # --- helper to compare data-field pairs ---
    def data_field_match(field_a: str, field_b: str) -> bool:
        return field_a.strip().lower() == field_b.strip().lower() if field_a and field_b else False

    for elabel, src_out, dst_in in edges:
        # Gather field-level metadata from both sides
        src_meta = {
            "format": join_field_values(a, "file_format"),
            #"semantics": join_field_values(a, "variable_semantics"),
            #"unit": join_field_values(a, "variable_unit"),
            #"temporal": join_field_values(a, "time_steps_temporal_resolution"),
            #"spatial": join_field_values(a, "spatial_resolution"),
        }
        dst_meta = {
            "format": join_field_values(b, "file_format"),
            #"semantics": join_field_values(b, "variable_semantics"),
            #"unit": join_field_values(b, "variable_unit"),
            #"temporal": join_field_values(b, "time_steps_temporal_resolution"),
            #"spatial": join_field_values(b, "spatial_resolution"),
        }

        # (1) If any required field missing → Metadata Gap
        if any(not v for v in src_meta.values()) or any(not v for v in dst_meta.values()):
            add_row(
                "Information Alignment Mismatch",
                f"Active Edge {elabel}",
                json.dumps(src_meta, ensure_ascii=False),
                json.dumps(dst_meta, ensure_ascii=False),
                "",
                "Metadata Gap",
                "Missing one or more required metadata fields (format, semantics, unit, temporal/spatial resolution).",
            )
            continue

        # (2) Check equality of all five attributes
        field_matches = {
            k: data_field_match(src_meta[k], dst_meta[k]) for k in src_meta.keys()
        }
        if all(field_matches.values()):
            result, reason = "Match", "All I/O metadata fields match."
        else:
            mismatched = [k for k, ok in field_matches.items() if not ok]
            result = "Mismatch"
            reason = f"Mismatched fields: {', '.join(mismatched)}."

        add_row(
            "Information Alignment Mismatch",
            f"Active Edge {elabel}",
            json.dumps(src_meta, ensure_ascii=False),
            json.dumps(dst_meta, ensure_ascii=False),
            "",
            result,
            reason,
        )

    # ------------------------------------------------------------------
    # 🧠 2️⃣ WHOLE-MODEL METADATA CHECKS
    # ------------------------------------------------------------------
    info_fields = [
        ("Dimensionality", "dimensionality"),
        ("Spatial Resolution", "spatial_resolution"),
        ("Variable Spatial Resolution", "variable_spatial_resolution"),
        ("Time Steps / Temporal Resolution", "time_steps_temporal_resolution"),
        ("Variable Temporal Resolution", "variable_temporal_resolution"),
        #("Resampling / Conversion Policies", "resampling_policies"),
    ]

    for label, key in info_fields:
        av = join_field_values(a, key)
        bv = join_field_values(b, key)
        abv = join_field_values(ab, key) if ab else ""

        # (a) Gap check: if any missing → Metadata Gap
        if not av or not bv or not abv:
            add_row(
                "Information Alignment Mismatch",
                label,
                av, bv, abv,
                "Metadata Gap",
                "One or more models missing required information metadata.",
            )
            continue

        # (b) Alignment check: A↔AB and B↔AB
        sim_a = jaccard_token_similarity(av, abv)
        sim_b = jaccard_token_similarity(bv, abv)
        if sim_a > 0 and sim_b > 0:
            result, reason = "Match", f"A↔AB={sim_a:.2f}, B↔AB={sim_b:.2f} (aligned)"
        else:
            result, reason = "Mismatch", f"A↔AB={sim_a:.2f}, B↔AB={sim_b:.2f} (misaligned)"

        add_row(
            "Information Alignment Mismatch",
            label,
            av, bv, abv,
            result, reason,
        )

    return rows




def check_phase2_information_alignment_weighted(
    group: str, a: ModelMeta, b: ModelMeta, ab: Optional[ModelMeta], pattern: str
) -> List[Dict[str, Any]]:
    """
    Phase 2 — Information Alignment (Weighted Version)
    --------------------------------------------------
    Evaluates alignment across multiple fields and returns a
    weighted composition of Match / Mismatch / Gap proportions.

    Each check contributes equally unless weighted otherwise.

    Output example:
      {
        "phase": "Phase 2",
        "check": "Information Alignment Weighted",
        "pattern": "One-Way",
        "group": "G4",
        "model_a": "A_model",
        "model_b": "B_model",
        "match_ratio": 0.50,
        "mismatch_ratio": 0.30,
        "gap_ratio": 0.20,
        "reason": "...summary..."
      }
    """
    rows: List[Dict[str, Any]] = []

    patt = (pattern or "").strip()

    # Define I/O-related fields (active edge)
    io_fields = ["file_format"
                 #,"variable_semantics"
                 #,"variable_unit"
                 #,"time_steps_temporal_resolution"
                 #,"spatial_resolution"
                 ]

    # Define model-level metadata fields
    model_fields = [
        #"model_identifier",
        #"parameters",
        "dimensionality",
        "spatial_resolution",
        #"variable_spatial_resolution",
        "time_steps_temporal_resolution"
        #"variable_temporal_resolution",
        #"resampling_policies"
         ]

    # Total number of factors to normalize weights
    total_fields = len(io_fields) + len(model_fields)
    if total_fields == 0:
        return []

    match_count = 0
    mismatch_count = 0
    gap_count = 0
    details = []

    # --- I/O fields (edge check) ---
    for key in io_fields:
        av = join_field_values(a, key)
        bv = join_field_values(b, key)
        if not av or not bv:
            gap_count += 1
            details.append(f"{key}: missing")
            continue
        if av.strip().lower() == bv.strip().lower():
            match_count += 1
            details.append(f"{key}: match")
        else:
            mismatch_count += 1
            details.append(f"{key}: mismatch")

    # --- Model-level fields (whole metadata check) ---
    for key in model_fields:
        av = join_field_values(a, key)
        bv = join_field_values(b, key)
        abv = join_field_values(ab, key) if ab else ""
        if not av or not bv or not abv:
            gap_count += 1
            details.append(f"{key}: missing (gap)")
            continue
        sim_a = jaccard_token_similarity(av, abv)
        sim_b = jaccard_token_similarity(bv, abv)
        if sim_a > 0 and sim_b > 0:
            match_count += 1
            details.append(f"{key}: aligned ({sim_a:.2f}/{sim_b:.2f})")
        else:
            mismatch_count += 1
            details.append(f"{key}: misaligned ({sim_a:.2f}/{sim_b:.2f})")

    # Normalize weights
    match_ratio = round(match_count / total_fields, 3)
    mismatch_ratio = round(mismatch_count / total_fields, 3)
    gap_ratio = round(gap_count / total_fields, 3)

    # Ensure sum = 1.0 (minor floating corrections)
    total_ratio = match_ratio + mismatch_ratio + gap_ratio
    if total_ratio != 1.0:
        correction = 1.0 - total_ratio
        match_ratio += correction  # small adjustment to keep sum=1

    reason = f"Match={match_ratio:.2f}, Mismatch={mismatch_ratio:.2f}, Gap={gap_ratio:.2f} " \
             f"based on {total_fields} factors. " + " | ".join(details)

    rows.append({
        "phase": "Phase 2",
        "check": "Information Alignment Mismatch",
        "subtype": "Weighted Composition",
        "pattern": pattern,
        "group": group,
        "model_a": a.name,
        "model_b": b.name,
        "match_ratio": match_ratio,
        "mismatch_ratio": mismatch_ratio,
        "gap_ratio": gap_ratio,
        "reason": reason,
    })

    return rows


# ============================================================
# --------- PHASE 3: Runtime checks (kept as-is) --------------
# ============================================================

def _pattern_level(p: str) -> int:
    order = {"One-Way": 1, "Loose": 2, "Shared": 3, "Integrated": 4, "Embedded": 5}
    return order.get(p or "", 3)

_TIGHT_TOKENS = ["in-memory", "in memory", "shared memory", "direct call", "api", "ffi"]
_SYNC_TOKENS  = ["synchronous", "lockstep", "barrier", "every step", "same timestep", "same time step"]
_LOWLAT_TOKENS = ["low latency", "substep", "sub-stepping", "real-time", "tight coupling"]

def _has_any(text: str, tokens: list[str]) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(tok in t for tok in tokens)

def _join_nonempty(*xs: str) -> str:
    return " | ".join([x for x in xs if x])

def _presence(a: str, b: str, ab: str) -> tuple[bool, bool, bool]:
    return bool(a), bool(b), bool(ab)

# ----- Computational -----

def check_computational(group: str, a: ModelMeta, b: ModelMeta, ab: Optional[ModelMeta], pattern: str) -> List[Dict[str,Any]]:
    rows: List[Dict[str,Any]] = []
    lvl = _pattern_level(pattern)

    # Communication mechanism
    key = "communication_mechanism"
    av = "; ".join(a.fields.get(key, []))
    bv = "; ".join(b.fields.get(key, []))
    abv = "; ".join((ab.fields.get(key, []) if ab else []))
    a_has, b_has, ab_has = _presence(av, bv, abv)
    tight_present = _has_any(_join_nonempty(av, bv, abv), _TIGHT_TOKENS)
    sync_present  = _has_any(_join_nonempty(av, bv, abv), _SYNC_TOKENS)
    if   lvl == 1:
        res, det = "Match", "Not required for One-Way."
    elif lvl == 2:
        ok = (a_has and b_has) or ab_has
        res, det = ("Match" if ok else "Missing","Both sides or AB canonical interface required.")
    elif lvl == 3:
        ok = (a_has and b_has)
        res, det = ("Match" if ok else "Missing","Both sides must declare mechanism.")
    elif lvl == 4:
        if not (a_has and b_has):
            res, det = "Missing", "Integrated requires both sides to declare mechanism."
        else:
            res, det = ("Match" if tight_present else "Mismatch","Expect tight/API/in-memory binding for Integrated.")
    else:
        required = a_has and b_has and ab_has
        if not required:
            res, det = "Missing","Embedded requires A, B, and AB declarations."
        else:
            strict_ok = tight_present and sync_present
            res, det = ("Match" if strict_ok else "Mismatch","Embedded requires in-memory/API and synchronous cues.")
    rows.append(row(group,"Communication Mechanism Mismatch",key,pattern,"Pattern-graded requirement",av,bv,abv,det,res))

    # Error handling
    key = "error_handling"
    av = "; ".join(a.fields.get(key, []))
    bv = "; ".join(b.fields.get(key, []))
    abv = "; ".join((ab.fields.get(key, []) if ab else []))
    a_has, b_has, ab_has = _presence(av, bv, abv)
    if   lvl == 1:
        ok = a_has or b_has or ab_has
        res, det = ("Match" if ok else "Missing","At least one of A/B/AB must define error handling.")
    elif lvl == 2:
        ok = (a_has and b_has) or ab_has
        res, det = ("Match" if ok else "Missing","Both A and B or AB canonical required.")
    elif lvl == 3:
        ok = a_has and b_has
        res, det = ("Match" if ok else "Missing","Shared requires A and B to define error handling.")
    elif lvl == 4:
        if not (a_has and b_has):
            res, det = "Missing","Integrated requires A and B error handling."
        else:
            if ab_has:
                simA = jaccard_token_similarity(av, abv); simB = jaccard_token_similarity(bv, abv)
                ok = (simA > 0.0 and simB > 0.0)
                res = "Match" if ok else "Mismatch"
                det = f"A↔AB={simA:.2f}, B↔AB={simB:.2f} (>0)"
            else:
                res, det = "Match","A and B present; AB not provided."
    else:
        if not (a_has and b_has and ab_has):
            res, det = "Missing","Embedded requires A, B, and AB error handling."
        else:
            simA = jaccard_token_similarity(av, abv); simB = jaccard_token_similarity(bv, abv)
            ok = (simA > 0.0 and simB > 0.0)
            res = "Match" if ok else "Mismatch"
            det = f"A↔AB={simA:.2f}, B↔AB={simB:.2f} (>0)."
    rows.append(row(group,"Error Handling Mismatch","error_handling",pattern,"Pattern-graded requirement",av,bv,abv,det,res))

    # Execution instructions
    key = "execution_instructions"
    av = "; ".join(a.fields.get(key, []))
    bv = "; ".join(b.fields.get(key, []))
    a_has, b_has, _ = _presence(av, bv, "")
    if   lvl == 1:
        res, det = ("Match" if a_has else "Missing","Producer (A) must provide run instructions.")
        rows.append(row(group,"Execution Instruction Gap","execution_instructions",pattern,"Require A only", av,"","", det, res))
        rows.append(row(group,"Execution Instruction Gap","execution_instructions",pattern,"B optional in One-Way","",bv,"","B optional","Match"))
    elif lvl == 2:
        ok = a_has or b_has
        rows.append(row(group,"Execution Instruction Gap","execution_instructions",pattern,"Require A or B", av,bv,"","At least one side must provide.","Match" if ok else "Missing"))
    else:
        ok = a_has and b_has
        rows.append(row(group,"Execution Instruction Gap","execution_instructions",pattern,"Require both sides", av,bv,"","Both A and B must provide.","Match" if ok else "Missing"))

    return rows

# ----- Engineering -----

def check_engineering(group: str, a: ModelMeta, b: ModelMeta, ab: Optional[ModelMeta], pattern: str) -> List[Dict[str,Any]]:
    rows: List[Dict[str,Any]] = []
    lvl = _pattern_level(pattern)

    def eval_presence(label: str, key: str, needs_sync_tokens: bool = False, needs_lowlat: bool = False):
        av = "; ".join(a.fields.get(key, []))
        bv = "; ".join(b.fields.get(key, []))
        abv = "; ".join((ab.fields.get(key, []) if ab else []))
        a_has, b_has, ab_has = _presence(av, bv, abv)
        if lvl == 1:
            res, det = "Match", "Not required for One-Way."
        elif lvl == 2:
            ok = a_has or b_has or ab_has
            res, det = ("Match" if ok else "Missing","Loose requires at least one side (or AB) to declare this.")
        elif lvl == 3:
            ok = a_has and b_has
            res, det = ("Match" if ok else "Missing","Shared requires A and B to declare this.")
        elif lvl == 4:
            if not (a_has and b_has):
                res, det = "Missing","Integrated requires A and B to declare this."
            else:
                if needs_sync_tokens or needs_lowlat:
                    text = _join_nonempty(av, bv, abv)
                    ok_sync = _has_any(text, _SYNC_TOKENS) if needs_sync_tokens else True
                    ok_lat = _has_any(text, _LOWLAT_TOKENS) if needs_lowlat else True
                    ok = ok_sync and ok_lat
                    res, det = ("Match" if ok else "Mismatch","Integrated expects explicit sync/latency cues.")
                else:
                    res, det = "Match","Both sides present."
        else:
            required = a_has and b_has and ab_has
            if not required:
                res, det = "Missing","Embedded requires A, B, and AB to declare this."
            else:
                text = _join_nonempty(av, bv, abv)
                ok_sync = _has_any(text, _SYNC_TOKENS) if needs_sync_tokens else True
                ok_lat  = _has_any(text, _LOWLAT_TOKENS) if needs_lowlat else True
                ok = ok_sync and ok_lat
                res, det = ("Match" if ok else "Mismatch","Embedded requires synchronous/low-latency cues (as applicable).")
        rows.append(row(group, label, key, pattern, "Pattern-graded requirement", av, bv, abv, det, res))

    eval_presence("Parallel Execution Incompatibility", "parallel_execution")
    eval_presence("Execution Constraint Mismatch", "execution_constraints", needs_sync_tokens=True)
    eval_presence("Acknowledgment Protocol Mismatch", "acknowledgment_protocols")
    eval_presence("Latency Expectation Mismatch", "latency_expectations", needs_lowlat=True)

    key = "data_synchronization"
    av = "; ".join(a.fields.get(key, []))
    bv = "; ".join(b.fields.get(key, []))
    abv = "; ".join((ab.fields.get(key, []) if ab else []))
    a_has, b_has, ab_has = _presence(av, bv, abv)
    if lvl == 1:
        res, det = "Match", "Not required for One-Way."
    elif lvl == 2:
        ok = a_has or b_has or ab_has
        res, det = ("Match" if ok else "Missing","Loose requires at least one declaration of synchronization.")
    elif lvl == 3:
        ok = a_has and b_has
        res, det = ("Match" if ok else "Missing","Shared requires A and B to declare synchronization.")
    elif lvl == 4:
        if not (a_has and b_has):
            res, det = "Missing","Integrated requires A and B synchronization metadata."
        else:
            ok_sync = _has_any(_join_nonempty(av, bv, abv), _SYNC_TOKENS)
            res, det = ("Match" if ok_sync else "Mismatch","Integrated expects explicit synchronous/lockstep cues.")
    else:
        required = a_has and b_has and ab_has
        if not required:
            res, det = "Missing","Embedded requires A, B, and AB synchronization metadata."
        else:
            ok_sync = _has_any(_join_nonempty(av, bv, abv), _SYNC_TOKENS)
            res, det = ("Match" if ok_sync else "Mismatch","Embedded expects explicit synchronous/lockstep cues.")
    rows.append(row(group, "Data Synchronization", key, pattern, "Pattern-graded requirement", av, bv, abv, det, res))
    return rows

# ----- Technology -----

def check_technology(group: str, a: ModelMeta, b: ModelMeta, ab: Optional[ModelMeta], pattern: str) -> List[Dict[str,Any]]:
    rows: List[Dict[str,Any]] = []
    lvl = _pattern_level(pattern)

    TH_SOFT   = 0.00
    TH_MEDIUM = 0.25
    TH_STRONG = 0.50

    def tech_compare(label: str, key: str,
                     exact_from_level: int = 5,
                     strong_from_level: int = 4,
                     required_from_level: int = 2):
        av = "; ".join(a.fields.get(key, []))
        bv = "; ".join(b.fields.get(key, []))
        abv = "; ".join((ab.fields.get(key, []) if ab else []))

        if lvl < required_from_level and not av and not bv:
            rows.append(row(group, label, key, pattern, "Not required at this pattern", av, bv, abv, "Absent but optional.", "Match"))
            return
        if not av and not bv:
            rows.append(row(group, label, key, pattern, "Presence required", av, bv, abv, "Neither provides metadata.", "Missing")); return
        if not av or not bv:
            rows.append(row(group, label, key, pattern, "Presence required", av, bv, abv, "Only one side provides metadata.", "Missing")); return

        if lvl >= exact_from_level:
            if av == bv:
                rows.append(row(group, label, key, pattern, "Exact equality required", av, bv, abv, "Equal.", "Match"))
            else:
                rows.append(row(group, label, key, pattern, "Exact equality required", av, bv, abv, "Not equal.", "Mismatch"))
            return

        thr = TH_SOFT if lvl <= 2 else (TH_MEDIUM if lvl == 3 else TH_STRONG)
        simAB = jaccard_token_similarity(av, bv)
        ok = simAB > thr

        if lvl >= strong_from_level and abv:
            simA = jaccard_token_similarity(av, abv)
            simB = jaccard_token_similarity(bv, abv)
            ok = ok and (simA > TH_MEDIUM) and (simB > TH_MEDIUM)
            det = f"A↔B={simAB:.2f} (> {thr:.2f}), A↔AB={simA:.2f} & B↔AB={simB:.2f} (> {TH_MEDIUM:.2f})"
        else:
            det = f"A↔B token-sim={simAB:.2f} (> {thr:.2f})"

        rows.append(row(group, label, key, pattern, "Pattern-graded similarity/equality", av, bv, abv, det, "Match" if ok else "Mismatch"))

    tech_compare("Programming Language Incompatibility", "programming_language",
                 exact_from_level=5, strong_from_level=4, required_from_level=2)

    def per_component_presence(label: str, key: str, who: str, required_from_level: int):
        vv = "; ".join((a if who == "A" else b).fields.get(key, []))
        if _pattern_level(pattern) == 1 and who == "B":
            rows.append(row(group, label, f"{key} ({who})", pattern, "Optional for One-Way (B)", vv, "", "", "Optional for B in One-Way.", "Match")); return
        if _pattern_level(pattern) < required_from_level and not vv:
            rows.append(row(group, label, f"{key} ({who})", pattern, "Optional at this pattern", vv, "", "", "Optional.", "Match")); return
        rows.append(row(group, label, f"{key} ({who})", pattern, "Presence required", vv, "", "", f"{who} provides metadata", "Match" if vv else "Missing"))

    per_component_presence("Source Code Availability Gap", "availability_of_source_code", "A", required_from_level=2)
    per_component_presence("Source Code Availability Gap", "availability_of_source_code", "B", required_from_level=3)
    per_component_presence("Implementation Verification Gap", "implementation_verification", "A", required_from_level=2)
    per_component_presence("Implementation Verification Gap", "implementation_verification", "B", required_from_level=3)

    tech_compare("Software Environment Mismatch", "software_specification_and_requirements",
                 exact_from_level=5, strong_from_level=4, required_from_level=2)
    tech_compare("Hardware Resource Mismatch", "hardware_specification_and_requirements",
                 exact_from_level=5, strong_from_level=4, required_from_level=2)
    tech_compare("Distribution Version Mismatch", "distribution_version",
                 exact_from_level=4, strong_from_level=4, required_from_level=2)
    tech_compare("File Format Mismatch", "file_formats",
                 exact_from_level=5, strong_from_level=4, required_from_level=2)

    key = "license"
    av = "; ".join(a.fields.get(key, []))
    bv = "; ".join(b.fields.get(key, []))
    if lvl < 2 and not av and not bv:
        rows.append(row(group, "License Incompatibility", key, pattern, "Optional at One-Way", av, bv, "", "Absent but optional.", "Match"))
    else:
        if not av and not bv:
            det, res = "Neither provides license.", "Missing"
        elif not av or not bv:
            det, res = "Only one provides license.", "Missing"
        else:
            if lvl >= 5:
                res = "Match" if av == bv else "Mismatch"
                det = "Exact equality required at Embedded."
            else:
                thr = 0.0 if lvl <= 2 else (0.25 if lvl == 3 else 0.50)
                sim = jaccard_token_similarity(av, bv)
                res = "Match" if sim > thr else "Mismatch"
                det = f"A↔B token-sim={sim:.2f} (> {thr:.2f})"
        rows.append(row(group, "License Incompatibility", key, pattern, "Pattern-graded requirement", av, bv, "", det, res))

    def landing_required(who: str) -> int:
        return 2 if who == "A" else 3

    for who, model in (("A", a), ("B", b)):
        key = "landing_page"
        vv = "; ".join(model.fields.get(key, []))
        req_from = landing_required(who)
        if lvl < req_from and not vv:
            rows.append(row(group, "Landing Page Gap", f"{key} ({who})", pattern, "Optional at this pattern", vv, "", "", "Optional.", "Match"))
        else:
            rows.append(row(group, "Landing Page Gap", f"{key} ({who})", pattern, "Presence required", vv, "", "", f"{who} provides landing page", "Match" if vv else "Missing"))

    return rows


# ============================================================
# =============== PHASE 3 — Pattern-aware Runtime ============
# ============================================================

def r3_execution_reproducibility_gap(
    group: str, a: ModelMeta, b: ModelMeta, ab: Optional[ModelMeta], pattern: str
) -> List[Dict[str, Any]]:
    """
    Phase 3 — Execution Reproducibility Gap
    ----------------------------------------------------------
    Checks whether each model individually provides sufficient
    execution metadata for reproducibility, with stricter criteria
    for tighter integration patterns.

    Progressive strictness:
        One-Way   → exec_type, entry_point
        Loose     → + build_system
        Shared    → + scheduler
        Integrated→ + command_args
        Embedded  → + environment/container metadata
    """

    # ------------------------------------------------------------------
    # Required fields by pattern (progressively stricter)
    # ------------------------------------------------------------------
    REQUIRED_FIELDS_BY_PATTERN = {
        "One-Way":       ["exec_type", "entry_point"],
        "Loose":         ["exec_type", "entry_point", "build_system"],
        "Shared":        ["exec_type", "entry_point", "build_system", "scheduler"],
        "Integrated":    ["exec_type", "entry_point", "build_system", "scheduler", "command_args"],
        "Embedded":      ["exec_type", "entry_point", "build_system", "scheduler",
                          "command_args", "environment"],
    }

    # Fallback if pattern missing
    required_fields = REQUIRED_FIELDS_BY_PATTERN.get(pattern, ["exec_type", "entry_point"])

    def check_exec_repro_fields(model: ModelMeta) -> Tuple[bool, List[str]]:
        """Return (is_complete, missing_fields) according to pattern-specific rule."""
        missing = []
        for field in required_fields:
            vals = model.fields.get(field, [])
            if not vals or not any(str(v).strip() for v in vals):
                missing.append(field)
        return (len(missing) == 0, missing)

    # ------------------------------------------------------------------
    # Evaluate both models separately (A and B)
    # ------------------------------------------------------------------
    rows = []
    for label, model in [("Model A", a), ("Model B", b)]:
        ok, missing = check_exec_repro_fields(model)
        result = "Match" if ok else "Metadata Gap"
        reason = (
            "All required execution metadata present and complete."
            if ok
            else f"Missing required fields: {', '.join(missing)}"
        )

        rows.append({
            "phase": "Phase 3",
            "check": "Execution Reproducibility Gap",
            "pattern": pattern,
            "group": group,
            "model_a": model.name,
            "model_b": None,  # single-model check
            "tokens_a": required_fields,
            "tokens_b": [],
            "families_a": [],
            "families_b": [],
            "result": result,
            "reason": reason,
        })

    return rows




# ----------------------- one way, loose -----------------------

def r3_io_library_compatibility_gap(group, a, b, ab, pattern):
    """
    Phase 3 – Runtime Integration Check
    Pattern: One-Way / Loose

    Purpose:
        Checks whether both models support compatible I/O libraries
        for the data formats they exchange (e.g., NetCDF, HDF5, CSV).

    Logic:
        - Scans each model’s declared file formats and software requirements.
        - Detects tokens that belong to either I/O libraries or front-end frameworks.
        - Maps tokens to high-level I/O families via IO_LIBRARY_FAMILIES or FRONTEND_TO_FAMILY.
        - Compares the resulting families between models:
              If both missing           → Metadata Gap
              If one missing            → Metadata Gap
              If disjoint families      → Mismatch
              If overlap                → Match
        - Returns a detailed record describing the comparison.
    """

    import re

    # ---------------------------------------------------------------------
    # Known I/O backends and front-end frameworks
    # ---------------------------------------------------------------------
    IO_LIBRARY_FAMILIES = {
        # === Scientific binary formats ===
        "NetCDF4_Family": {"netcdf4", "h5netcdf", "nco"},
        "NetCDF3_Family": {"scipy.io.netcdf", "netcdf", "ncdf"},
        "HDF5_Family": {"libhdf5", "h5py", "pytables"},

        # === Geospatial / Raster data ===
        "GDAL_Family": {"gdal", "rasterio", "ogr", "osgeo"},
        "GeoTIFF_Family": {"libtiff"},  # handled by GDAL otherwise

        # === Tabular / Delimited formats ===
        "CSV_Family": {"csv"},

        # === Structured text formats ===
        "JSON_Family": {"json", "orjson", "ujson", "rapidjson"},
        "XML_Family": {"lxml", "xmltodict", "elementtree", "minidom"},

        # === Image / raster graphics ===
        "Image_Family": {"pillow", "opencv", "scikit-image"},

        # === Spreadsheet / tabular interchange ===
        "Excel_Family": {"openpyxl", "xlrd", "pyxlsb"},

        # === Modern columnar storage ===
        "Parquet_Family": {"pyarrow", "fastparquet"},
    }

    FRONTEND_TO_FAMILY = {
        # Scientific and array data
        "xarray": {"NetCDF4_Family", "HDF5_Family"},
        "pandas": {"CSV_Family", "Excel_Family", "HDF5_Family", "Parquet_Family"},
        "dask": {"CSV_Family", "HDF5_Family", "Parquet_Family"},

        # Geospatial
        "geopandas": {"GDAL_Family"},
        "rasterio": {"GDAL_Family"},
        "rioxarray": {"GDAL_Family"},
        "earthengine-api": {"GDAL_Family"},

        # Imaging
        "scikit-image": {"Image_Family"},
        "matplotlib": {"Image_Family"},
        "opencv-python": {"Image_Family"},

        # Spreadsheet tools
        "xlwings": {"Excel_Family"},
    }

    # ---------------------------------------------------------------------
    # Helper function: extract families for one model
    # ---------------------------------------------------------------------
    def extract_families(model_dict):
        """Extract recognized I/O families from file_formats + software fields."""
        text_fields = []
        for key in ["file_formats", "software_specification_and_requirements", "distribution_version"]:
            val = model_dict.get(key)
            if isinstance(val, str):
                text_fields.append(val.lower())
            elif isinstance(val, list):
                text_fields.extend([str(x).lower() for x in val if x])
        joined = " ".join(text_fields)

        detected_families = set()
        detected_tokens = set()

        # --- Check I/O backend tokens ---
        for fam, tokens in IO_LIBRARY_FAMILIES.items():
            for token in tokens:
                if re.search(rf"\b{re.escape(token)}\b", joined):
                    detected_families.add(fam)
                    detected_tokens.add(token)

        # --- Check front-end libraries and map them to families ---
        for frontend, fams in FRONTEND_TO_FAMILY.items():
            if re.search(rf"\b{re.escape(frontend)}\b", joined):
                detected_families.update(fams)
                detected_tokens.add(frontend)

        return detected_families, detected_tokens

    # ---------------------------------------------------------------------
    # Extract families for model A and B
    # ---------------------------------------------------------------------
    fam_a, tokens_a = extract_families(a)
    fam_b, tokens_b = extract_families(b)

    # ---------------------------------------------------------------------
    # Determine comparison result
    # ---------------------------------------------------------------------
    if not fam_a and not fam_b:
        result = "Metadata Gap"
        reason = "Both models missing I/O library information"
    elif not fam_a or not fam_b:
        result = "Metadata Gap"
        reason = "One model missing I/O library information"
    elif fam_a.isdisjoint(fam_b):
        result = "Mismatch"
        reason = f"No shared I/O families (A={sorted(fam_a)}, B={sorted(fam_b)})"
    else:
        result = "Match"
        reason = f"Shared I/O families: {sorted(fam_a.intersection(fam_b))}"

    # ---------------------------------------------------------------------
    # Construct result row
    # ---------------------------------------------------------------------
    return {
        "phase": "Phase 3",
        "check": "I/O Library Compatibility Mismatch",
        "pattern": pattern,
        "group": group,
        "model_a": a.get("name"),
        "model_b": b.get("name"),
        "tokens_a": sorted(tokens_a),
        "tokens_b": sorted(tokens_b),
        "families_a": sorted(fam_a),
        "families_b": sorted(fam_b),
        "result": result,
        "reason": reason,
    }


# ------------- Shared only---------------

def r3_shared_store_config_mismatch(
    group: str, a: dict, b: dict, ab: Optional[dict], pattern: str
) -> dict:
    """
    Phase 3 — Shared Store Configuration Compatibility
    ----------------------------------------------------------
    Purpose:
        Detect whether both models can operate on the same persistent
        storage backend (e.g., Postgres, Redis, MongoDB, etc.).

    Logic:
        • Extract technology tokens from Software Requirements + Distribution Version.
        • Map tokens to known storage families.
        • If both missing        → Metadata Gap
        • If one missing         → Metadata Gap
        • If disjoint families   → Mismatch
        • If overlap             → Match
    """

    import re

    # ---------------------------------------------------------------------
    # Known storage technologies grouped by family
    # ---------------------------------------------------------------------
    STORAGE_FAMILIES = {
        "Postgres_Family": {"postgres", "postgresql", "psycopg", "pg"},
        "MySQL_Family": {"mysql", "mariadb"},
        "SQLite_Family": {"sqlite"},
        "Redis_Family": {"redis"},
        "MongoDB_Family": {"mongo", "mongodb"},
        "Kafka_Family": {"kafka", "confluent"},
        "RabbitMQ_Family": {"rabbitmq", "amqp"},
    }

    # ---------------------------------------------------------------------
    # Helper: extract family tokens from model fields
    # ---------------------------------------------------------------------
    def extract_storage_families(model_dict):
        text_fields = []
        for key in ["software_specification_and_requirements", "distribution_version"]:
            val = model_dict.get(key)
            if isinstance(val, str):
                text_fields.append(val.lower())
            elif isinstance(val, list):
                text_fields.extend([str(x).lower() for x in val if x])
        joined = " ".join(text_fields)

        detected_families = set()
        detected_tokens = set()

        for fam, tokens in STORAGE_FAMILIES.items():
            for t in tokens:
                if re.search(rf"\b{re.escape(t)}\b", joined):
                    detected_families.add(fam)
                    detected_tokens.add(t)
        return detected_families, detected_tokens

    # ---------------------------------------------------------------------
    # Extract from both models
    # ---------------------------------------------------------------------
    fam_a, tokens_a = extract_storage_families(a)
    fam_b, tokens_b = extract_storage_families(b)

    # ---------------------------------------------------------------------
    # Determine result
    # ---------------------------------------------------------------------
    if not fam_a and not fam_b:
        result = "Metadata Gap"
        reason = "Both models missing declared storage backend information."
    elif not fam_a or not fam_b:
        result = "Metadata Gap"
        reason = "One model missing declared storage backend information."
    elif fam_a.isdisjoint(fam_b):
        result = "Mismatch"
        reason = f"Incompatible storage families (A={sorted(fam_a)}, B={sorted(fam_b)})"
    else:
        result = "Match"
        reason = f"Shared storage families overlap: {sorted(fam_a.intersection(fam_b))}"

    # ---------------------------------------------------------------------
    # Return unified result structure
    # ---------------------------------------------------------------------
    return {
        "phase": "Phase 3",
        "check": "Shared Store Compatibility Mismatch",
        "pattern": pattern,
        "group": group,
        "model_a": a.get("name"),
        "model_b": b.get("name"),
        "tokens_a": sorted(tokens_a),
        "tokens_b": sorted(tokens_b),
        "families_a": sorted(fam_a),
        "families_b": sorted(fam_b),
        "result": result,
        "reason": reason,
    }


# ----------------------- Integrated only --------------------
def r3_runtime_compat_mismatch(
    group: str, a: dict, b: dict, ab: Optional[dict], pattern: str
) -> dict:
    """
    Phase 3 — Runtime Language Compatibility
    ----------------------------------------------------------
    Pattern: Integrated
    Purpose:
        Checks programming-language compatibility at runtime.

    Logic:
        • Extracts programming languages from both models.
        • Maps each to known ABI and FFI families.
        • Compares for overlap:
              If both missing           → Metadata Gap
              If one missing            → Metadata Gap
              If joint ABI families     → Match
              Else if joint FFI families→ Match
              Else                      → Mismatch

    Why:
        In Integrated coupling, components often execute within the same or closely
        linked runtime; ABI/FFI compatibility ensures integration feasibility.
    """

    import re

    # === ABI / FFI family definitions ===
    ABI_FAMILIES = {
        # Virtual-machine based ABIs
        "JVM": {"Java", "Kotlin", "Scala", "Groovy"},
        ".NET": {"C#", "F#", "VB.NET", "PowerShell"},

        # Native compiled binaries (C ABI compatible)
        "Native_C_CPP_Fortran": {"C", "C++", "Fortran"},

        # Languages compiled to C ABI via LLVM or similar
        "LLVM_C_ABI": {"Rust", "Go", "Zig"},

        # Scripting languages embedding the C runtime
        "Python_R_CPP": {"Python", "R", "C", "C++"},  # many R/Python extensions use C ABI
    }

    FFI_FAMILIES = {
        # Standard C ABI bridge (ctypes/libffi-based)
        "C_ABI": {"Python", "R", "Julia", "C", "C++", "Fortran"},

        # Java Native Interface family
        "JNI": {"Java", "Kotlin", "Scala", "Groovy", "C", "C++"},

        # .NET Platform Invoke
        "PInvoke": {"C#", "F#", "VB.NET", "PowerShell", "C", "C++"},

        # MetaFFI or other cross-language bridges
        "MetaFFI": {"Python", "Java", "C#", "C++", "Kotlin", "Node.js", "R"},

        # R–C interface (Rcpp, .Call)
        "Rcpp_ABI": {"R", "C", "C++", "Fortran"},

        # Python C-extension interface (Cython, CPython API)
        "Python_C_Ext": {"Python", "C", "C++"},

        # Julia’s native FFI layer
        "Julia_FFI": {"Julia", "C", "C++", "Fortran"},
    }

    def _norm_lang(s: str) -> str:
        return (s or "").strip().lower()

    # ---------------------------------------------------------------------
    # Helper: extract language families
    # ---------------------------------------------------------------------
    def extract_lang_families(lang_str: str):
        families_abi = set()
        families_ffi = set()
        detected_langs = set()

        for token in re.split(r"[;,]", lang_str or ""):
            token = token.strip()
            if not token:
                continue
            detected_langs.add(token)

            # Check ABI families
            for fam, langs in ABI_FAMILIES.items():
                if token in langs:
                    families_abi.add(fam)

            # Check FFI families
            for fam, langs in FFI_FAMILIES.items():
                if token in langs:
                    families_ffi.add(fam)

        return detected_langs, families_abi, families_ffi

    # ---------------------------------------------------------------------
    # Extract language info
    # ---------------------------------------------------------------------
    lang_a = "; ".join(a.get("programming_language", [])) if isinstance(a.get("programming_language"), list) else a.get("programming_language", "")
    lang_b = "; ".join(b.get("programming_language", [])) if isinstance(b.get("programming_language"), list) else b.get("programming_language", "")

    langs_a, abi_a, ffi_a = extract_lang_families(lang_a)
    langs_b, abi_b, ffi_b = extract_lang_families(lang_b)

    # ---------------------------------------------------------------------
    # Determine compatibility
    # ---------------------------------------------------------------------
    if not langs_a and not langs_b:
        result = "Metadata Gap"
        reason = "Both models missing programming language information."
    elif not langs_a or not langs_b:
        result = "Metadata Gap"
        reason = "One model missing programming language information."
    else:
        # Check same language, ABI, FFI
        same_lang = bool({l.lower() for l in langs_a} & {l.lower() for l in langs_b})
        abi_overlap = bool(abi_a & abi_b)
        ffi_overlap = bool(ffi_a & ffi_b)

        if same_lang or abi_overlap or ffi_overlap:
            result = "Match"
            shared = set()
            if abi_overlap:
                shared.update(abi_a & abi_b)
            if ffi_overlap:
                shared.update(ffi_a & ffi_b)
            reason = (
                f"Compatible runtime environment: "
                f"same_lang={same_lang}, shared families={sorted(shared)}"
            )
        else:
            result = "Mismatch"
            reason = (
                f"No common language or ABI/FFI families. "
                f"A_ABI={sorted(abi_a)}, B_ABI={sorted(abi_b)}; "
                f"A_FFI={sorted(ffi_a)}, B_FFI={sorted(ffi_b)}"
            )

    # ---------------------------------------------------------------------
    # Return unified structured result
    # ---------------------------------------------------------------------
    return {
        "phase": "Phase 3",
        "check": "Runtime Compatibility Mismatch",
        "pattern": pattern,
        "group": group,
        "model_a": a.get("name"),
        "model_b": b.get("name"),
        "tokens_a": sorted(langs_a),
        "tokens_b": sorted(langs_b),
        "families_a": sorted(abi_a | ffi_a),
        "families_b": sorted(abi_b | ffi_b),
        "result": result,
        "reason": reason,
    }


# ------------------------ Embedded only ---------------------

def r3_runtime_containment_mismatch(
    group: str, a: dict, b: dict, ab: Optional[dict], pattern: str
) -> dict:
    """
    Phase 3 — Runtime Containment Compatibility (Embedded)
    ----------------------------------------------------------
    Pattern: Embedded
    Purpose:
        Ensures host and embedded models run in the same process/runtime.

    Logic:
        • Reads 'programming_language' from both models.
        • Maps each language to known ABI families.
        • Compares for overlap:
              If both missing         → Metadata Gap
              If one missing          → Metadata Gap
              If joint ABI families   → Match
              Else                    → Mismatch
        • FFI families are NOT considered (requires in-process ABI match).
    """

    import re

    # ---------------------------------------------------------------------
    # Define ABI families (only, no FFI here)
    # ---------------------------------------------------------------------
    ABI_FAMILIES = {
        "JVM_Family": {"Java", "Kotlin", "Scala", "Groovy"},
        ".NET_Family": {"C#", "F#", "VB.NET", "PowerShell"},
        "Native_C_CPP_Fortran_Family": {"C", "C++", "Fortran"},
        "LLVM_C_ABI_Family": {"Rust", "Go", "Zig"},
        "Python_R_CPP_Family": {"Python", "R", "C", "C++"},  # scripting languages using C ABI
    }

    def _norm_lang(s: str) -> str:
        return (s or "").strip().lower()

    # ---------------------------------------------------------------------
    # Helper: extract ABI families for a model
    # ---------------------------------------------------------------------
    def extract_abi_families(model_dict):
        text_val = model_dict.get("programming_language", "")
        if isinstance(text_val, list):
            langs = [str(x).strip() for x in text_val if x]
        else:
            langs = [x.strip() for x in re.split(r"[;,]", str(text_val)) if x.strip()]

        detected_langs = set()
        detected_families = set()
        for lang in langs:
            if not lang:
                continue
            detected_langs.add(lang)
            for fam, members in ABI_FAMILIES.items():
                if lang in members:
                    detected_families.add(fam)
        return detected_langs, detected_families

    # ---------------------------------------------------------------------
    # Extract languages and ABI families
    # ---------------------------------------------------------------------
    langs_a, abi_a = extract_abi_families(a)
    langs_b, abi_b = extract_abi_families(b)

    # ---------------------------------------------------------------------
    # Determine containment compatibility
    # ---------------------------------------------------------------------
    if not langs_a and not langs_b:
        result = "Metadata Gap"
        reason = "Both models missing programming language metadata."
    elif not langs_a or not langs_b:
        result = "Metadata Gap"
        reason = "One model missing programming language metadata."
    else:
        same_lang = bool({_norm_lang(l) for l in langs_a} & {_norm_lang(l) for l in langs_b})
        abi_overlap = bool(abi_a & abi_b)

        if same_lang or abi_overlap:
            result = "Match"
            reason = f"In-process compatible: same_lang={same_lang}, shared ABI families={sorted(abi_a & abi_b)}"
        else:
            result = "Mismatch"
            reason = (
                f"Different runtime environments. "
                f"A_ABI={sorted(abi_a)}, B_ABI={sorted(abi_b)}"
            )

    # ---------------------------------------------------------------------
    # Return unified result structure
    # ---------------------------------------------------------------------
    return {
        "phase": "Phase 3",
        "check": "Runtime Containment Mismatch",
        "pattern": pattern,
        "group": group,
        "model_a": a.get("name"),
        "model_b": b.get("name"),
        "tokens_a": sorted(langs_a),
        "tokens_b": sorted(langs_b),
        "families_a": sorted(abi_a),
        "families_b": sorted(abi_b),
        "result": result,
        "reason": reason,
    }




# ============================================================
# Pattern-aware Phase 3 dispatcher (calls relevant mismatch fns)
# ============================================================

def run_phase3_runtime_checks(group: str, a: ModelMeta, b: ModelMeta, ab: Optional[ModelMeta], pattern: str) -> List[Dict[str,Any]]:
    rows: List[Dict[str,Any]] = []
    patt = (pattern or "").strip()
    #rows.extend(r3_execution_reproducibility_gap(group, a, b, ab, pattern))

    if patt == "One-Way":
        rows.append(r3_io_library_compatibility_gap(group, a.root, b.root, ab.root if ab else None, pattern))
        return rows

    if patt == "Loose":
        rows.append(r3_io_library_compatibility_gap(group, a.root, b.root, ab.root if ab else None, pattern))
        return rows

    if patt == "Shared":
        rows.append(r3_shared_store_config_mismatch(group, a.root, b.root, ab.root if ab else None, pattern))
        return rows

    if patt == "Integrated":
        rows.append(r3_runtime_compat_mismatch(group, a.root, b.root, ab.root if ab else None, pattern))
        return rows

    if patt == "Embedded":
        rows.append(r3_runtime_containment_mismatch(group, a.root, b.root, ab.root if ab else None, pattern))
        return rows

    return rows




# ============================================================
# Group evaluation
# ============================================================

def evaluate_group(gid: str, A: ModelMeta, B: ModelMeta, AB: Optional[ModelMeta], debug: bool=False) -> List[Dict[str,Any]]:
    pattern, pr = detect_integration_pattern(A, B, AB)
    if debug:
        print(f"[GROUP {gid}] inferred pattern: {pattern} | {pr}")
    rows: List[Dict[str,Any]] = []
    # Phase 1
    rows += check_phase1_conceptual_and_legal_weighted(gid, A, B, AB, pattern)
    # rows += check_conceptual_quality_gap(gid, A, B, AB, pattern)
    # Phase 2
    rows += check_phase2_information_alignment_weighted(gid, A, B, AB, pattern)
    # Phase 3 (new pattern-aware runtime dispatcher)
    rows += run_phase3_runtime_checks(gid, A, B, AB, pattern)
    return rows

# -----------------------------------------------------------------
# Summary helper (must appear before main)
# -----------------------------------------------------------------
from collections import Counter, defaultdict
from typing import List, Dict, Any

def summarize_results(rows: List[Dict[str, Any]]) -> None:
    """
    Summarize total number of Matches, Gaps, and Mismatches across all checks.
    Also prints counts by specific mismatch/gap subtype when available.

    Compatible with the standardized Phase 3 result format:
    - 'result'  → 'Match', 'Mismatch', 'Metadata Gap'
    - 'check'   → check type (e.g. 'Runtime Compatibility')
    - 'reason'  → explanation of finding
    """
    type_counter = Counter()
    subtype_counter = defaultdict(Counter)

    for r in rows:
        result = r.get("result", "").strip()
        if not result:
            continue

        # Normalize result labels
        if result.lower() in ("gap", "metadata gap"):
            category = "Metadata Gap"
        elif result.lower() == "mismatch":
            category = "Mismatch"
        elif result.lower() == "match":
            category = "Match"
        else:
            category = "Other"

        type_counter[category] += 1

        # Subtype breakdown by check type
        check_name = r.get("check", "Unknown Check")

        if category == "Metadata Gap":
            subtype_counter["Metadata Gap"][check_name] += 1
        elif category == "Mismatch":
            subtype_counter["Mismatch"][check_name] += 1

    total = sum(type_counter.values())
    print("\n=== Integration Analysis Summary ===")
    print(f"Total checks: {total}")
    for k in ["Match", "Metadata Gap", "Mismatch"]:
        print(f"{k:15s}: {type_counter.get(k, 0)}")

    # Breakdown by subtype
    if subtype_counter["Metadata Gap"]:
        print("\n--- Metadata Gap Types ---")
        for subtype, count in subtype_counter["Metadata Gap"].items():
            print(f"{subtype:40s}: {count}")

    if subtype_counter["Mismatch"]:
        print("\n--- Mismatch Types ---")
        for subtype, count in subtype_counter["Mismatch"].items():
            print(f"{subtype:40s}: {count}")

    print("====================================\n")

#______________________________

from typing import Dict, List, Set

# ------------------------------------------------------------
# Viewpoint categorization from FIELD_ALIASES
# ------------------------------------------------------------
VIEWPOINT_FIELDS = {
    "Domain": [
        "scope", "purpose_pattern", "assumptions",
        "validation_capabilities", "conceptual_model_evaluation",
        "links_to_publications_and_reports", "authors_unique_identifier",
        "calibration_tools_data", "sensitivity_analysis", "uncertainty_analysis"
    ],
    "Information": [
        "unique_identifier", "parameters", "datasets", "data",
        "dimensionality", "spatial_extent_coverage", "spatial_resolution",
        "variable_spatial_resolution", "temporal_extent_coverage",
        "time_steps_temporal_resolution", "variable_temporal_resolution",
        "resampling_policies", "input", "output", "integrated_input"
    ],
    "Computational": [
        "error_handling", "integration_pattern", "communication_mechanism", "acknowledgment_protocols"
    ],
    "Engineering": [
        "parallel_execution", "latency_expectations", "data_synchronization","execution_constraints","execution_instructions"
    ],
    "Technology": [
        "programming_language", "availability_of_source_code", "implementation_verification",
        "software_specification_and_requirements", "hardware_specification_and_requirements",
        "license", "landing_page", "distribution_version", "file_formats"
    ],
}

# ------------------------------------------------------------
# Mapping mismatch types to fields involved (from your table)
# ------------------------------------------------------------
MISMATCH_FIELDS = {
    "Conceptual Consistency Mismatch": [
        "scope", "purpose_pattern", "assumptions", "validation_capabilities", "conceptual_model_evaluation"
    ],
    "License Incompatibility": ["license"],
    "Provenance and Accessibility Gap": [
        "model_identifier", "landing_page", "source_code_availability"
    ],
    "Conceptual Quality Gap": [
        "links_to_publications_and_reports", "authors_unique_identifier",
        "conceptual_model_evaluation", "calibration_tools_data",
        "validation_capabilities", "sensitivity_analysis", "uncertainty_analysis"
    ],
    "Information Alignment Mismatch": [
        "parameters", "datasets", "data", "input", "output",
        "spatial_resolution", "time_steps_temporal_resolution",  "unique_identifier",
        "dimensionality", "spatial_extent_coverage",
        "variable_spatial_resolution", "temporal_extent_coverage", "variable_temporal_resolution",
        "resampling_policies", "input", "output", "integrated_input"
    ],
    "I/O Library Compatibility Mismatch": [
        "software_specification_and_requirements", "distribution_version"
    ],
    "Shared Store Compatibility Mismatch": [
        "software_specification_and_requirements", "distribution_version",
        "execution_instructions"
    ],
    "Runtime Compatibility Mismatch": [
        "programming_language", "software_specification_and_requirements",
        "distribution_version", "execution_constraints",
        "execution_instructions"
        ,"error_handling"
    ],
    "Runtime Containment Mismatch": [
        "programming_language", "software_specification_and_requirements",
        "distribution_version", "execution_constraints",
        "execution_instructions"
         ,"error_handling"
    ],
}
# ------------------------------------------------------------

# Define mismatch categories
hard_mismatches = {
    "Conceptual Consistency Mismatch",
    "License Incompatibility",
    "Conceptual Quality Gap",
    "Runtime Compatibility Mismatch (Integrated)",
    "Runtime Containment Mismatch (Embedded)"
}

soft_mismatches = {
    "Provenance and Accessibility Gap",
    "Information Alignment Mismatch",
    "I/O Library Compatibility Mismatch"
    "Shared Store Compatibility Mismatch",

}

def classify_mismatch(mismatch_name: str) -> str:
    """
    Classifies a mismatch as hard, soft, or unknown,
    using substring matching for robustness.
    """
    name = mismatch_name.strip()

    for h in hard_mismatches:
        if name in h or h in name:
            return "Hard Mismatch"

    for s in soft_mismatches:
        if name in s or s in name:
            return "Soft Mismatch"

    return "Unknown"


# Example usage:
print(classify_mismatch("License Incompatibility"))
print(classify_mismatch("Information Alignment Mismatch"))
print(classify_mismatch("Something Else"))

# ------------------------------------------------------------
# Pattern → applicable mismatch/check types
# ------------------------------------------------------------
PATTERN_MISMATCHES: Dict[str, List[str]] = {
    "One-Way": [
        "Conceptual Consistency Mismatch",
        "License Incompatibility",
        "Provenance and Accessibility Gap",
        "Conceptual Quality Gap",
        "Information Alignment Mismatch",
        "I/O Library Compatibility Mismatch",
    ],
    "Loose": [
        "Conceptual Consistency Mismatch",
        "License Incompatibility",
        "Provenance and Accessibility Gap",
        "Conceptual Quality Gap",
        "Information Alignment Mismatch",
        "I/O Library Compatibility Mismatch",
    ],
    "Shared": [
        "Conceptual Consistency Mismatch",
        "License Incompatibility",
        "Provenance and Accessibility Gap",
        "Conceptual Quality Gap",
        "Information Alignment Mismatch",
        "Shared Store Compatibility Mismatch",
    ],
    "Integrated": [
        "Conceptual Consistency Mismatch",
        "License Incompatibility",
        "Provenance and Accessibility Gap",
        "Conceptual Quality Gap",
        "Information Alignment Mismatch",
        "Runtime Compatibility Mismatch",
    ],
    "Embedded": [
        "Conceptual Consistency Mismatch",
        "License Incompatibility",
        "Provenance and Accessibility Gap",
        "Conceptual Quality Gap",
        "Information Alignment Mismatch",
        "Runtime Containment Mismatch",
    ],
}

# Optional: normalized alias map for safer lookups
_PATTERN_NORMALIZATION = {
    "one-way": "One-Way",
    "one way": "One-Way",
    "loose": "Loose",
    "shared": "Shared",
    "integrated": "Integrated",
    "embedded": "Embedded",
}

def mismatches_for_pattern(pattern: str) -> List[str]:
    """
    Return the list of mismatch/check names applicable to the given pattern.
    Normalizes common variations ('one way' → 'One-Way').
    """
    if not pattern:
        return []
    key = _PATTERN_NORMALIZATION.get(pattern.strip().lower(), pattern.strip())
    return PATTERN_MISMATCHES.get(key, [])

# (Convenience) filter a dataframe to only rows whose `check` is applicable for its `pattern`
def df_applicable_checks_only(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    # normalize a helper co


# ------------------------------------------------------------

def get_mismatches_for_viewpoint_and_pattern(viewpoint: str, pattern: str) -> List[str]:
    """
    Return the list of mismatches that are relevant to BOTH a given viewpoint
    and integration pattern (i.e., the intersection of the two sets).

    Parameters
    ----------
    viewpoint : str
        One of ["Domain", "Information", "Computational", "Engineering", "Technology"]
    pattern : str
        One of ["One-Way", "Loose", "Shared", "Integrated", "Embedded"]

    Returns
    -------
    List[str]
        List of mismatch/check names common to both the given viewpoint and pattern.
    """
    # Normalize pattern name
    norm_pat = _PATTERN_NORMALIZATION.get(pattern.strip().lower(), pattern.strip())

    # Get mismatches for this pattern
    pattern_mismatches = set(PATTERN_MISMATCHES.get(norm_pat, []))

    # Get mismatches for this viewpoint using your earlier mapping
    mismatch_to_vp = map_mismatch_to_viewpoints()
    viewpoint_mismatches = {m for m, vps in mismatch_to_vp.items() if viewpoint in vps}

    # Intersection = relevant to both
    common = sorted(pattern_mismatches.intersection(viewpoint_mismatches))
    return common

# ------------------------------------------------------------
# Compute which viewpoints each mismatch touches
# ------------------------------------------------------------
def map_mismatch_to_viewpoints() -> Dict[str, Set[str]]:
    mismatch_to_vp: Dict[str, Set[str]] = {}

    for mismatch, fields in MISMATCH_FIELDS.items():
        viewpoints = set()
        for field in fields:
            for vp_name, vp_fields in VIEWPOINT_FIELDS.items():
                if field in vp_fields:
                    viewpoints.add(vp_name)
                    break  # found its viewpoint
        mismatch_to_vp[mismatch] = viewpoints

    return mismatch_to_vp


def get_relevant_mismatches(viewpoint_name):
    """
    Return all mismatch types that use fields belonging to a given viewpoint.
    """
    # Validate viewpoint name
    if viewpoint_name not in VIEWPOINT_FIELDS:
        raise ValueError(f"Invalid viewpoint name: {viewpoint_name}")

    # Get the fields for this viewpoint
    viewpoint_fields = set(VIEWPOINT_FIELDS[viewpoint_name])

    # Find all mismatches that share any field with this viewpoint
    relevant_mismatches = []
    for mismatch, fields in MISMATCH_FIELDS.items():
        if viewpoint_fields.intersection(fields):
            relevant_mismatches.append(mismatch)

    return relevant_mismatches
# ============================================================
def plot_viewpoint_pies2(df: pd.DataFrame, figs_dir: str = "figs"):

    os.makedirs(figs_dir, exist_ok=True)
    df = df.copy()

    print("\n================ DEBUG: Raw Columns ================")
    print(df.columns.tolist())
    print("====================================================\n")

    # ------------------------------------------------------------
    # STEP 1 — Define extra viewpoint → field mappings (YOUR RULES)
    # ------------------------------------------------------------
    ENGINEERING_FIELDS = [
        "software_specification_and_requirements",
        "distribution_version",
        "execution_instructions",
        "programming_language",
        "execution_constraints"
    ]

    COMPUTATION_FIELDS = [
        "programming_language",
        "software_specification_and_requirements",
        "distribution_version",
        "execution_constraints",
        "execution_instructions",
        "error_handling",
    ]

    # ------------------------------------------------------------
    # STEP 2 — Build CHECK → VIEWPOINT mapping from your mismatch logic
    # ------------------------------------------------------------
    CHECK_TO_VP = {}

    # Add existing mappings from your mismatch framework
    for vp in VP_ORDER:
        for pat in PATTERN_ORDER:
            for chk in get_mismatches_for_viewpoint_and_pattern(vp, pat):
                CHECK_TO_VP[chk] = vp

    # ------------------------------------------------------------
    # STEP 3 — Add Engineering + Computation corrections
    # ------------------------------------------------------------
    for chk in ENGINEERING_FIELDS:
        CHECK_TO_VP[chk] = "Engineering"

    for chk in COMPUTATION_FIELDS:
        CHECK_TO_VP[chk] = "Computation"

    print("\n=== DEBUG: FULL CHECK → VIEWPOINT mapping AFTER EXTENSION ===")
    for chk, vp in CHECK_TO_VP.items():
        print(f"{chk:60s} → {vp}")
    print("===================================================\n")

    # ------------------------------------------------------------
    # STEP 4 — Assign viewpoint to each mismatch row
    # ------------------------------------------------------------
    df["_check"] = df["check"].astype(str)
    df["Viewpoint"] = df["_check"].map(CHECK_TO_VP).fillna("Unknown")

    print("\n=== DEBUG: First 20 viewpoint assignments ===")
    print(df[["_check", "Viewpoint"]].head(20).to_string(index=False))
    print("===================================================\n")

    print("\n=== DEBUG: COUNT of rows per viewpoint BEFORE removing Unknown ===")
    print(df["Viewpoint"].value_counts())
    print("================================================================\n")

    # remove unmapped rows
    df = df[df["Viewpoint"] != "Unknown"].copy()

    print("\n=== DEBUG: COUNT per viewpoint AFTER removing Unknown ===")
    print(df["Viewpoint"].value_counts())
    print("========================================================\n")

    if df.empty:
        print("[ERROR] No mapped rows. Cannot plot viewpoint pies.")
        return

    # ------------------------------------------------------------
    # STEP 5 — Normalize and compute ratios
    # ------------------------------------------------------------
    df["_phase"] = df["phase"].astype(str)
    df["_result"] = df["result"].astype(str).str.title().replace({"Gap": "Metadata Gap"})

    # Phase split
    phase3 = df[df["_phase"].str.contains("Phase 3", case=False, na=False)].copy()
    phase12 = df[~df["_phase"].str.contains("Phase 3", case=False, na=False)].copy()

    # ---------- Phase 3 ----------
    if not phase3.empty:
        phase3["_match_ratio"] = (phase3["_result"] == "Match").astype(float)
        phase3["_gap_ratio"] = (phase3["_result"] == "Metadata Gap").astype(float)

        phase3["_hard_ratio"] = phase3.apply(
            lambda r: 1.0 if (
                r["_result"] == "Mismatch" and classify_mismatch(r["_check"]) == "Hard Mismatch"
            ) else 0.0,
            axis=1
        )

        phase3["_soft_ratio"] = phase3.apply(
            lambda r: 1.0 if (
                r["_result"] == "Mismatch" and classify_mismatch(r["_check"]) == "Soft Mismatch"
            ) else 0.0,
            axis=1
        )

    else:
        phase3["_match_ratio"] = phase3["_gap_ratio"] = 0.0
        phase3["_hard_ratio"] = phase3["_soft_ratio"] = 0.0

    # ---------- Phase 1–2 ----------
    if not phase12.empty:
        phase12["_match_ratio"] = pd.to_numeric(phase12["match_ratio"], errors="coerce").fillna(0)
        phase12["_gap_ratio"] = pd.to_numeric(phase12["gap_ratio"], errors="coerce").fillna(0)

        phase12["_hard_ratio"] = phase12.apply(
            lambda r: r["mismatch_ratio"]
            if classify_mismatch(r["_check"]) == "Hard Mismatch" else 0.0,
            axis=1
        )

        phase12["_soft_ratio"] = phase12.apply(
            lambda r: r["mismatch_ratio"]
            if classify_mismatch(r["_check"]) == "Soft Mismatch" else 0.0,
            axis=1
        )

    else:
        phase12["_match_ratio"] = phase12["_gap_ratio"] = 0.0
        phase12["_hard_ratio"] = phase12["_soft_ratio"] = 0.0

    # ------------------------------------------------------------
    # STEP 6 — Combine both phases
    # ------------------------------------------------------------
    df2 = pd.concat([phase12, phase3], ignore_index=True)

    print("\n=== DEBUG: Combined Rows (first 20) ===")
    print(df2[["_check", "Viewpoint", "_match_ratio", "_hard_ratio", "_soft_ratio", "_gap_ratio"]]
          .head(20).to_string(index=False))
    print("===================================================\n")

    # ------------------------------------------------------------
    # STEP 7 — Aggregate per viewpoint
    # ------------------------------------------------------------
    summary = []

    for vp in VP_ORDER:
        sub = df2[df2["Viewpoint"] == vp]

        print(f"\n=== DEBUG: Rows contributing to '{vp}' ===")
        print(sub[["_check", "_result"]].to_string(index=False))
        print("===================================================\n")

        if sub.empty:
            continue

        match_sum = sub["_match_ratio"].sum()
        hard_sum  = sub["_hard_ratio"].sum()
        soft_sum  = sub["_soft_ratio"].sum()
        gap_sum   = sub["_gap_ratio"].sum()

        total = match_sum + hard_sum + soft_sum + gap_sum
        if total == 0:
            continue

        summary.append({
            "Viewpoint": vp,
            "Match": match_sum / total,
            "Hard Mismatch": hard_sum / total,
            "Soft Mismatch": soft_sum / total,
            "Metadata Gap": gap_sum / total,
        })

    df_vp = pd.DataFrame(summary).set_index("Viewpoint")

    print("\n=== FINAL SUMMARY TABLE ===")
    print(df_vp.to_string())
    print("=========================================\n")

    # ------------------------------------------------------------
    # STEP 8 — Plot pie charts
    # ------------------------------------------------------------
    COLORS = {
        "Match": "#4CAF50",
        "Hard Mismatch": "#d62728",
        "Soft Mismatch": "#ff9896",
        "Metadata Gap": "#FFC107",
    }

    required_cols = list(COLORS.keys())

    n = len(df_vp)
    cols = 3
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), squeeze=False)

    for i, (vp, row) in enumerate(df_vp.iterrows()):
        ax = axes[i // cols][i % cols]
        vals = [row[c] for c in required_cols]

        ax.pie(vals,
               colors=[COLORS[c] for c in required_cols],
               autopct=lambda p: f"{p:.1f}%",
               textprops={"fontsize": 13, "weight": "bold"})

        ax.set_title(f"{string.ascii_lowercase[i]}) {vp}", fontsize=16, weight="bold")

    # Hide unused axes
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    # Legend
    handles = [mpatches.Patch(color=COLORS[c], label=c) for c in required_cols]
    fig.legend(handles=handles,
               loc="center right",
               bbox_to_anchor=(1.15, 0.5),
               fontsize=14)

    fig.tight_layout()
    out = os.path.join(figs_dir, "viewpoint_pies.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Saved viewpoint pies → {out}")

# ============================================================
def plot_viewpoint_pies(df: pd.DataFrame, figs_dir: str = "figs"):
    """
    Creates one pie chart per viewpoint showing:
       Match / Hard Mismatch / Soft Mismatch / Metadata Gap
    """

    os.makedirs(figs_dir, exist_ok=True)
    df = df.copy()

    # ==============================================================
    # DEBUG: Print useful information
    # ==============================================================
    print("\n================ DEBUG: DataFrame Columns ================")
    print(df.columns.tolist())
    print("==========================================================\n")

    print("============= DEBUG: First 10 rows =======================")
    print(df.head(10).to_string(index=False))
    print("==========================================================\n")

    # ==============================================================
    # STEP 1 — Build CHECK → VIEWPOINT mapping
    # ==============================================================
    CHECK_TO_VP = {}

    for vp in VP_ORDER:
        for pat in PATTERN_ORDER:
            relevant = get_mismatches_for_viewpoint_and_pattern(vp, pat)
            for chk in relevant:
                # Last viewpoint wins if overlaps exist (rare)
                CHECK_TO_VP[chk] = vp

    print("\n=== DEBUG: CHECK → VIEWPOINT mapping ===")
    for chk, vp in CHECK_TO_VP.items():
        print(f"  {chk} → {vp}")
    print("===========================================================\n")

    # ==============================================================
    # STEP 2 — Attach viewpoint column based on _check field
    # ==============================================================
    df["_check"] = df["check"].astype(str)
    df["viewpoint"] = df["_check"].map(CHECK_TO_VP).fillna("Unknown")

    print("\n=== DEBUG: viewpoint assignment (first 20) ===")
    print(df[["_check", "viewpoint"]].head(20).to_string(index=False))
    print("===========================================================\n")

    # ==============================================================
    # STEP 3 — Normalize needed fields
    # ==============================================================
    df["_phase"] = df["phase"].astype(str)
    df["_result"] = df["result"].astype(str).str.strip().str.title()
    df["_result"] = df["_result"].replace({"Gap": "Metadata Gap"})

    # ==============================================================
    # STEP 4 — Phase 3 handling (categorical → ratios)
    # ==============================================================
    phase3 = df[df["_phase"].str.contains("Phase 3", case=False, na=False)].copy()
    phase12 = df[~df["_phase"].str.contains("Phase 3", case=False, na=False)].copy()

    # ---------- PHASE 3 ----------
    if not phase3.empty:
        # match + gap
        phase3["_match_ratio"] = (phase3["_result"] == "Match").astype(float)
        phase3["_gap_ratio"] = (phase3["_result"] == "Metadata Gap").astype(float)

        # classify mismatches
        phase3["_hard_ratio"] = phase3.apply(
            lambda r: 1.0 if (r["_result"] == "Mismatch"
                              and classify_mismatch(r["_check"]) == "Hard Mismatch") else 0.0,
            axis=1,
        )

        phase3["_soft_ratio"] = phase3.apply(
            lambda r: 1.0 if (r["_result"] == "Mismatch"
                              and classify_mismatch(r["_check"]) == "Soft Mismatch") else 0.0,
            axis=1,
        )
    else:
        phase3["_match_ratio"] = phase3["_gap_ratio"] = 0.0
        phase3["_hard_ratio"] = phase3["_soft_ratio"] = 0.0

    # ---------- PHASE 1 & 2 ----------
    if not phase12.empty:
        phase12["_match_ratio"] = pd.to_numeric(phase12["match_ratio"], errors="coerce").fillna(0)
        phase12["_gap_ratio"] = pd.to_numeric(phase12["gap_ratio"], errors="coerce").fillna(0)

        phase12["_hard_ratio"] = phase12.apply(
            lambda r: r["mismatch_ratio"]
                      if classify_mismatch(r["_check"]) == "Hard Mismatch" else 0.0,
            axis=1,
        )

        phase12["_soft_ratio"] = phase12.apply(
            lambda r: r["mismatch_ratio"]
                      if classify_mismatch(r["_check"]) == "Soft Mismatch" else 0.0,
            axis=1,
        )
    else:
        phase12["_match_ratio"] = phase12["_gap_ratio"] = 0.0
        phase12["_hard_ratio"] = phase12["_soft_ratio"] = 0.0

    # ==============================================================
    # STEP 5 — Combine Phase1&2 + Phase3
    # ==============================================================
    df2 = pd.concat([phase12, phase3], ignore_index=True)

    print("\n=== DEBUG: After Phase processing (first 10) ===")
    print(df2[
        ["_check", "viewpoint", "_match_ratio", "_hard_ratio", "_soft_ratio", "_gap_ratio"]
    ].head(10).to_string(index=False))
    print("===========================================================\n")

    # ==============================================================
    # STEP 6 — Aggregate per VIEWPOINT
    # ==============================================================
    summary = []

    for vp in VP_ORDER:
        sub = df2[df2["viewpoint"] == vp]
        if sub.empty:
            print(f"[INFO] No checks found for viewpoint: {vp}")
            continue

        # Use get(..., 0) to avoid KeyError if column missing
        match_sum = sub.get("_match_ratio", 0).sum()
        hard_sum  = sub.get("_hard_ratio", 0).sum()
        soft_sum  = sub.get("_soft_ratio", 0).sum()
        gap_sum   = sub.get("_gap_ratio", 0).sum()

        total = match_sum + hard_sum + soft_sum + gap_sum
        if total == 0:
            total = 1.0

        row = {
            "Viewpoint": vp,
            "Match": match_sum / total,
            "Hard Mismatch": hard_sum / total,
            "Soft Mismatch": soft_sum / total,
            "Metadata Gap": gap_sum / total,
        }

        print(f"\n=== DEBUG: Aggregated row for viewpoint '{vp}' ===")
        for k, v in row.items():
            print(f"   {k}: {v}")
        print("===================================================")

        summary.append(row)

    df_vp = pd.DataFrame(summary).set_index("Viewpoint")

    print("\n=========== DEBUG: Final df_vp summary table ===========")
    print(df_vp.to_string())
    print("========================================================\n")

    # ==============================================================
    # STEP 7 — PLOT PIE CHARTS (safer version)
    # ==============================================================
    REQUIRED = ["Match", "Hard Mismatch", "Soft Mismatch", "Metadata Gap"]
    COLORS = {
        "Match": "#4CAF50",
        "Hard Mismatch": "#d62728",
        "Soft Mismatch": "#ff9896",
        "Metadata Gap": "#FFC107",
        "Mismatch": "#d62728",  # fallback for old name
        "N/A": "#B0B0B0",
    }

    # Check missing columns BEFORE plotting
    for col in REQUIRED:
        if col not in df_vp.columns:
            raise ValueError(f"[ERROR] Missing aggregated column: {col}\n"
                             f"Available columns: {df_vp.columns.tolist()}")

    n = len(df_vp)
    cols = 3
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), squeeze=False)

    for i, (vp, row) in enumerate(df_vp.iterrows()):
        ax = axes[i // cols][i % cols]

        values = [row[col] for col in REQUIRED]

        print(f"\n=== DEBUG: Pie values for {vp} ===")
        print(values)
        print("=================================")

        ax.pie(
            values,
            autopct=lambda p: f"{p:.1f}%",
            labels=None,
            colors=[COLORS[c] for c in REQUIRED],
            textprops={"fontsize": 13, "weight": "bold"}
        )

        ax.set_title(f"{string.ascii_lowercase[i]}) {vp}", fontsize=16, weight="bold")

    # Hide unused axes
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    # Legend
    handles = [mpatches.Patch(color=COLORS[c], label=c) for c in REQUIRED]
    fig.legend(handles=handles, loc="center right", bbox_to_anchor=(1.12, 0.5), fontsize=14)

    fig.tight_layout()
    out_path = os.path.join(figs_dir, "viewpoint_pies.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Saved viewpoint pie charts → {out_path}")

# ============================================================
# Viewpoint Summary Chart
# ============================================================
def plot_viewpoint_pattern_summary3(df: pd.DataFrame, figs_dir: str = "figs", pattern_counts=None) -> None:
    """
    Generates a figure with Match / Hard Mismatch / Soft Mismatch / Metadata Gap
    proportions for each viewpoint–pattern combination.
    """

    os.makedirs(figs_dir, exist_ok=True)
    df = df.copy()

    ORDER4 = ["Match", "Hard Mismatch", "Soft Mismatch", "Metadata Gap", "N/A"]
    COLORS = {
        "Match": "#4CAF50",
        "Hard Mismatch": "#d62728",
        "Soft Mismatch": "#ff9896",
        "Metadata Gap": "#FFC107",
        "N/A": "#B0B0B0"  # light gray
    }

    # Normalize cols
    df["_pattern"] = df["pattern"].astype(str).str.strip().str.title()
    df["_phase"] = df.get("phase", "")
    df["_check"] = df.get("check", "").astype(str)
    df["_result"] = (
        df.get("result", "")
        .astype(str)
        .str.strip()
        .str.title()
        .replace({"Gap": "Metadata Gap"})
    )

    # ================================
    # ALWAYS create phase3 & phase12
    # ================================

    phase3 = df[df["_phase"].str.contains("Phase 3", case=False, na=False)].copy()
    phase12 = df[~df["_phase"].str.contains("Phase 3", case=False, na=False)].copy()
    # -------------------------------
    # DEBUG: PRINT BEFORE CHANGES
    # -------------------------------
    print("\n=== DEBUG: PHASE 3 BEFORE ratio computation ===")
    print(
        phase3[
            ["_phase", "_check", "_result"]
        ].to_string(index=False)
    )
    print("===============================================\n")

    # --------------------
    # Process Phase 3 rows
    # --------------------
    if not phase3.empty:
        phase3["_match_ratio"] = (phase3["_result"] == "Match").astype(float)
        phase3["_gap_ratio"] = (phase3["_result"] == "Metadata Gap").astype(float)
        print("\n=== DEBUG: Phase 3 mismatch classification ===")
        for idx, r in phase3.iterrows():
            if r["_result"] == "Mismatch":
                cls = classify_mismatch(r["_check"])
                print(f"{r['_check']}:  {cls}")
        print("================================================\n")
        phase3["_hard_ratio"] = phase3.apply(
            lambda r: 1.0 if (r["_result"] == "Mismatch" and
                              classify_mismatch(r["_check"]) == "Hard Mismatch") else 0.0,
            axis=1
        )

        phase3["_soft_ratio"] = phase3.apply(
            lambda r: 1.0 if (r["_result"] == "Mismatch" and
                              classify_mismatch(r["_check"]) == "Soft Mismatch") else 0.0,
            axis=1
        )
    else:
        # Ensure it has all columns if empty
        phase3["_match_ratio"] = 0.0
        phase3["_gap_ratio"] = 0.0
        phase3["_hard_ratio"] = 0.0
        phase3["_soft_ratio"] = 0.0

        # -------------------------------
        # DEBUG: PRINT AFTER CHANGES
        # -------------------------------
    print("\n=== DEBUG: PHASE 3 AFTER ratio computation ===")
    print(
            phase3[
                ["_phase", "_check", "_result",
                 "_match_ratio", "_mismatch_ratio", "_gap_ratio",
                 "_hard_ratio", "_soft_ratio"]
            ].to_string(index=False)
        )
    print("==============================================\n")

    # -----------------------
    # Process Phase 1–2 rows
    # -----------------------
    # -------------------------------
    # DEBUG: PRINT BEFORE CHANGES
    # -------------------------------
    print("\n=== DEBUG: PHASE 3 BEFORE ratio computation ===")
    print(
        phase12[
            ["_phase", "_check", "match_ratio",'mismatch_ratio','gap_ratio' ]
        ].to_string(index=False)
    )
    print("===============================================\n")


    if not phase12.empty:
        phase12["_match_ratio"] = phase12["match_ratio"]
        phase12["_gap_ratio"] = phase12["gap_ratio"]
        for idx, r in phase12.iterrows():
            if r["_result"] == "Mismatch":
                cls = classify_mismatch(r["_check"])
                print(f"{r['_check']}:  {cls}")
        print("================================================\n")
        phase12["_hard_ratio"] = phase12.apply(
            lambda r: r["mismatch_ratio"] if classify_mismatch(r["_check"]) == "Hard Mismatch" else 0.0,
            axis=1
        )

        phase12["_soft_ratio"] = phase12.apply(
            lambda r: r["mismatch_ratio"] if classify_mismatch(r["_check"]) == "Soft Mismatch" else 0.0,
            axis=1
        )
    else:
        # Ensure it has all columns if empty
        phase12["_match_ratio"] = 0.0
        phase12["_gap_ratio"] = 0.0
        phase12["_hard_ratio"] = 0.0
        phase12["_soft_ratio"] = 0.0
       # -------------------------------
        # DEBUG: PRINT AFTER CHANGES
        # -------------------------------
    print("\n=== DEBUG: PHASE 3 AFTER ratio computation ===")
    print(
            phase12[
                ["_phase", "_check", "_result",
                 "_match_ratio", "_mismatch_ratio", "_gap_ratio",
                 "_hard_ratio", "_soft_ratio"]
            ].to_string(index=False)
        )
    print("==============================================\n")

    # Combine
    df = pd.concat([phase12, phase3], ignore_index=True)
    print("\n=== DEBUG: NEW RATIO COLUMNS ONLY ===")
    cols = ['check', '_match_ratio', '_soft_ratio','_hard_ratio', '_gap_ratio']
    existing = [c for c in cols if c in df.columns]

    print("\n=== DEBUG: ALL ROWS FOR SELECTED COLUMNS ===")
    print(df[existing].to_string(index=False))
    print("Columns present:", existing)
    print("All dataframe columns:", df.columns.tolist())
    print("====================================\n")

    # ============================================================
    # Aggregate per viewpoint–pattern
    # ============================================================
    results = []

    for vp in VP_ORDER:
        for pat in PATTERN_ORDER:

            print("\n" + "=" * 80)
            print(f"VIEWPOINT = {vp} | PATTERN = {pat}")
            print("=" * 80)

            relevant = get_mismatches_for_viewpoint_and_pattern(vp, pat)

            print(f"Relevant checks for this VP+Pattern → {relevant}")

            if not relevant:
                print("→ No relevant checks → Adding N/A row.")
                results.append({
                    "Viewpoint": vp,
                    "Pattern": pat,
                    "Match": 0,
                    "Hard Mismatch": 0,
                    "Soft Mismatch": 0,
                    "Metadata Gap": 0,
                    "N/A": 1.0
                })
                continue

            sub = df[(df["_check"].isin(relevant)) & (df["pattern"] == pat)]

            print(f"Number of DF rows selected = {len(sub)}")
            if sub.empty:
                print("→ DF subset EMPTY → Adding N/A row.")
                results.append({
                    "Viewpoint": vp,
                    "Pattern": pat,
                    "Match": 0,
                    "Hard Mismatch": 0,
                    "Soft Mismatch": 0,
                    "Metadata Gap": 0,
                    "N/A": 1.0
                })
                continue

            print("\n--- Individual rows contributing to aggregation ---")
            for idx, row in sub.iterrows():
                print(f"• check={row['_check']}")
                print(f"  match={row['_match_ratio']}, "
                      f"hard={row['_hard_ratio']}, "
                      f"soft={row['_soft_ratio']}, "
                      f"gap={row['_gap_ratio']}\n")

            # --- compute sums  ---
            match_sum = sub["_match_ratio"].sum()
            hard_sum = sub["_hard_ratio"].sum()
            soft_sum = sub["_soft_ratio"].sum()
            gap_sum = sub["_gap_ratio"].sum()

            print("--- Averages BEFORE normalization ---")
            print(f"Match sum         = {match_sum}")
            print(f"Hard mismatch sum = {hard_sum}")
            print(f"Soft mismatch sum = {soft_sum}")
            print(f"Gap sum           = {gap_sum}")

            total = match_sum + hard_sum + soft_sum + gap_sum
            #if total == 0:
             #   total = 1.0

            print(f"Total (for normalization) = {total}")

            results.append({
                "Viewpoint": vp,
                "Pattern": pat,
                "Match": match_sum,
                "Hard Mismatch": hard_sum,
                "Soft Mismatch": soft_sum,
                "Metadata Gap": gap_sum
            })

            print("--- Normalized (final) values ---")
            print(f"Final Match         = {match_sum:.3f}")
            print(f"Final Hard mismatch = {hard_sum :.3f}")
            print(f"Final Soft mismatch = {soft_sum:.3f}")
            print(f"Final Gap           = {gap_sum:.3f}")

    if not results:
        print("[WARNING] No data available.")
        return

    df_sum = pd.DataFrame(results)

    # ============================================================
    # Plot
    # ============================================================
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    axes = axes.flatten()

    for i, vp in enumerate(VP_ORDER):
        ax = axes[i]

        sub = df_sum[df_sum["Viewpoint"] == vp]
        if sub.empty:
            ax.axis("off")
            continue

        x = np.arange(len(sub["Pattern"]))

        # normalized names to match keys
        pattern_names = sub["Pattern"].values

        # Divisors from pattern_counts
        divisors = np.array([pattern_counts.get(pat, 1) for pat in pattern_names])

        print("\n==============================")
        print(f"VIEWPOINT = {vp}")
        print("Pattern counts:", pattern_counts)
        print("==============================\n")

        # PRINT VALUES per pattern
        for idx, row in sub.iterrows():
            pat = row["Pattern"]
            pat_norm = row["Pattern"]
            divisor = pattern_counts.get(pat_norm, 1)

            print(f"  Pattern: {pat}")
            print(f"    Raw Match         = {row['Match']}")
            print(f"    Raw Hard Mismatch = {row['Hard Mismatch']}")
            print(f"    Raw Soft Mismatch = {row['Soft Mismatch']}")
            print(f"    Raw Gap           = {row['Metadata Gap']}")
            print(f"    Divisor           = {divisor}")
            print(f"    Normalized Match  = {row['Match'] / divisor}")
            print(f"    Normalized Hard   = {row['Hard Mismatch'] / divisor}")
            print(f"    Normalized Soft   = {row['Soft Mismatch'] / divisor}")
            print(f"    Normalized Gap    = {row['Metadata Gap'] / divisor}")
            print()

        # Compute normalized values for plotting
        match_vals = sub["Match"].values / divisors
        hard_vals = sub["Hard Mismatch"].values / divisors
        soft_vals = sub["Soft Mismatch"].values / divisors
        gap_vals = sub["Metadata Gap"].values / divisors
        na_vals = sub.get("N/A", pd.Series([0] * len(sub))).values / divisors

        # Draw stacked bars
        ax.bar(x, match_vals, color=COLORS["Match"])
        ax.bar(x, hard_vals, bottom=match_vals, color=COLORS["Hard Mismatch"])
        ax.bar(x, soft_vals, bottom=match_vals + hard_vals, color=COLORS["Soft Mismatch"])
        ax.bar(x, gap_vals, bottom=match_vals + hard_vals + soft_vals, color=COLORS["Metadata Gap"])
        ax.bar(x, na_vals, bottom=match_vals + hard_vals + soft_vals + gap_vals, color=COLORS["N/A"])

        ax.set_xticks(x)
        ax.set_xticklabels(sub["Pattern"], rotation=30, ha="right")
        ax.set_ylabel("Count (Normalized by Pattern Count)")
        ax.set_title(f"{string.ascii_lowercase[i]}) {vp}", fontsize=13, weight="bold")

    # Legend
    handles = [mpatches.Patch(color=COLORS[c], label=c) for c in ORDER4]
    axes[-1].axis("off")
    axes[-1].legend(handles=handles, loc="center")

    fig.tight_layout()
    fig.savefig(os.path.join(figs_dir, "viewpoint_pattern_summary3.png"), dpi=300)
    plt.close(fig)

    print("[INFO] Saved viewpoint_pattern_summary3.png")







def plot_viewpoint_pattern_summary2(df: pd.DataFrame, figs_dir: str = "figs") -> None:
    """
    Generates a figure with Match / Hard Mismatch / Soft Mismatch / Metadata Gap
    proportions for each viewpoint–pattern combination.
    """

    os.makedirs(figs_dir, exist_ok=True)
    df = df.copy()

    ORDER4 = ["Match", "Hard Mismatch", "Soft Mismatch", "Metadata Gap", "N/A"]
    COLORS = {
        "Match": "#4CAF50",
        "Hard Mismatch": "#d62728",
        "Soft Mismatch": "#ff9896",
        "Metadata Gap": "#FFC107",
        "N/A": "#B0B0B0"  # light gray
    }

    # Normalize cols
    df["_pattern"] = df["pattern"].astype(str).str.strip().str.title()
    df["_phase"] = df.get("phase", "")
    df["_check"] = df.get("check", "").astype(str)
    df["_result"] = (
        df.get("result", "")
        .astype(str)
        .str.strip()
        .str.title()
        .replace({"Gap": "Metadata Gap"})
    )

    # ================================
    # ALWAYS create phase3 & phase12
    # ================================

    phase3 = df[df["_phase"].str.contains("Phase 3", case=False, na=False)].copy()
    phase12 = df[~df["_phase"].str.contains("Phase 3", case=False, na=False)].copy()
    # -------------------------------
    # DEBUG: PRINT BEFORE CHANGES
    # -------------------------------
    print("\n=== DEBUG: PHASE 3 BEFORE ratio computation ===")
    print(
        phase3[
            ["_phase", "_check", "_result"]
        ].to_string(index=False)
    )
    print("===============================================\n")

    # --------------------
    # Process Phase 3 rows
    # --------------------
    if not phase3.empty:
        phase3["_match_ratio"] = (phase3["_result"] == "Match").astype(float)
        phase3["_gap_ratio"] = (phase3["_result"] == "Metadata Gap").astype(float)
        print("\n=== DEBUG: Phase 3 mismatch classification ===")
        for idx, r in phase3.iterrows():
            if r["_result"] == "Mismatch":
                cls = classify_mismatch(r["_check"])
                print(f"{r['_check']}:  {cls}")
        print("================================================\n")
        phase3["_hard_ratio"] = phase3.apply(
            lambda r: 1.0 if (r["_result"] == "Mismatch" and
                              classify_mismatch(r["_check"]) == "Hard Mismatch") else 0.0,
            axis=1
        )

        phase3["_soft_ratio"] = phase3.apply(
            lambda r: 1.0 if (r["_result"] == "Mismatch" and
                              classify_mismatch(r["_check"]) == "Soft Mismatch") else 0.0,
            axis=1
        )
    else:
        # Ensure it has all columns if empty
        phase3["_match_ratio"] = 0.0
        phase3["_gap_ratio"] = 0.0
        phase3["_hard_ratio"] = 0.0
        phase3["_soft_ratio"] = 0.0

        # -------------------------------
        # DEBUG: PRINT AFTER CHANGES
        # -------------------------------
    print("\n=== DEBUG: PHASE 3 AFTER ratio computation ===")
    print(
            phase3[
                ["_phase", "_check", "_result",
                 "_match_ratio", "_mismatch_ratio", "_gap_ratio",
                 "_hard_ratio", "_soft_ratio"]
            ].to_string(index=False)
        )
    print("==============================================\n")

    # -----------------------
    # Process Phase 1–2 rows
    # -----------------------
    # -------------------------------
    # DEBUG: PRINT BEFORE CHANGES
    # -------------------------------
    print("\n=== DEBUG: PHASE 3 BEFORE ratio computation ===")
    print(
        phase12[
            ["_phase", "_check", "match_ratio",'mismatch_ratio','gap_ratio' ]
        ].to_string(index=False)
    )
    print("===============================================\n")


    if not phase12.empty:
        phase12["_match_ratio"] = phase12["match_ratio"]
        phase12["_gap_ratio"] = phase12["gap_ratio"]
        for idx, r in phase12.iterrows():
            if r["_result"] == "Mismatch":
                cls = classify_mismatch(r["_check"])
                print(f"{r['_check']}:  {cls}")
        print("================================================\n")
        phase12["_hard_ratio"] = phase12.apply(
            lambda r: r["mismatch_ratio"] if classify_mismatch(r["_check"]) == "Hard Mismatch" else 0.0,
            axis=1
        )

        phase12["_soft_ratio"] = phase12.apply(
            lambda r: r["mismatch_ratio"] if classify_mismatch(r["_check"]) == "Soft Mismatch" else 0.0,
            axis=1
        )
    else:
        # Ensure it has all columns if empty
        phase12["_match_ratio"] = 0.0
        phase12["_gap_ratio"] = 0.0
        phase12["_hard_ratio"] = 0.0
        phase12["_soft_ratio"] = 0.0
       # -------------------------------
        # DEBUG: PRINT AFTER CHANGES
        # -------------------------------
    print("\n=== DEBUG: PHASE 3 AFTER ratio computation ===")
    print(
            phase12[
                ["_phase", "_check", "_result",
                 "_match_ratio", "_mismatch_ratio", "_gap_ratio",
                 "_hard_ratio", "_soft_ratio"]
            ].to_string(index=False)
        )
    print("==============================================\n")

    # Combine
    df = pd.concat([phase12, phase3], ignore_index=True)
    print("\n=== DEBUG: NEW RATIO COLUMNS ONLY ===")
    cols = ['check', '_match_ratio', '_soft_ratio','_hard_ratio', '_gap_ratio']
    existing = [c for c in cols if c in df.columns]

    print("\n=== DEBUG: ALL ROWS FOR SELECTED COLUMNS ===")
    print(df[existing].to_string(index=False))
    print("Columns present:", existing)
    print("All dataframe columns:", df.columns.tolist())
    print("====================================\n")

    # ============================================================
    # Aggregate per viewpoint–pattern
    # ============================================================
    results = []

    for vp in VP_ORDER:
        for pat in PATTERN_ORDER:

            print("\n" + "=" * 80)
            print(f"VIEWPOINT = {vp} | PATTERN = {pat}")
            print("=" * 80)

            relevant = get_mismatches_for_viewpoint_and_pattern(vp, pat)

            print(f"Relevant checks for this VP+Pattern → {relevant}")

            if not relevant:
                print("→ No relevant checks → Adding N/A row.")
                results.append({
                    "Viewpoint": vp,
                    "Pattern": pat,
                    "Match": 0,
                    "Hard Mismatch": 0,
                    "Soft Mismatch": 0,
                    "Metadata Gap": 0,
                    "N/A": 1.0
                })
                continue

            sub = df[(df["_check"].isin(relevant)) & (df["pattern"] == pat)]

            print(f"Number of DF rows selected = {len(sub)}")
            if sub.empty:
                print("→ DF subset EMPTY → Adding N/A row.")
                results.append({
                    "Viewpoint": vp,
                    "Pattern": pat,
                    "Match": 0,
                    "Hard Mismatch": 0,
                    "Soft Mismatch": 0,
                    "Metadata Gap": 0,
                    "N/A": 1.0
                })
                continue

            print("\n--- Individual rows contributing to aggregation ---")
            for idx, row in sub.iterrows():
                print(f"• check={row['_check']}")
                print(f"  match={row['_match_ratio']}, "
                      f"hard={row['_hard_ratio']}, "
                      f"soft={row['_soft_ratio']}, "
                      f"gap={row['_gap_ratio']}\n")

            # --- compute means ---
            match_mean = sub["_match_ratio"].mean()
            hard_mean = sub["_hard_ratio"].mean()
            soft_mean = sub["_soft_ratio"].mean()
            gap_mean = sub["_gap_ratio"].mean()

            print("--- Averages BEFORE normalization ---")
            print(f"Match mean         = {match_mean}")
            print(f"Hard mismatch mean = {hard_mean}")
            print(f"Soft mismatch mean = {soft_mean}")
            print(f"Gap mean           = {gap_mean}")

            total = match_mean + hard_mean + soft_mean + gap_mean
            if total == 0:
                total = 1.0

            print(f"Total (for normalization) = {total}")

            results.append({
                "Viewpoint": vp,
                "Pattern": pat,
                "Match": match_mean / total,
                "Hard Mismatch": hard_mean / total,
                "Soft Mismatch": soft_mean / total,
                "Metadata Gap": gap_mean / total
            })

            print("--- Normalized (final) values ---")
            print(f"Final Match         = {match_mean / total:.3f}")
            print(f"Final Hard mismatch = {hard_mean / total:.3f}")
            print(f"Final Soft mismatch = {soft_mean / total:.3f}")
            print(f"Final Gap           = {gap_mean / total:.3f}")

    if not results:
        print("[WARNING] No data available.")
        return

    df_sum = pd.DataFrame(results)

    # ============================================================
    # Plot
    # ============================================================
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    axes = axes.flatten()

    for i, vp in enumerate(VP_ORDER):
        ax = axes[i]

        sub = df_sum[df_sum["Viewpoint"] == vp]
        if sub.empty:
            ax.axis("off")
            continue

        x = np.arange(len(sub["Pattern"]))

        match_vals = sub["Match"].values
        hard_vals = sub["Hard Mismatch"].values
        soft_vals = sub["Soft Mismatch"].values
        gap_vals = sub["Metadata Gap"].values
        na_vals = sub.get("N/A", pd.Series([0] * len(sub))).values

        ax.bar(x, match_vals, color=COLORS["Match"])
        ax.bar(x, hard_vals, bottom=match_vals, color=COLORS["Hard Mismatch"])
        ax.bar(x, soft_vals, bottom=match_vals + hard_vals, color=COLORS["Soft Mismatch"])
        ax.bar(x, gap_vals, bottom=match_vals + hard_vals + soft_vals, color=COLORS["Metadata Gap"])
        ax.bar(x, na_vals, bottom=match_vals + hard_vals + soft_vals + gap_vals, color=COLORS["N/A"])

        ax.set_xticks(x)
        ax.set_xticklabels(sub["Pattern"], rotation=30, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Proportion")
        ax.set_title(f"{string.ascii_lowercase[i]}) {vp}", fontsize=13, weight="bold")

    # Legend
    handles = [mpatches.Patch(color=COLORS[c], label=c) for c in ORDER4]
    axes[-1].axis("off")
    axes[-1].legend(handles=handles, loc="center")

    fig.tight_layout()
    fig.savefig(os.path.join(figs_dir, "viewpoint_pattern_summary2.png"), dpi=300)
    plt.close(fig)

    print("[INFO] Saved viewpoint_pattern_summary2.png")




def plot_viewpoint_pattern_summary(df: pd.DataFrame, figs_dir: str = "figs") -> None:
    """
    Generates a figure with one subplot per RM-ODP viewpoint.
    Each subplot shows five stacked bars (one per integration pattern),
    representing Match / Mismatch / Metadata Gap proportions
    across all mismatches relevant to both viewpoint and pattern.
    """

    os.makedirs(figs_dir, exist_ok=True)
    df = df.copy()

    # --- Normalize core columns ---
    df["_pattern"] = df["pattern"].astype(str).str.strip().str.title()
    df["_phase"] = df.get("phase", "")
    df["_check"] = df.get("check", "").astype(str)
    df["_result"] = (
        df.get("result", "")
        .astype(str)
        .str.strip()
        .str.title()
        .replace({"Gap": "Metadata Gap"})
    )

    # --- Convert categorical Phase 3 results to ratios ---
    phase3 = df[df["_phase"].str.contains("Phase 3", case=False, na=False)].copy()
    if not phase3.empty:
        phase3["_match_ratio"] = (phase3["_result"] == "Match").astype(float)
        phase3["_mismatch_ratio"] = (phase3["_result"] == "Mismatch").astype(float)
        phase3["_gap_ratio"] = (phase3["_result"].str.contains("Gap", case=False)).astype(float)
        print("\n=== DEBUG: Phase 3 ratio calculations ===")
        print(
            phase3[
                ["_phase", "_check", "_result", "_match_ratio", "_mismatch_ratio", "_gap_ratio"]
            ].to_string(index=False)
        )
        print("=========================================\n")

    # Keep Phase 1–2 ratios and combine with Phase 3
    phase12 = df[~df["_phase"].str.contains("Phase 3", case=False, na=False)].copy()
    df = pd.concat([phase12, phase3], ignore_index=True)
    print("\n=== DEBUG: NEW RATIO COLUMNS ONLY ===")
    cols = ['check', '_match_ratio', '_mismatch_ratio', '_gap_ratio']
    existing = [c for c in cols if c in df.columns]

    print("\n=== DEBUG: ALL ROWS FOR SELECTED COLUMNS ===")
    print(df[existing].to_string(index=False))
    print("Columns present:", existing)
    print("All dataframe columns:", df.columns.tolist())
    print("====================================\n")

    print(df.columns)

    # ============================================================
    # Compute aggregated ratios per viewpoint–pattern combination
    # ============================================================

    print("[DEBUG] NAFAS Unique mismatch types in df:")
    print(sorted(df["_check"].unique().tolist()))

    results = []

    for vp in VP_ORDER:
        print(f"\n[DEBUG] ▶ Viewpoint: {vp}")
        for pat in PATTERN_ORDER:
            relevant_mismatches = get_mismatches_for_viewpoint_and_pattern(vp, pat)

            if not relevant_mismatches:
                print(f"   ⚠️  Pattern {pat}: no relevant mismatches found.")
                continue

            print(f"   🔹 Pattern: {pat}")
            print(f"      Relevant mismatches ({len(relevant_mismatches)}): {relevant_mismatches}")

            # Removed pattern filter — use only the relevant mismatches
            sub = df[df["_check"].isin(relevant_mismatches)]
            n_rows = len(sub)
            print(f"      Rows in data for these mismatches: {n_rows}")

            if n_rows == 0:
                print("      ⚠️  No matching rows found in dataframe.")
                continue

            # Compute mean ratios
            match_mean = sub["_match_ratio"].mean(skipna=True)
            mismatch_mean = sub["_mismatch_ratio"].mean(skipna=True)
            gap_mean = sub["_gap_ratio"].mean(skipna=True)

            total = match_mean + mismatch_mean + gap_mean
            if total == 0:
                total = 1.0

            match_share = match_mean / total
            mismatch_share = mismatch_mean / total
            gap_share = gap_mean / total

            print(f"      → match={match_share:.2f}, mismatch={mismatch_share:.2f}, gap={gap_share:.2f}")

            results.append({
                "Viewpoint": vp,
                "Pattern": pat,
                "Match": match_share,
                "Mismatch": mismatch_share,
                "Metadata Gap": gap_share
            })

    if not results:
        print("[WARNING] No data available for viewpoint–pattern summary.")
        return

    df_sum = pd.DataFrame(results)

    # ============================================================
    # Plot: one subplot per viewpoint
    # ============================================================
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    axes = axes.flatten()

    for i, vp in enumerate(VP_ORDER):
        ax = axes[i]
        sub = df_sum[df_sum["Viewpoint"] == vp]
        if sub.empty:
            ax.axis("off")
            continue

        x = np.arange(len(sub["Pattern"]))
        match_vals = sub["Match"].values
        mismatch_vals = sub["Mismatch"].values
        gap_vals = sub["Metadata Gap"].values

        ax.bar(x, match_vals, color=COLORS["Match"], label="Match")
        ax.bar(x, mismatch_vals, bottom=match_vals, color=COLORS["Mismatch"], label="Mismatch")
        ax.bar(x, gap_vals, bottom=match_vals + mismatch_vals, color=COLORS["Metadata Gap"], label="Metadata Gap")

        ax.set_xticks(x)
        ax.set_xticklabels(sub["Pattern"], rotation=30, ha="right", fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Proportion", fontsize=12)
        ax.set_title(f"{string.ascii_lowercase[i]}) {vp}", fontsize=13, weight="bold")

    # Legend in last subplot
    handles = [mpatches.Patch(color=COLORS[k], label=k) for k in ORDER3]
    axes[-1].axis("off")
    axes[-1].legend(handles=handles, loc="center", frameon=True,
                    fancybox=True, fontsize=12, title="Categories", title_fontsize=13)

    # Hide any unused subplot
    for j in range(len(VP_ORDER), len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    path = os.path.join(figs_dir, "viewpoint_pattern_summary.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved {path}")


    # ------------------------------------------------------------

def plot_viewpoint_summary(df: pd.DataFrame, figs_dir: str = "figs") -> None:
    """
    Generates a stacked bar chart showing, for each RM-ODP viewpoint,
    the percentage of Match / Mismatch / Metadata Gap, using ratio-based
    scores for Phase 1–2 and categorical conversion for Phase 3.
    """

    os.makedirs(figs_dir, exist_ok=True)
    df = df.copy()

    # ------------------------------------------------------------
    # 1️⃣ Normalize result and check names
    # ------------------------------------------------------------
    df["result_norm"] = (
        df.get("result", "")
        .astype(str)
        .str.strip()
        .str.title()
        .replace({"Gap": "Metadata Gap"})
    )

    df["check"] = df.get("check", "").astype(str)
    if "CHECK_NORMALIZATION" in globals():
        df["check"] = df["check"].map(CHECK_NORMALIZATION).fillna(df["check"])

    df["phase"] = df.get("phase", "").astype(str)

    # ------------------------------------------------------------
    # 2️⃣ Prepare ratio columns for all phases
    # ------------------------------------------------------------
    for col in ["match_ratio", "mismatch_ratio", "gap_ratio"]:
        if col not in df.columns:
            df[col] = np.nan

    # --- Phase 3 categorical → numeric ---
    phase3 = df[df["phase"].str.contains("Phase 3", case=False, na=False)].copy()
    if not phase3.empty:
        phase3["match_ratio"] = (phase3["result_norm"] == "Match").astype(float)
        phase3["mismatch_ratio"] = (phase3["result_norm"] == "Mismatch").astype(float)
        phase3["gap_ratio"] = (phase3["result_norm"].str.contains("Gap", case=False)).astype(float)

    # --- Combine back with Phase 1 + 2 (already have ratios) ---
    phase12 = df[~df["phase"].str.contains("Phase 3", case=False, na=False)].copy()
    df = pd.concat([phase12, phase3], ignore_index=True)

    # ------------------------------------------------------------
    # 3️⃣ Compute weighted mean ratios per viewpoint
    # ------------------------------------------------------------
    data = []

    for vp in VIEWPOINT_FIELDS.keys():
        relevant = get_relevant_mismatches(vp)
        sub = df[df["check"].isin(relevant)]
        if sub.empty or sub["match_ratio"].isna().all():
            data.append({
                "Viewpoint": vp,
                "Match": 0,
                "Mismatch": 0,
                "Metadata Gap": 0,
                "N/A": 100  # 100% gray
            })
            continue

        # Compute mean ratios (skip NaNs)
        match_mean = sub["match_ratio"].mean(skipna=True)
        mismatch_mean = sub["mismatch_ratio"].mean(skipna=True)
        gap_mean = sub["gap_ratio"].mean(skipna=True)

        # Normalize so the three proportions sum to 1
        total = match_mean + mismatch_mean + gap_mean
        if total == 0:
            total = 1.0

        data.append({
            "Viewpoint": vp,
            "Match": (match_mean / total) * 100,
            "Mismatch": (mismatch_mean / total) * 100,
            "Metadata Gap": (gap_mean / total) * 100,
        })

    if not data:
        print("[WARNING] No data available for viewpoint summary chart.")
        return

    df_vp = pd.DataFrame(data).set_index("Viewpoint").reindex(VP_ORDER)

    # ------------------------------------------------------------
    # 4️⃣ Plot stacked bar chart
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(df_vp))

    for cat in ORDER3:
        vals = df_vp[cat].fillna(0).values
        ax.bar(df_vp.index, vals, bottom=bottom, color=COLORS[cat], label=cat)
        bottom += vals

    ax.set_ylabel("Percentage (%)", fontsize=14, weight="bold")
    ax.set_xlabel("Viewpoint", fontsize=14, weight="bold")
    ax.set_title("Viewpoint-wise Match / Mismatch / Gap Distribution", fontsize=16, weight="bold")
    ax.set_ylim(0, 100)
    ax.legend(title="Result", fontsize=12, title_fontsize=13)
    plt.xticks(rotation=30, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    fig.tight_layout()

    path = os.path.join(figs_dir, "viewpoint_match_mismatch_gap_distribution.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved {path}")


# ============================================================
import os, string, numpy as np, matplotlib.pyplot as plt, matplotlib.patches as mpatches
import pandas as pd
import os, string
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


VP_ORDER = ["Domain", "Information", "Computational", "Engineering", "Technology"]

ORDER3 = ["Match", "Mismatch", "Metadata Gap"]
COLORS = {
    "Match": "#4CAF50",
    "Mismatch": "#F44336",
    "Metadata Gap": "#FFC107",
}
PATTERN_ORDER = ["One-Way", "Loose", "Shared", "Integrated", "Embedded"]



# ------------------------------------------------------------
# Mapping normalization for mismatch check names
# ------------------------------------------------------------
CHECK_NORMALIZATION = {
    # Conceptual
    "Conceptual Consistency Mismatch": "Conceptual Consistency Mismatch",
    "Conceptual Quality Gap": "Conceptual Quality Gap",

    # License / Provenance
    "License Incompatibility": "License Incompatibility",
    "Provenance and Accessibility Gap": "Provenance and Accessibility Gap",

    # Information
    "Information Alignment Weighted": "Information Alignment Mismatch",
    "Information Alignment Mismatch": "Information Alignment Mismatch",

    # Computational / Engineering / Technology
    "I/O Library Compatibility": "I/O Library Compatibility Mismatch",
    "I/O Library Compatibility Mismatch": "I/O Library Compatibility Mismatch",
    "Shared Store Configuration Compatibility": "Shared Store Compatibility Mismatch",
    "Shared Store Compatibility Mismatch": "Shared Store Compatibility Mismatch",
    "Runtime Containment Compatibility": "Runtime Containment Mismatch",
    "Runtime Containment Mismatch": "Runtime Containment Mismatch",
    "Runtime Language Compatibility": "Runtime Compatibility Mismatch",
    "Runtime Compatibility Mismatch": "Runtime Compatibility Mismatch",
}


# ============================================================
# Generate Figures (Weighted Pie + Viewpoint Distribution)
# ============================================================

def generate_figures(df: pd.DataFrame, figs_dir: str = "figs", pattern_counts=None) -> None:
    """
    Generates weighted summary figures from the combined integration mismatch report.

    Produces:
      1️⃣ Weighted per-pattern pie charts (Match/Mismatch/Metadata Gap)
      2️⃣ Viewpoint-by-pattern stacked-bar chart (Domain, Information, etc.)

    Combines Phase 1 (conceptual/legal), Phase 2 (information), and Phase 3 (runtime).
    """

    # --------------------------------------------------------------
    # 🧩 Basic setup
    # --------------------------------------------------------------
    os.makedirs(figs_dir, exist_ok=True)
    df = df.copy()

    # Normalize essential columns
    for col in ["pattern", "check", "phase", "result"]:
        if col not in df.columns:
            df[col] = ""
    df["_pattern"] = df["pattern"].astype(str).str.strip().str.title()
    df["_check"] = df["check"].astype(str).map(lambda x: CHECK_NORMALIZATION.get(x.strip(), x))
    df["_phase"] = df["phase"].astype(str)
    df["_result"] = df["result"].astype(str).str.strip().str.title().replace({"Gap": "Metadata Gap"})

    # Ratio columns — ensure numeric
    for col in ["match_ratio", "mismatch_ratio", "gap_ratio"]:
        df[f"_{col}"] = pd.to_numeric(df.get(col, np.nan), errors="coerce")

    patterns = [p for p in PATTERN_ORDER if p in df["_pattern"].unique()]
    if not patterns:
        print("[WARNING] No recognizable integration patterns found.")
        print("[DEBUG] Unique patterns:", df["_pattern"].unique())
        return

    print("[INFO] Computing weighted summary per pattern…")

    # --------------------------------------------------------------
    # ⚖️ Compute weighted averages
    # --------------------------------------------------------------
    summary = []
    for pat in patterns:
        sub = df[df["_pattern"] == pat]
        if sub.empty:
            continue

        # Phase 1 + 2: use ratios (if available)
        phase12 = sub[sub["_phase"].isin(["Phase 1", "Phase 2"])]

        # Phase 3: convert categorical results to ratios
        phase3 = sub[sub["_phase"].str.contains("Phase 3", case=False, na=False)].copy()
        if not phase3.empty:
            phase3["_match_ratio"] = (phase3["_result"].eq("Match")).astype(float)
            phase3["_mismatch_ratio"] = (phase3["_result"].eq("Mismatch")).astype(float)
            phase3["_gap_ratio"] = (phase3["_result"].str.contains("Gap", case=False)).astype(float)

        combined = pd.concat([phase12, phase3], ignore_index=True)
        if combined.empty:
            continue

        match_mean = combined["_match_ratio"].mean(skipna=True)
        mismatch_mean = combined["_mismatch_ratio"].mean(skipna=True)
        gap_mean = combined["_gap_ratio"].mean(skipna=True)

        total = match_mean + mismatch_mean + gap_mean
        if total == 0:
            total = 1.0
        summary.append({
            "pattern": pat,
            "Match": match_mean / total,
            "Mismatch": mismatch_mean / total,
            "Metadata Gap": gap_mean / total
        })

    if not summary:
        print("[INFO] No valid data for summary plotting.")
        return

    df_summary = pd.DataFrame(summary).set_index("pattern").reindex(PATTERN_ORDER)

    # --------------------------------------------------------------
    # 1️⃣ Weighted Pie Charts – Match/Mismatch/Gap per pattern
    # --------------------------------------------------------------
    print("[INFO] Generating weighted per-pattern pie charts…")

    n = len(df_summary)
    cols, rows = 3, int(np.ceil(n / 3))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5.5), squeeze=False)

    for i, (pat, row) in enumerate(df_summary.iterrows()):
        ax = axes[i // cols][i % cols]
        data = [row["Match"], row["Mismatch"], row["Metadata Gap"]]
        if np.allclose(data, 0):
            ax.axis("off")
            continue

        wedges, _, autotexts = ax.pie(
            data,
            labels=None,
            autopct=lambda p: f"{p:.1f}%",
            colors=[COLORS[k] for k in ORDER3],
            textprops={"fontsize": 14, "weight": "bold"},
        )
        for t in autotexts:
            t.set_fontsize(14)
            t.set_weight("bold")

        ax.text(
            0.5, -0.08, f"{string.ascii_lowercase[i]}) {pat}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=16, weight="bold"
        )

    handles = [mpatches.Patch(color=COLORS[k], label=k) for k in ORDER3]
    if len(df_summary) < rows * cols:
        ax_leg = axes[len(df_summary) // cols][len(df_summary) % cols]
        ax_leg.axis("off")
        ax_leg.legend(handles=handles, loc="center", frameon=True,
                      fancybox=True, fontsize=15, title="Result", title_fontsize=17)

    for j in range(len(df_summary), rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.tight_layout()
    path1 = os.path.join(figs_dir, "weighted_match_mismatch_gap_by_pattern.png")
    fig.savefig(path1, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved {path1}")


    # --------------------------------------------------------------

    # 2️⃣ Viewpoint-by-pattern (stacked bar by viewpoint)
    print("[INFO] Generating viewpoint summary chart…")
    plot_viewpoint_summary(df, figs_dir=figs_dir)
    plot_viewpoint_pattern_summary(df, figs_dir=figs_dir)
    plot_viewpoint_pattern_summary2(df, figs_dir=figs_dir)
    plot_viewpoint_pattern_summary3(df, figs_dir=figs_dir,pattern_counts=pattern_counts)
    #plot_viewpoint_pies(df, figs_dir=figs_dir)
    plot_viewpoint_pies2(df, figs_dir=figs_dir)

# ============================================================
# CLI main
# ============================================================

def main(argv: List[str]) -> None:
    """
    Main entry point for mismatch detection.
    Scans YAML metadata files, extracts model information,
    evaluates A/B pairs using the mismatch detection rules,
    writes the detailed CSV report, and prints a summary.
    """
    # ------------------------------------------------------------
    # Parse command-line arguments
    # ------------------------------------------------------------
    debug = "--debug" in argv
    args = [a for a in argv if a != "--debug"]

    # ------------------------------------------------------------
    # Step 1: Locate YAML metadata files
    # ------------------------------------------------------------
    paths: List[str] = []
    import os
    print(os.getcwd())
    print(os.path.isdir("modelsMetadataFullV3"))

    default_root = "modelsMetadataFullV3" if os.path.isdir("modelsMetadataFullV3") else "."
    scan_roots = args or [default_root]

    for root_path in scan_roots:
        if os.path.isdir(root_path):
            for root, _, files in os.walk(root_path):
                for fn in files:
                    if fn.lower().endswith((".yml", ".yaml")):
                        paths.append(os.path.join(root, fn))
        elif root_path.lower().endswith((".yml", ".yaml")):
            paths.append(root_path)

    print(f"[INFO] Found {len(paths)} YAML files to process.")

    # ------------------------------------------------------------
    # Step 2: Load and extract model metadata
    # ------------------------------------------------------------
    models = load_yaml_models(paths)
    for m in models:
        try:
            extract_io(m, debug=debug)
        except Exception as e:
            print(f"[WARNING] Failed to extract IO for {m.file}: {e}")
            if debug:
                print(traceback.format_exc())

    # Group models by integration case (group ID)
    groups: Dict[str, List[ModelMeta]] = defaultdict(list)
    for m in models:
        groups[m.group].append(m)

    if debug:
        for gid, items in sorted(groups.items()):
            roles = ", ".join([f"{it.role}:{os.path.basename(it.file)}" for it in items])
            print(f"[DEBUG] Group {gid}: {roles}")

    # ------------------------------------------------------------
    # Step 3: Evaluate each A/B pair
    # ------------------------------------------------------------
    report_rows: List[Dict[str, Any]] = []
    for gid, items in sorted(groups.items()):
        A = next((x for x in items if x.role == "A"), None)
        B = next((x for x in items if x.role == "B"), None)
        AB_intended = next((x for x in items if x.role == "AB-INTENDED"), None)
        AB_integrated = next((x for x in items if x.role == "AB-INTEGRATED"), None)

        if not (A and B):
            print(f"[WARNING] Group {gid} is missing model(s): "
                  f"{'A' if not A else ''}{' and ' if (not A and not B) else ''}{'B' if not B else ''}")
            continue

        try:
            #if AB_intended:
             #   rows = evaluate_group(gid, A, B, AB_intended, debug=debug)
              #  for r in rows:
               #     r["ab_kind"] = "INTENDED"
               # report_rows.extend(rows)

            if AB_integrated:
                rows = evaluate_group(gid, A, B, AB_integrated, debug=debug)
                for r in rows:
                    r["ab_kind"] = "INTEGRATED"
                    report_rows.extend(rows)

        except Exception as e:
            print(f"[WARNING] Failed to evaluate group {gid}: {e}")
            if debug:
                print(traceback.format_exc())

    if not report_rows:
        print("[INFO] No A/B pairs found to score.")
        return

    # ------------------------------------------------------------
    # Step 4: Write report and generate summary
    # ------------------------------------------------------------
    pattern_counts = Counter()
    df = pd.DataFrame(report_rows)
    pattern_counts = (
        df[['group', 'pattern']]
        .drop_duplicates()
        .groupby('pattern')
        .size()
    )

    print("\n===== PATTERN COUNTS (per AB integration) =====")
    print(pattern_counts)
    print("==============================================\n")
    out_csv = "match_report.csv"
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Wrote {out_csv} with {len(df)} rows.")

    # ------------------------------------------------------------
    # Step 5: Summarize and print results
    # ------------------------------------------------------------
    summarize_results(report_rows)

    total = len(df)
    n_match = len(df[df["result"].str.lower() == "match"])
    success_rate = (n_match / total) * 100 if total > 0 else 0

    print("\n=== Overall Integration Success ===")
    print(f"Matched checks : {n_match}/{total} ({success_rate:.1f}%)")
    print("===================================\n")

    # Optional: Save summary as CSV for dashboards
    summary_csv = "summary_report.csv"
    summary_data = {
        "total_checks": [total],
        "matches": [n_match],
        "success_rate_percent": [round(success_rate, 1)]
    }
    pd.DataFrame(summary_data).to_csv(summary_csv, index=False)
    print(f"[INFO] Summary exported to {summary_csv}")

    # ------------------------------------------------------------
    # Step 6: Generate summary figures
    # ------------------------------------------------------------
    try:
        generate_figures(df, figs_dir="figsv1", pattern_counts=pattern_counts)
    except Exception as e:
        print(f"[WARNING] Failed to generate figures: {e}")

    if not paths:
        print("[ERROR] No YAML files found. Exiting.")
        return


# -----------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------
if __name__ == "__main__":
    main(sys.argv[1:])
