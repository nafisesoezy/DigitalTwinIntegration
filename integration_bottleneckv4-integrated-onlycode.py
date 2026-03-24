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
    # IO (explicit)
    "input": ["input"],
    "output": ["output"],
    "integrated_input": ["integrated_input"],
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

def check_phase1_conceptual_and_legal(group: str, a: ModelMeta, b: ModelMeta, ab: Optional[ModelMeta], pattern: str) -> List[Dict[str,Any]]:
    rows: List[Dict[str,Any]] = []

    # 1) Conceptual Consistency Mismatch
    # Fields: Scope, Purpose, Assumptions, Validation Capabilities, Conceptual Model Evaluation
    fields_cc = [
        ("Scope","scope"),
        ("Purpose","purpose_pattern"),
        ("Assumptions","assumptions"),
        ("Validation Capabilities","validation_capabilities"),
        ("Conceptual Model Evaluation","conceptual_model_evaluation"),
    ]
    for label, key in fields_cc:
        av = join_field_values(a, key)
        bv = join_field_values(b, key)
        abv = ""  # table does not require AB here
        if not av or not bv:
            rows.append(row(group, "Conceptual Consistency Mismatch", label, pattern,
                            "Both sides must provide comparable metadata", av, bv, abv,
                            "One or both sides missing.", "Missing"))
            continue
        sim = jaccard_token_similarity(av, bv)
        res = "Match" if sim > 0 else "Mismatch"
        det = f"A↔B token-sim={sim:.2f} (>0 ⇒ coherent)"
        rows.append(row(group, "Conceptual Consistency Mismatch", label, pattern,
                        "Token overlap (>0)", av, bv, abv, det, res))

    # 2) License Incompatibility
    key = "license"
    av = join_field_values(a, key)
    bv = join_field_values(b, key)
    if not av or not bv:
        rows.append(row(group, "License Incompatibility", "License", pattern,
                        "Both sides must declare license", av, bv, "",
                        "One or both sides missing license.", "Missing"))
    else:
        sim = jaccard_token_similarity(av, bv)
        res = "Match" if sim > 0 else "Mismatch"
        det = f"A↔B token-sim={sim:.2f} (>0 ⇒ compatible family)"
        rows.append(row(group, "License Incompatibility", "License", pattern,
                        "Token overlap (>0)", av, bv, "", det, res))

    # 3) Provenance and Accessibility Gap
    key = "landing_page"
    av = join_field_values(a, key)
    bv = join_field_values(b, key)
    # presence check only
    if av and bv:
        res, det = "Match", "Both provide Landing Page / PID."
    elif not av and not bv:
        res, det = "Missing", "Neither provides Landing Page."
    else:
        res, det = "Missing", "Only one provides Landing Page."
    rows.append(row(group, "Provenance and Accessibility Gap", "Landing Page", pattern,
                    "Presence of persistent link/landing page", av, bv, "", det, res))

    # 4) Conceptual Quality Gap
    # Fields: Links to Publications & Reports, Authors’ ID, Conceptual Model Evaluation,
    #         Calibration Tools/Data, Validation Capabilities, Sensitivity Analysis, Uncertainty Analysis
    fields_q = [
        ("Links to Publications & Reports","links_to_publications_and_reports"),
        ("Authors’ ID","authors_unique_identifier"),
        ("Conceptual Model Evaluation","conceptual_model_evaluation"),
        ("Calibration Tools/Data","calibration_tools_data"),
        ("Validation Capabilities","validation_capabilities"),
        ("Sensitivity Analysis","sensitivity_analysis"),
        ("Uncertainty Analysis","uncertainty_analysis"),
    ]
    for label, key in fields_q:
        av = join_field_values(a, key)
        bv = join_field_values(b, key)
        if av and bv:
            res, det = "Match","Both provide evidence/metadata."
        elif not av and not bv:
            res, det = "Missing","Neither provides metadata."
        else:
            res, det = "Missing","Only one provides metadata."
        rows.append(row(group, "Conceptual Quality Gap", label, pattern,
                        "Presence on both sides", av, bv, "", det, res))
    return rows

# ============================================================
# --------- PHASE 2: Information Alignment (table-aligned) ----
# ============================================================

def check_phase2_information_alignment(group: str, a: ModelMeta, b: ModelMeta, ab: Optional[ModelMeta], pattern: str) -> List[Dict[str,Any]]:
    rows: List[Dict[str,Any]] = []

    # Activate edges by pattern
    patt = pattern or ""
    edges: List[Tuple[str, List[str], List[str]]] = []  # (edge_label, src_out_vars, dst_in_vars)
    if patt == "One-Way":
        edges.append(("A→B", a.outputs.variables, b.inputs.variables))
    elif patt in ("Loose","Shared"):
        edges.append(("A→B", a.outputs.variables, b.inputs.variables))
        edges.append(("B→A", b.outputs.variables, a.inputs.variables))
    else:  # Integrated / Embedded / unspecified => allow both if present
        edges.append(("A→B", a.outputs.variables, b.inputs.variables))
        edges.append(("B→A", b.outputs.variables, a.inputs.variables))

    # (i) Variables — semantic overlap along active edges
    for elabel, src_out, dst_in in edges:
        if not src_out or not dst_in:
            res, det = "Missing", "Source outputs or destination inputs are missing."
        else:
            ok, pairs = any_semantic_overlap(src_out, dst_in, min_sim=0.50)
            res = "Match" if ok else "Mismatch"
            det = pretty_pairs(pairs) or "No variable pairs with sufficient similarity (≥0.50)."
        rows.append(row(group, "Information Alignment Mismatch",
                        f"Variables ({elabel})", pattern,
                        "At least one semantically similar variable per active edge",
                        "; ".join(src_out) or "(none)", "; ".join(dst_in) or "(none)",
                        "", det, res))

    # Helper to compare model-level tokens for each listed field
    def compare_info_field(label: str, key: str, prefer_src: Optional[str] = None):
        # You asked for token-based checks using the exact fields listed in your table.
        # These fields are model-level descriptors; we compare A vs B directly.
        av = join_field_values(a, key)
        bv = join_field_values(b, key)
        abv = "; ".join((ab.fields.get(key, []) if ab else []))
        if not av or not bv:
            res, det = "Missing", "One or both sides missing metadata."
        else:
            sim = jaccard_token_similarity(av, bv)
            res = "Match" if sim > 0 else "Mismatch"
            det = f"A↔B token-sim={sim:.2f} (>0 ⇒ aligned)"
        rows.append(row(group, "Information Alignment Mismatch", label, pattern,
                        "Token overlap (>0)", av, bv, abv, det, res))

    # (ii) Remaining Phase-2 fields from your table:
    compare_info_field("Model's Unique Identifier", "unique_identifier")
    compare_info_field("Parameters", "parameters")
    compare_info_field("Input Datasets", "datasets")
    compare_info_field("Output (schema/description)", "data")
    compare_info_field("Dimensionality", "dimensionality")
    compare_info_field("Spatial Resolution", "spatial_resolution")
    compare_info_field("Variable Spatial Resolution", "variable_spatial_resolution")
    compare_info_field("Time Steps/Temporal Resolution", "time_steps_temporal_resolution")
    compare_info_field("Variable Temporal Resolution", "variable_temporal_resolution")
    compare_info_field("Resampling/Conversion Policies (per variable)", "resampling_policies")

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

# === ABI / FFI families (for language compatibility) ===
ABI_FAMILIES = {
    "JVM": {"Java", "Kotlin", "Scala", "Groovy"},
    ".NET": {"C#", "F#", "VB.NET", "PowerShell"},
    "Native_C_CPP": {"C", "C++"},
}

FFI_FAMILIES = {
    "C_ABI": {"Python", "Dart", "Node.js", "C", "C++"},
    "JNI": {"Java", "Kotlin", "Scala", "Groovy"},
    "PInvoke": {"C#", "F#", "VB.NET", "PowerShell"},
    "MetaFFI": {"Python", "Java", "C#", "C++", "Kotlin", "Node.js"},
}

def _norm_lang(s: str) -> str:
    return (s or "").strip().lower()

def same_abi(lang1: str, lang2: str) -> bool:
    l1, l2 = _norm_lang(lang1), _norm_lang(lang2)
    for _, langs in ABI_FAMILIES.items():
        family = { _norm_lang(x) for x in langs }
        if l1 in family and l2 in family:
            return True
    return False

def same_ffi(lang1: str, lang2: str) -> bool:
    l1, l2 = _norm_lang(lang1), _norm_lang(lang2)
    for _, langs in FFI_FAMILIES.items():
        family = { _norm_lang(x) for x in langs }
        if l1 in family and l2 in family:
            return True
    return False

# --- helpers for Phase 3 ---
def _get_joined(model: Optional[ModelMeta], key: str) -> str:
    if not model:
        return ""
    return "; ".join(model.fields.get(key, []))

def _presence3(a: str, b: str, ab: str = "") -> tuple[bool,bool,bool]:
    return bool(a), bool(b), bool(ab)

def _result_row(group: str, name: str, field_label: str, pattern: str,
                req: str, av: str, bv: str, abv: str, detail: str, res: str) -> Dict[str, Any]:
    return row(group, name, field_label, pattern, req, av, bv, abv, detail, res)

def _token_match_nonempty(av: str, bv: str, thr: float = 0.0) -> Tuple[str, str]:
    """
    If either side empty -> Missing. Else token Jaccard > thr => Match, else Mismatch.
    Returns (res, detail).
    """
    if not av and not bv:
        return "Missing", "Neither side provided metadata."
    if not av or not bv:
        return "Missing", "Only one side provided metadata."
    sim = jaccard_token_similarity(av, bv)
    return ("Match" if sim > thr else "Mismatch", f"A↔B token-sim={sim:.2f} (> {thr:.2f})")

# ----------------------- All Patterns -----------------------

def r3_execution_reproducibility_gap(
    group: str, a: ModelMeta, b: ModelMeta, ab: Optional[ModelMeta], pattern: str
) -> List[Dict[str, Any]]:
    """
    Phase 3 — Execution Reproducibility Gap
    ----------------------------------------------------------
    Checks whether each model individually provides complete and executable
    metadata in its ExecutionInstructions. A model is considered reproducible
    if the following fields are present and non-empty:
        - exec_type (script, binary, api, container, workflow, library)
        - entry_point (main executable or function)
        - build_system (make, cmake, setup.py, docker, etc.)
          [required unless exec_type ∈ {binary, api}]
    Optional but recommended:
        - scheduler
        - command_args

    Missing any required field ⇒ GAP (type = Reproducibility Gap)
    """
    def check_exec_repro_fields(model: ModelMeta) -> Tuple[bool, List[str]]:
        required = ["exec_type", "entry_point"]
        missing = []

        # Always required
        for f in required:
            vals = model.fields.get(f, [])
            if not vals or not any(v.strip() for v in vals):
                missing.append(f)

        # Conditionally required build_system
        etype = (model.fields.get("exec_type", [""])[0]).strip().lower() if model.fields.get("exec_type") else ""
        if etype not in {"binary", "api"}:
            vals = model.fields.get("build_system", [])
            if not vals or not any(v.strip() for v in vals):
                missing.append("build_system")

        return (len(missing) == 0, missing)

    rows = []
    # Evaluate both models separately (A and B)
    for label, model in [("Model A", a), ("Model B", b)]:
        ok, missing = check_exec_repro_fields(model)
        result_type = "Match" if ok else "Gap"
        gap_type = "Reproducibility Gap" if not ok else ""
        av = "; ".join(
            [f"{k}={','.join(v)}" for k, v in model.fields.items()
             if k in ("exec_type", "entry_point", "scheduler", "build_system", "command_args")]
        )
        detail = (
            "All required execution metadata present and complete."
            if ok
            else f"Missing required fields: {', '.join(missing)}"
        )

        rows.append(
            _result_row(
                group,
                "Execution Reproducibility Gap",
                "Execution Instructions",
                pattern,
                "Execution metadata completeness (exec_type, entry_point, build_system)",
                av,
                "",
                "",
                f"{detail} | Type: {gap_type}" if gap_type else detail,
                result_type,
            )
        )

    return rows



def r3_dependency_compat_mismatch(group: str, a: ModelMeta, b: ModelMeta, ab: Optional[ModelMeta], pattern: str) -> List[Dict[str,Any]]:
    """
    Fields: Software Requirements
    Rule: token overlap > 0 => Match, else Mismatch/Missing.
    """
    rows = []
    key = "software_specification_and_requirements"
    av = _get_joined(a, key)
    bv = _get_joined(b, key)
    res, det = _token_match_nonempty(av, bv, thr=0.0)
    rows.append(_result_row(group, "Dependency Compatibility Mismatch", "Software Requirements", pattern,
                            "Token overlap > 0 implies compatible deps/versions.",
                            av, bv, "", det, res))
    return rows

def r3_environment_consistency_mismatch(group: str, a: ModelMeta, b: ModelMeta, ab: Optional[ModelMeta], pattern: str) -> List[Dict[str,Any]]:
    """
    Fields: Distribution Version
    Rule: token overlap > 0 => Match, else Mismatch/Missing.
    """
    rows = []
    key = "distribution_version"
    av = _get_joined(a, key)
    bv = _get_joined(b, key)
    res, det = _token_match_nonempty(av, bv, thr=0.0)
    rows.append(_result_row(group, "Environment Consistency Mismatch", "Distribution Version", pattern,
                            "Token overlap > 0 implies compatible runtime distros.",
                            av, bv, "", det, res))
    return rows

# ----------------------- Loose only -------------------------

def r3_comm_stability_mismatch(group: str, a: ModelMeta, b: ModelMeta, ab: Optional[ModelMeta], pattern: str) -> List[Dict[str,Any]]:
    """
    Fields: Latency Expectations, Error Handling
    Rule: Both fields should exist for both sides; token overlap > 0 on each pair.
    """
    rows = []
    # Latency
    la, lb = _get_joined(a, "latency_expectations"), _get_joined(b, "latency_expectations")
    resL, detL = _token_match_nonempty(la, lb, thr=0.0)
    rows.append(_result_row(group, "Communication Stability Mismatch", "Latency Expectations", pattern,
                            "Latency expectations overlap.", la, lb, "", detL, resL))
    # Error handling
    ea, eb = _get_joined(a, "error_handling"), _get_joined(b, "error_handling")
    resE, detE = _token_match_nonempty(ea, eb, thr=0.0)
    rows.append(_result_row(group, "Communication Stability Mismatch", "Error Handling", pattern,
                            "Error policies overlap.", ea, eb, "", detE, resE))
    return rows

def r3_coordinated_repro_gap(group: str, a: ModelMeta, b: ModelMeta, ab: Optional[ModelMeta], pattern: str) -> List[Dict[str,Any]]:
    """
    Fields: Execution Instructions, Execution Constraints
    Rule: token overlap > 0 for both fields.
    """
    rows = []
    # Exec instructions
    ia, ib = _get_joined(a, "execution_instructions"), _get_joined(b, "execution_instructions")
    r1, d1 = _token_match_nonempty(ia, ib, thr=0.0)
    rows.append(_result_row(group, "Coordinated Reproducibility Gap", "Execution Instructions", pattern,
                            "Overlap in orchestration steps.", ia, ib, "", d1, r1))
    # Exec constraints
    ca, cb = _get_joined(a, "execution_constraints"), _get_joined(b, "execution_constraints")
    r2, d2 = _token_match_nonempty(ca, cb, thr=0.0)
    rows.append(_result_row(group, "Coordinated Reproducibility Gap", "Execution Constraints", pattern,
                            "Overlap in timing/ordering constraints.", ca, cb, "", d2, r2))
    return rows

# ------------- Shared / Integrated / Embedded ---------------

def r3_deterministic_sync_mismatch(group: str, a: ModelMeta, b: ModelMeta, ab: Optional[ModelMeta], pattern: str) -> List[Dict[str,Any]]:
    """
    Fields: Execution Instructions, Data Synchronization
    Rule: token overlap > 0 on both fields.
    """
    rows = []
    # Data synchronization
    sa, sb = _get_joined(a, "data_synchronization"), _get_joined(b, "data_synchronization")
    rS, dS = _token_match_nonempty(sa, sb, thr=0.0)
    rows.append(_result_row(group, "Deterministic Synchronization Mismatch", "Data Synchronization", pattern,
                            "Synchronization policies overlap.", sa, sb, "", dS, rS))
    # Execution instructions (event cadence/ordering)
    ia, ib = _get_joined(a, "execution_instructions"), _get_joined(b, "execution_instructions")
    rI, dI = _token_match_nonempty(ia, ib, thr=0.0)
    rows.append(_result_row(group, "Deterministic Synchronization Mismatch", "Execution Instructions", pattern,
                            "Event cadence/ordering overlap.", ia, ib, "", dI, rI))
    return rows

def r3_concurrency_policy_mismatch(group: str, a: ModelMeta, b: ModelMeta, ab: Optional[ModelMeta], pattern: str) -> List[Dict[str,Any]]:
    """
    Fields: Execution Constraints, Support for Parallel Execution (parallel_execution)
    Rule: token overlap > 0 on constraints; and presence+overlap on parallel intent.
    """
    rows = []
    # Constraints
    ca, cb = _get_joined(a, "execution_constraints"), _get_joined(b, "execution_constraints")
    rC, dC = _token_match_nonempty(ca, cb, thr=0.0)
    rows.append(_result_row(group, "Concurrency Policy Mismatch", "Execution Constraints", pattern,
                            "Overlap in locking/ordering policies.", ca, cb, "", dC, rC))
    # Parallel execution
    pa, pb = _get_joined(a, "parallel_execution"), _get_joined(b, "parallel_execution")
    rP, dP = _token_match_nonempty(pa, pb, thr=0.0)
    rows.append(_result_row(group, "Concurrency Policy Mismatch", "Support for Parallel Execution", pattern,
                            "Overlap in parallel intent.", pa, pb, "", dP, rP))
    return rows

def r3_shared_store_config_mismatch(group: str, a: ModelMeta, b: ModelMeta, ab: Optional[ModelMeta], pattern: str) -> List[Dict[str,Any]]:
    """
    Fields: Software Requirements, Distribution Version
    Rule: both fields token overlap > 0.
    """
    rows = []
    sra, srb = _get_joined(a, "software_specification_and_requirements"), _get_joined(b, "software_specification_and_requirements")
    rSR, dSR = _token_match_nonempty(sra, srb, thr=0.0)
    rows.append(_result_row(group, "Shared Store Configuration Mismatch", "Software Requirements", pattern,
                            "Shared store/driver/client overlap.", sra, srb, "", dSR, rSR))
    dva, dvb = _get_joined(a, "distribution_version"), _get_joined(b, "distribution_version")
    rDV, dDV = _token_match_nonempty(dva, dvb, thr=0.0)
    rows.append(_result_row(group, "Shared Store Configuration Mismatch", "Distribution Version", pattern,
                            "Runtime/driver version overlap.", dva, dvb, "", dDV, rDV))
    return rows

# ----------------------- Integrated only --------------------

def r3_runtime_compat_mismatch(group: str, a: ModelMeta, b: ModelMeta, ab: Optional[ModelMeta], pattern: str) -> List[Dict[str,Any]]:
    """
    Fields: Programming Language
    Rule: Match if same language OR same ABI family OR same FFI family.
    """
    rows = []
    key = "programming_language"
    av = _get_joined(a, key)
    bv = _get_joined(b, key)
    if not av and not bv:
        res, det = "Missing", "Neither side provided programming language."
    elif not av or not bv:
        res, det = "Missing", "Only one side provided programming language."
    else:
        # Compare first tokens (treat ';' multi-values conservatively by first token)
        la = av.split(";")[0].strip()
        lb = bv.split(";")[0].strip()
        eq = _norm_lang(la) == _norm_lang(lb)
        abi_ok = same_abi(la, lb)
        ffi_ok = same_ffi(la, lb)
        ok = eq or abi_ok or ffi_ok
        res = "Match" if ok else "Mismatch"
        det = f"eq={eq}, same_abi={abi_ok}, same_ffi={ffi_ok}"
    rows.append(_result_row(group, "Runtime Compatibility Mismatch", "Programming Language", pattern,
                            "Same language OR same ABI/FFI family.", av, bv, "", det, res))
    return rows

def r3_error_resilience_gap(group: str, a: ModelMeta, b: ModelMeta, ab: Optional[ModelMeta], pattern: str) -> List[Dict[str,Any]]:
    """
    Fields: Error Handling
    Rule: token overlap > 0.
    """
    rows = []
    key = "error_handling"
    av = _get_joined(a, key)
    bv = _get_joined(b, key)
    res, det = _token_match_nonempty(av, bv, thr=0.0)
    rows.append(_result_row(group, "Error Resilience Gap", "Error Handling", pattern,
                            "Overlap in error/recovery policies.", av, bv, "", det, res))
    return rows

# ------------------------ Embedded only ---------------------

def r3_runtime_containment_mismatch(group: str, a: ModelMeta, b: ModelMeta, ab: Optional[ModelMeta], pattern: str) -> List[Dict[str,Any]]:
    """
    Fields: Programming Language
    Rule: Embedded requires host+guest to run in one runtime. Match if same language OR same ABI family.
          (FFI alone may not guarantee in-process containment.)
    """
    rows = []
    key = "programming_language"
    av = _get_joined(a, key)
    bv = _get_joined(b, key)
    if not av and not bv:
        res, det = "Missing", "Neither side provided programming language."
    elif not av or not bv:
        res, det = "Missing", "Only one side provided programming language."
    else:
        la = av.split(";")[0].strip()
        lb = bv.split(";")[0].strip()
        eq = _norm_lang(la) == _norm_lang(lb)
        abi_ok = same_abi(la, lb)
        ok = eq or abi_ok
        res = "Match" if ok else "Mismatch"
        det = f"eq={eq}, same_abi={abi_ok} (FFI not counted for containment)."
    rows.append(_result_row(group, "Runtime Containment Mismatch", "Programming Language", pattern,
                            "Same language or same ABI required for in-process embedding.",
                            av, bv, "", det, res))
    return rows

def r3_memory_safety_mismatch(group: str, a: ModelMeta, b: ModelMeta, ab: Optional[ModelMeta], pattern: str) -> List[Dict[str,Any]]:
    """
    Fields: Execution Constraints, Support for Parallel Execution
    Rule: token overlap > 0 on constraints and parallel intent (need both present).
    """
    rows = []
    ca, cb = _get_joined(a, "execution_constraints"), _get_joined(b, "execution_constraints")
    rC, dC = _token_match_nonempty(ca, cb, thr=0.0)
    rows.append(_result_row(group, "Memory Safety Mismatch", "Execution Constraints", pattern,
                            "Overlap in thread/memory safety constraints.", ca, cb, "", dC, rC))
    pa, pb = _get_joined(a, "parallel_execution"), _get_joined(b, "parallel_execution")
    rP, dP = _token_match_nonempty(pa, pb, thr=0.0)
    rows.append(_result_row(group, "Memory Safety Mismatch", "Support for Parallel Execution", pattern,
                            "Overlap in parallel intent.", pa, pb, "", dP, rP))
    return rows

def r3_host_sync_mismatch(group: str, a: ModelMeta, b: ModelMeta, ab: Optional[ModelMeta], pattern: str) -> List[Dict[str,Any]]:
    """
    Fields: Data Synchronization
    Rule: token overlap > 0.
    """
    rows = []
    key = "data_synchronization"
    av = _get_joined(a, key)
    bv = _get_joined(b, key)
    res, det = _token_match_nonempty(av, bv, thr=0.0)
    rows.append(_result_row(group, "Host Synchronization Mismatch", "Data Synchronization", pattern,
                            "Overlap in in-memory sync policies.", av, bv, "", det, res))
    return rows

def r3_error_propagation_gap(group: str, a: ModelMeta, b: ModelMeta, ab: Optional[ModelMeta], pattern: str) -> List[Dict[str,Any]]:
    """
    Fields: Error Handling
    Rule: token overlap > 0 (ensure exceptions map across host/guest boundary).
    """
    rows = []
    key = "error_handling"
    av = _get_joined(a, key)
    bv = _get_joined(b, key)
    res, det = _token_match_nonempty(av, bv, thr=0.0)
    rows.append(_result_row(group, "Error Propagation Gap", "Error Handling", pattern,
                            "Overlap in exception mapping/propagation.", av, bv, "", det, res))
    return rows

# ============================================================
# Pattern-aware Phase 3 dispatcher (calls relevant mismatch fns)
# ============================================================

def run_phase3_runtime_checks(group: str, a: ModelMeta, b: ModelMeta, ab: Optional[ModelMeta], pattern: str) -> List[Dict[str,Any]]:
    """
    Selects and runs the Phase 3 mismatch functions based on the integration pattern,
    exactly as specified in the table.
    """
    rows: List[Dict[str,Any]] = []

    # --- All patterns (always run) ---
    rows += r3_execution_reproducibility_gap(group, a, b, ab, pattern)
    rows += r3_dependency_compat_mismatch(group, a, b, ab, pattern)
    rows += r3_environment_consistency_mismatch(group, a, b, ab, pattern)

    patt = (pattern or "").strip()

    # --- Pattern-specific blocks ---
    if patt == "One-Way":
        # (No extra Phase 3 mismatches beyond "All patterns")
        return rows

    if patt == "Loose":
        rows += r3_comm_stability_mismatch(group, a, b, ab, patt)
        rows += r3_coordinated_repro_gap(group, a, b, ab, patt)
        return rows

    if patt == "Shared":
        rows += r3_deterministic_sync_mismatch(group, a, b, ab, patt)
        rows += r3_concurrency_policy_mismatch(group, a, b, ab, patt)
        rows += r3_shared_store_config_mismatch(group, a, b, ab, patt)
        return rows

    if patt == "Integrated":
        rows += r3_deterministic_sync_mismatch(group, a, b, ab, patt)
        rows += r3_concurrency_policy_mismatch(group, a, b, ab, patt)
        rows += r3_runtime_compat_mismatch(group, a, b, ab, patt)
        rows += r3_error_resilience_gap(group, a, b, ab, patt)
        return rows

    if patt == "Embedded":
        rows += r3_deterministic_sync_mismatch(group, a, b, ab, patt)
        rows += r3_concurrency_policy_mismatch(group, a, b, ab, patt)
        rows += r3_runtime_containment_mismatch(group, a, b, ab, patt)
        rows += r3_memory_safety_mismatch(group, a, b, ab, patt)
        rows += r3_host_sync_mismatch(group, a, b, ab, patt)
        rows += r3_error_propagation_gap(group, a, b, ab, patt)
        return rows

    # Fallback if pattern is unspecified: treat like Shared (conservative)
    rows += r3_deterministic_sync_mismatch(group, a, b, ab, patt or "(unspecified)")
    rows += r3_concurrency_policy_mismatch(group, a, b, ab, patt or "(unspecified)")
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
    rows += check_phase1_conceptual_and_legal(gid, A, B, AB, pattern)
    # rows += check_conceptual_quality_gap(gid, A, B, AB, pattern)
    # Phase 2
    rows += check_phase2_information_alignment(gid, A, B, AB, pattern)
    # Phase 3 (new pattern-aware runtime dispatcher)
    rows += run_phase3_runtime_checks(gid, A, B, AB, pattern)
    return rows

# -----------------------------------------------------------------
# Summary helper (must appear before main)
# -----------------------------------------------------------------
from collections import Counter, defaultdict

def summarize_results(rows: List[Dict[str, Any]]) -> None:
    """
    Summarize total number of Matches, Gaps, and Mismatches across all checks.
    Also prints counts by specific gap/mismatch subtype when available.
    """
    type_counter = Counter()
    subtype_counter = defaultdict(Counter)

    for r in rows:
        result = r.get("result", "").strip()
        if not result:
            continue
        type_counter[result] += 1
        if result == "Gap":
            # Extract subtype if encoded in detail or bottleneck
            subtype = ""
            if "Type:" in r.get("detail", ""):
                subtype = r["detail"].split("Type:")[-1].strip()
            elif "Gap" in r.get("bottleneck", ""):
                subtype = r["bottleneck"]
            subtype_counter["Gap"][subtype or "Generic Gap"] += 1
        elif result == "Mismatch":
            subtype_counter["Mismatch"][r["bottleneck"]] += 1

    total = sum(type_counter.values())
    print("\n=== Integration Analysis Summary ===")
    print(f"Total checks: {total}")
    for k in ["Match", "Gap", "Mismatch"]:
        print(f"{k:10s}: {type_counter[k]}")

    # Breakdown by subtype
    if subtype_counter["Gap"]:
        print("\n--- Gap Types ---")
        for subtype, count in subtype_counter["Gap"].items():
            print(f"{subtype:35s}: {count}")

    if subtype_counter["Mismatch"]:
        print("\n--- Mismatch Types ---")
        for subtype, count in subtype_counter["Mismatch"].items():
            print(f"{subtype:35s}: {count}")

    print("====================================\n")



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
    default_root = "modelsMetadataFull" if os.path.isdir("modelsMetadataFull") else "."
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
    if not paths:
        print("[ERROR] No YAML files found. Exiting.")
        return

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
            if AB_intended:
                rows = evaluate_group(gid, A, B, AB_intended, debug=debug)
                for r in rows:
                    r["ab_kind"] = "INTENDED"
                report_rows.extend(rows)

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
    df = pd.DataFrame(report_rows)
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

# -----------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------
if __name__ == "__main__":
    main(sys.argv[1:])
