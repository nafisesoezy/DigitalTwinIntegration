#!/usr/bin/env python3
# Updated for the RM-ODP constraint-template pipeline (see README.md).
"""
constraint_templates.py

The environmental ConstraintTemplate registry described in the paper
(Section 4.2 "Pattern-Aware Compatibility Assessment" and the Appendix
Table `environmental_constraint_templates`, Section 5.2).

This module is the single source of truth for:
  - which ConstraintCategory (PatternAgnostic / InformationLevel / RuntimeLevel)
    a compatibility condition belongs to,
  - which RM-ODP viewpoint it is reported under (for Fig. viewpoint_macroF1),
  - which EvaluationMode is PREDEFINED for it (Deterministic / LLMAssisted) --
    per Section 4.3, this is fixed by the template, never chosen at run time
    by the rule engine or the LLM.

It does NOT implement the checks themselves. It is a lookup layer that the
existing rule engine (integration_bottleneckv6.py) and the new hybrid
evaluator (hybrid_evaluate.py) both import, so a single (bottleneck, field)
combination is classified identically everywhere -- in match_report.csv,
in the LLM stage, and in every Results*/figures.py script.

Usage:
    from constraint_templates import classify
    tmpl = classify(bottleneck="Semantic Mismatch", field="Scope")
    tmpl.name              -> "Scope Compatibility"
    tmpl.category          -> "PatternAgnostic"
    tmpl.viewpoint         -> "Domain"
    tmpl.evaluation_mode   -> "LLMAssisted"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class ConstraintTemplate:
    name: str                # Appendix Table "Constraint Template" column
    category: str            # PatternAgnostic | InformationLevel | RuntimeLevel
    viewpoint: str            # Domain | Information | Computational | Engineering | Technology
    evaluation_mode: str      # Deterministic | LLMAssisted
    concern: str              # Table `environmental_constraint_summary` "Compatibility Concern"
    in_paper_appendix: bool = True   # False = legacy/auxiliary field kept for backward
                                      # compatibility with older reports, but NOT one of
                                      # the 19 templates in Appendix B. Excluded by default
                                      # from Table 1 / Fig 7-8 scoring so numbers stay
                                      # reproducible against the paper as written.


# ---------------------------------------------------------------------------
# 1) Templates keyed by the rule engine's `bottleneck` label (integration_
#    bottleneckv6.py bottleneck names), for constraints where every field
#    under that bottleneck maps to the SAME template.
# ---------------------------------------------------------------------------
_BY_BOTTLENECK: Dict[str, ConstraintTemplate] = {
    "Conceptual Quality Gap": ConstraintTemplate(
        "Conceptual Quality Evidence", "PatternAgnostic", "Domain",
        "LLMAssisted", "Conceptual Quality"),

    "Temporal Resolution Mismatch": ConstraintTemplate(
        "Temporal Resolution Compatibility", "InformationLevel", "Information",
        "Deterministic", "Information Alignment"),
    "Spatial Resolution Mismatch": ConstraintTemplate(
        "Spatial Resolution Compatibility", "InformationLevel", "Information",
        "Deterministic", "Information Alignment"),
    "Dimensionality Mismatch": ConstraintTemplate(
        "Dimensionality Compatibility", "InformationLevel", "Information",
        "Deterministic", "Information Alignment"),
    "Data Schema Mismatch": ConstraintTemplate(
        "Variable Semantic Compatibility", "InformationLevel", "Information",
        "LLMAssisted", "Information Alignment"),
    "File Format Mismatch": ConstraintTemplate(
        "Data Format Compatibility", "InformationLevel", "Information",
        "Deterministic", "Information Alignment"),

    "Communication Mechanism Mismatch": ConstraintTemplate(
        "Interface Compatibility", "RuntimeLevel", "Computational",
        "LLMAssisted", "Integration Interface"),
    "Error Handling Mismatch": ConstraintTemplate(
        "Error-Handling Compatibility", "RuntimeLevel", "Computational",
        "LLMAssisted", "Integration Interface"),

    "Parallel Execution Incompatibility": ConstraintTemplate(
        "Execution Constraint Compatibility", "RuntimeLevel", "Engineering",
        "LLMAssisted", "Runtime Coordination"),
    "Acknowledgment Protocol Mismatch": ConstraintTemplate(
        "Synchronization Compatibility", "RuntimeLevel", "Engineering",
        "LLMAssisted", "Runtime Coordination"),
    "Latency Expectation Mismatch": ConstraintTemplate(
        "Latency Compatibility", "RuntimeLevel", "Engineering",
        "LLMAssisted", "Runtime Coordination"),
    "Data Synchronization": ConstraintTemplate(
        "Synchronization Compatibility", "RuntimeLevel", "Engineering",
        "LLMAssisted", "Runtime Coordination"),

    "Programming Language Incompatibility": ConstraintTemplate(
        "Execution Environment Compatibility", "RuntimeLevel", "Technology",
        "LLMAssisted", "Technological Compatibility"),
    "Software Environment Mismatch": ConstraintTemplate(
        "Execution Environment Compatibility", "RuntimeLevel", "Technology",
        "LLMAssisted", "Technological Compatibility"),
    "Hardware Resource Mismatch": ConstraintTemplate(
        "Execution Environment Compatibility", "RuntimeLevel", "Technology",
        "LLMAssisted", "Technological Compatibility"),
    "Distribution Version Mismatch": ConstraintTemplate(
        "Execution Environment Compatibility", "RuntimeLevel", "Technology",
        "LLMAssisted", "Technological Compatibility"),
    "Execution Instruction Gap": ConstraintTemplate(
        "Execution Environment Compatibility", "RuntimeLevel", "Technology",
        "LLMAssisted", "Technological Compatibility"),
    "License Incompatibility": ConstraintTemplate(
        "License Compatibility", "RuntimeLevel", "Technology",
        "LLMAssisted", "Technological Compatibility"),

    # --- Not yet checked by integration_bottleneckv6.py; registered here so
    #     classify() and the F1-by-template report show them as an explicit
    #     GAP-producing template rather than silently absent. See the
    #     "Known coverage gaps" note at the bottom of this file.
    "Operating Environment Mismatch": ConstraintTemplate(
        "Operating Environment Compatibility", "RuntimeLevel", "Technology",
        "Deterministic", "Technological Compatibility"),
    "Unit Mismatch": ConstraintTemplate(
        "Unit Compatibility", "InformationLevel", "Information",
        "Deterministic", "Information Alignment"),
    "Execution Ordering Mismatch": ConstraintTemplate(
        "Execution Ordering Compatibility", "RuntimeLevel", "Engineering",
        "Deterministic", "Runtime Coordination"),
}

# ---------------------------------------------------------------------------
# 2) "Semantic Mismatch" and "Execution Constraint Mismatch" fan out to
#    DIFFERENT templates depending on `field`, so they need field-level
#    overrides on top of the bottleneck-level table above.
# ---------------------------------------------------------------------------
_BY_BOTTLENECK_FIELD: Dict[Tuple[str, str], ConstraintTemplate] = {
    ("Semantic Mismatch", "Scope"): ConstraintTemplate(
        "Scope Compatibility", "PatternAgnostic", "Domain",
        "LLMAssisted", "Semantic Compatibility"),
    ("Semantic Mismatch", "Purpose & Pattern"): ConstraintTemplate(
        "Purpose Compatibility", "PatternAgnostic", "Domain",
        "LLMAssisted", "Semantic Compatibility"),
    ("Semantic Mismatch", "Assumptions"): ConstraintTemplate(
        "Assumption Compatibility", "PatternAgnostic", "Domain",
        "LLMAssisted", "Semantic Compatibility"),

    # execution_constraints is evaluated by TWO templates in the paper:
    # a deterministic ordering check (vs. the active dependency edges) and
    # a broader LLM-assisted judgment of whether the declared constraints
    # are compatible with the integration's requirements. The rule engine
    # currently emits ONE row for this field (a sync-token / pattern-graded
    # rule) -- classify it as the deterministic "Execution Ordering
    # Compatibility" template, and run "Execution Constraint Compatibility"
    # as an *additional* LLM-assisted evaluation over the same evidence
    # (see hybrid_evaluate.py: EXTRA_LLM_TEMPLATES).
    ("Execution Constraint Mismatch", "execution_constraints"): ConstraintTemplate(
        "Execution Ordering Compatibility", "RuntimeLevel", "Engineering",
        "Deterministic", "Runtime Coordination"),
}

# ---------------------------------------------------------------------------
# 3) Fields that are extracted and checked by the rule engine but are NOT
#    among the paper's 19 constraint templates (Table
#    environmental_constraint_templates / environmental_constraint_summary).
#    They are legacy/auxiliary bookkeeping fields (title, description,
#    keywords, model type, provenance links, source-code/verification
#    presence, landing pages, ...). They are still tagged with a viewpoint
#    (so the RM-ODP field-completeness figures in Section 6.1 keep working
#    unchanged) but `in_paper_appendix=False` so the Section 6.2 detector
#    metrics (Table detection_performance, Fig viewpoint/pattern macroF1)
#    can filter them out and reproduce exactly the templates the paper
#    defines.
# ---------------------------------------------------------------------------
_AUXILIARY_FIELDS: Dict[Tuple[str, str], ConstraintTemplate] = {}


def _aux(bottleneck: str, field: str, viewpoint: str) -> None:
    _AUXILIARY_FIELDS[(bottleneck, field)] = ConstraintTemplate(
        name=f"(auxiliary) {bottleneck} / {field}",
        category="PatternAgnostic" if viewpoint == "Domain" else "RuntimeLevel",
        viewpoint=viewpoint,
        evaluation_mode="Deterministic",
        concern="Auxiliary / not in Appendix B",
        in_paper_appendix=False,
    )


for _f in ("Title", "Model Version", "Description", "Keywords", "Model Type"):
    _aux("Semantic Mismatch", _f, "Domain")

_aux("Temporal Coverage Mismatch", "temporal_extent_coverage", "Information")
_aux("Spatial Coverage Mismatch", "spatial_extent_coverage", "Information")

for _f in ("availability_of_source_code (A)", "availability_of_source_code (B)"):
    _aux("Source Code Availability Gap", _f, "Technology")
for _f in ("implementation_verification (A)", "implementation_verification (B)"):
    _aux("Implementation Verification Gap", _f, "Technology")
for _f in ("landing_page (A)", "landing_page (B)"):
    _aux("Landing Page Gap", _f, "Technology")

del _f


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def classify(bottleneck: str, field: str) -> Optional[ConstraintTemplate]:
    """
    Map a rule-engine (bottleneck, field) pair to its ConstraintTemplate.
    Returns None only for combinations this registry has never seen --
    treat that as "needs a registry entry", not as a silent Gap.
    """
    key2 = (bottleneck, field)
    if key2 in _BY_BOTTLENECK_FIELD:
        return _BY_BOTTLENECK_FIELD[key2]
    if key2 in _AUXILIARY_FIELDS:
        return _AUXILIARY_FIELDS[key2]
    # Data Schema Mismatch has many field spellings (A.output, B.output,
    # "(A or B).input vs AB.input", "Direction (any)", ...) that all share
    # one template -- match by bottleneck only for that case, and for every
    # other bottleneck that has one template regardless of field.
    if bottleneck in _BY_BOTTLENECK:
        return _BY_BOTTLENECK[bottleneck]
    return None


def all_templates() -> Dict[str, ConstraintTemplate]:
    """All templates that ARE in the paper's Appendix B, keyed by template name."""
    seen: Dict[str, ConstraintTemplate] = {}
    for t in list(_BY_BOTTLENECK.values()) + list(_BY_BOTTLENECK_FIELD.values()):
        if t.in_paper_appendix:
            seen[t.name] = t
    return seen


if __name__ == "__main__":
    # Self-check against the paper: Appendix Table environmental_constraint_
    # templates lists 19 templates across 6 concerns. Print what this
    # registry currently defines, so a diff against the paper table is a
    # one-glance operation.
    tmpls = all_templates()
    print(f"{len(tmpls)} constraint templates registered (paper Appendix B lists 19):\n")
    by_concern: Dict[str, list] = {}
    for t in tmpls.values():
        by_concern.setdefault(t.concern, []).append(t)
    for concern, ts in by_concern.items():
        print(f"[{concern}]")
        for t in sorted(ts, key=lambda x: x.name):
            print(f"  - {t.name:<38} category={t.category:<16} viewpoint={t.viewpoint:<13} mode={t.evaluation_mode}")
        print()
