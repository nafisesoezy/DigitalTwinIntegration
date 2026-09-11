# Digital Twin Model Integration — Compatibility Assessment

Code and data for the RM-ODP-guided metadata schema and pattern-aware,
LLM-assisted compatibility assessment framework (Sections 4-6), instantiated
for environmental-modeling model integration.

This README documents, in order: what the pipeline actually does, how to
run every step that produces a number or figure in the paper, where the
metadata comes from, how the LLM-assisted stage is designed, and — explicitly
— what its safeguards and limitations are.

**Scope note.** This repository implements the RM-ODP-guided metadata
schema and the pattern-aware/LLM-assisted compatibility assessment described
in Sections 4-5, instantiated for the environmental-modeling case study in
Section 5 and evaluated in Section 6. It does not generate executable
integration code or perform runtime orchestration — its output is a
compatibility report (Match/Mismatch/Gap + evidence + explanation +
candidate adaptation) for a proposed integration, produced before
implementation.

---

## 1. Repository structure

```
constraint_templates.py         Registry of the paper's Appendix Table
                                 `environmental_constraint_templates`: for every
                                 (bottleneck, field) pair the rule engine can
                                 emit, returns its ConstraintTemplate name,
                                 ConstraintCategory (PatternAgnostic /
                                 InformationLevel / RuntimeLevel), RM-ODP
                                 viewpoint, and PREDEFINED EvaluationMode
                                 (Deterministic / LLMAssisted). Single source
                                 of truth used by both stages below.

integration_bottleneckv6.py     STAGE 1 + the Deterministic half of STAGE 2
                                 (Algorithm 1, Section 4.3.4): loads the YAML
                                 metadata, infers the integration pattern,
                                 instantiates all applicable constraints
                                 (pattern-agnostic + information-level over
                                 activated data-flow edges + runtime-level),
                                 evaluates every Deterministic-mode constraint
                                 with an explicit rule, and writes
                                 match_report.csv enriched with the
                                 constraint_template/category/viewpoint/
                                 evaluation_mode/verdict/explanation/adaptation
                                 columns (via enrich_with_templates()).
                                 Also regenerates the Section 6.1 expert-survey
                                 figures (figs/*.png).

hybrid_evaluate.py               STAGE 2, LLMAssisted half (Algorithm 1): for
                                 every constraint whose template says
                                 EvaluationMode=LLMAssisted, calls the LLM to
                                 determine the verdict FROM EVIDENCE (not by
                                 re-triaging a rule-based guess) and to
                                 produce an explanation + candidate adaptation.
                                 Deterministic-mode rows are passed through
                                 unchanged (their verdict already comes from
                                 integration_bottleneckv6.py). Supersedes
                                 `llm_mismatch_solver_basedonMismatchReport_v3.py`
                                 (kept in the repo for provenance/audit of the
                                 pipeline's evolution — do not run it as part
                                 of the current reproduction path).

merge2.py                        Builds the ground-truth column used by
                                 Section 6.2's Table `detection_performance`:
                                 for each constraint, the ground truth is the
                                 DETERMINISTIC rule engine's own verdict when
                                 applied to the realized (AB / "INTEGRATED")
                                 model's metadata — never an LLM's opinion.
                                 See §8 for why this matters and what its
                                 remaining limitation is.

merge3.py                        Drops rows with no evaluable ground truth
                                 (`result == "Missing"`), producing the
                                 *_withoutMissing.csv variant some of the
                                 figure scripts read.

Resultsv2-2-withoutMissing.py    Produces Table `detection_performance`
                                 (accuracy/precision/recall/F1 per method,
                                 "mismatch" as the positive class) as
                                 detection_performance.tex.

Resultsv3-1-ViewPoints-Chartv3.py   Fig. `viewpoint_macroF1` (F1 by RM-ODP
                                     viewpoint) -> fig_viewpoint_macroF1_with_missing.png.
Resultsv4-Patterns-Chartv2.py       Fig. `pattern_macroF1` (F1 by integration
                                     pattern) -> fig_pattern_macroF1_with_missing.png.
Resultsv5-bestModel.py              Figs. 9-11 (descriptive Match/Mismatch/Gap
                                     distributions for the best-performing model).

modelsMetadataFullV3/            THE metadata corpus evaluated in Section 6:
                                 11 integration configurations x
                                 {A, B, AB-Intended, AB-Integrated} = 44 models.
                                 See §3 for provenance.
removed/                         Earlier candidate model YAMLs considered
                                 during curation and later excluded/replaced
                                 (e.g., an earlier candidate for group 1 was
                                 later replaced). Kept for audit purposes;
                                 not part of the evaluated 44.
Model repository/all/            Source PDFs the 44 model descriptions were
                                 extracted from (see §3).

figs/, Figures/                  Generated figures. `figs/` is written by
                                 integration_bottleneckv6.py (expert-survey
                                 and rule-engine descriptive plots); `Figures/`
                                 is written by Resultsv5-bestModel.py. The
                                 fig_*.png / *.tex files at the repository
                                 root are the outputs already produced by the
                                 §4 reproduction commands below, checked in so
                                 the paper's exact figures/tables are visible
                                 without re-running the pipeline.
```

Files not listed above (`integration_bottleneckv4*`, `integration_bottleneckv5.py`,
`Results.py`/`Resultsv2*.py` other than `-2-withoutMissing`,
`Resultsv3-1-Viewpoints.py`, `Resultsv3-Viewpoints.py`,
`Resultsv3-1-ViewPoints-Chartv1.py`, `Resultsv3-1-ViewPoints-Chartv2.py`,
`Resultsv4-Patterns.py`, `Resultsv4-Patterns-Chartv1.py`,
`llm_mismatch_solver_basedon*` other than `_v3`, `merge.py`,
`keyword_bottlenecks.py`, `figures.py`) are earlier iterations kept for
provenance/audit of how the pipeline evolved. **They are not part of the
reproduction path in §4 and should not be run** — where a script name in
this README doesn't match the paper text you're reading, this file is the
current source of truth.

---

## 2. Requirements

```bash
python -m pip install pandas numpy pyyaml matplotlib seaborn requests
```

LLM access (Stage 2, LLMAssisted constraints only — Stage 1 and the
Deterministic half of Stage 2 need no network access at all):

```bash
export LLM_BASE_URL="https://<your-inference-endpoint>"
export LLM_API_KEY="<your-key>"
```

The endpoint is expected to expose an OpenAI-compatible
`/api/v0/chat/completions` route. Model identifiers (`LLM_MODELS`, or the
three models in `DEFAULT_MODELS` inside `hybrid_evaluate.py`) must be served
by that endpoint — this repository was run against OpenAI GPT-OSS-120B,
Mistral Small 3.2 24B Instruct, and Llama 3.3 70B Instruct (AWQ) (aliased as
`default-text-large` on the inference infrastructure used for the paper).

---

## 3. Metadata provenance (addresses: "no description how the models have
## been created and where they come from")

The 44 models in `modelsMetadataFullV3/` were produced by the four-phase
schema-engineering process in Section 4.2.2 / Section 5.1:

1. Each model's RM-ODP-viewpoint metadata (Domain, Information, Computational,
   Engineering, Technology fields per Table `env_viewpoint_fields`) was
   extracted manually from its source publication. The source PDFs are
   archived under `Model repository/all/` (filename pattern
   `n<N>_<pattern>-<firstauthor>-<title-fragment>.pdf`; `<N>` matches the
   group number used everywhere else, e.g. `n1_oneway-...pdf` corresponds to
   group `1`).
2. Extracted metadata was cross-checked against the paper's ten participating
   domain experts (Section 6.1) for consistency and completeness.
3. Each YAML filename follows `<group>-<role>-<name>.yaml`, where role is
   `A`, `B`, `AB-Intended` (the pre-implementation Integration Specification,
   minimal metadata) or `AB-Integrated` (the realized, documented system).
   `integration_bottleneckv6.py`'s `guess_group_role_from_filename()` parses
   this convention; renaming a file breaks grouping.
4. `removed/` retains earlier candidate models considered for a group before
   a better-documented alternative was selected (audit trail only — these
   are not part of the 44 evaluated models).

**Known limitation, stated explicitly rather than left implicit:** metadata
extraction was manual and is therefore only as complete as what each source
publication reported — this is exactly why the schema includes an explicit
`Gap` verdict (missing evidence) as distinct from `Mismatch` (evidence of
incompatibility), and why Section 6.1's per-viewpoint completeness ratings
matter as a companion result to Section 6.2's detection metrics.

---

## 4. Reproducing every result in Section 6, in order

```bash
# 1) Deterministic Stage 1 + 2 (rule engine). No network access needed.
#    Regenerates match_report.csv, mismatch_reports/, integration_reports/,
#    and the Section 6.1 expert-survey figures under figs/.
python integration_bottleneckv6.py modelsMetadataFullV3

# 2) LLM-assisted Stage 2 (needs LLM_BASE_URL / LLM_API_KEY, see §2).
#    Reads match_report.csv, writes allLLM_match_report.csv.
python hybrid_evaluate.py

# 3) Build the ground-truth column from the realized (AB/"INTEGRATED") rows.
#    Writes allLLM_match_report_groundTruth.csv.
python merge2.py

# 4) (optional) drop rows with no evaluable ground truth.
python merge3.py

# 5) Table `detection_performance` (Section 6.2.2).
python Resultsv2-2-withoutMissing.py        # -> detection_performance.tex

# 6) Fig. viewpoint_macroF1 / pattern_macroF1 (Section 6.2.2).
python Resultsv3-1-ViewPoints-Chartv3.py    # -> fig_viewpoint_macroF1_with_missing.png
python Resultsv4-Patterns-Chartv2.py        # -> fig_pattern_macroF1_with_missing.png

# 7) Figs. 9-11 (best-model descriptive distributions).
python Resultsv5-bestModel.py
```

Every script reads/writes CSV in the working directory using the filenames
above; none of them take positional CSV arguments beyond what's shown (check
each script's top-of-file env-var list if you need to redirect paths). The
repository ships with the outputs of a prior full run already checked in
(`allLLM_match_report*.csv`, `detection_performance*.tex`, `fig_*.png`) so
the paper's exact artifacts are visible without an LLM key; re-running step
2 onward with your own credentials will regenerate them from the current
code.

---

## 5. The constraint-template registry (addresses: "no single example how
## this rule may look like")

`constraint_templates.py` is a plain data registry: for `bottleneck="Data
Schema Mismatch"`, or the specific `("Semantic Mismatch", "Scope")` pair, it
returns the paper's Appendix `ConstraintTemplate` — name, category,
viewpoint, and predefined evaluation mode. It does not itself decide
Match/Mismatch/Gap. A concrete, complete rule — Unit Compatibility,
`environmental_constraint_templates` row 4, Deterministic —
looks like this end to end (`integration_bottleneckv6.py`,
`check_unit_compatibility`):

```python
def _units_equivalent(u1: str, u2: str) -> bool:
    n1, n2 = _normalize_unit(u1), _normalize_unit(u2)
    if n1 == n2:
        return True
    return any(n1 in cls and n2 in cls for cls in _UNIT_EQUIV_CLASSES)

# for each matched (source_output, target_input) variable pair on an
# activated data-flow edge:
if not u_src or not u_dst:
    verdict = "Gap"        # unit not declared on one or both sides
elif _units_equivalent(u_src, u_dst):
    verdict = "Match"      # e.g. "degC" vs "celsius"
else:
    verdict = "Mismatch"   # e.g. degrees Celsius vs Kelvin -- conversion required
```

This is intentionally a conservative equivalence check (a fixed table of
known-equal spellings), not general unit algebra — stated here rather than
implied, since it is a real limitation: an as-yet-unlisted equivalent unit
pair reads as a false "Mismatch" until added to `_UNIT_EQUIV_CLASSES`.

**Every other Deterministic template follows the same shape**: an explicit,
inspectable Python function in `integration_bottleneckv6.py` (grep for
`check_` to find all of them), each returning `Match`, `Mismatch`, or `Gap`
plus the evidence and a one-line `detail` string that becomes the row's
`explanation` (see `enrich_with_templates()` /
`_local_explanation_and_adaptation()`).

Two templates the paper's Appendix defines were previously unimplemented:
**Unit Compatibility** and **Operating Environment Compatibility**. Both are
now implemented (`check_unit_compatibility`, `check_operating_environment`).
Operating Environment Compatibility will legitimately return `Gap` for most
of the current corpus, because `modelsMetadataFullV3/*.yaml` does not
populate an Operating System field for most models — that is a genuine
Technology-viewpoint completeness gap (consistent with Section 6.1's finding
that the Technology and Engineering viewpoints have the lowest completeness
ratings), not a bug in the check.

---

## 6. LLM-assisted evaluation: prompt design, context selection, robustness
## (addresses: "prompt design, context-selection logic, handling of
## unstable outputs... not explained with enough depth")

**Which constraints go to the LLM, and why.** Only constraints whose
template has `evaluation_mode == "LLMAssisted"` in `constraint_templates.py`
are ever sent to the LLM (see the table in that file's module docstring for
the full list — e.g. Purpose/Scope/Assumption Compatibility, Conceptual
Quality Evidence, Variable Semantic Compatibility, Interface/Error-Handling
Compatibility, Execution Constraint/Synchronization/Latency Compatibility,
Execution Environment Compatibility, License Compatibility). This is a
static, predefined assignment made when the template was designed — not a
decision the LLM or the rule engine makes at run time (Section 4.3.3).

**Context-selection logic.** For each LLM-assisted row, the prompt includes
exactly: the constraint template's name, the `field`/`required_check` it
targets, the extracted `A_value`/`B_value`/`AB_value` evidence (truncated to
400 characters each — long free-text fields are clipped rather than omitted,
to keep chunk sizes and latency bounded), and the selected integration
`pattern`. No metadata outside what that specific template lists as required
is included — this mirrors Section 4.3.2's constraint-instantiation step,
which binds a template only to the metadata it declares as needed.

**Prompt strategy.** One JSON call handles a chunk of `LLM_CHUNK_SIZE` rows
(default 20) to bound both latency and the chance of a truncated response.
The system message pins the model to `"Output strictly valid JSON only. No
markdown, no prose."`; `temperature=0.1` trades response diversity for
consistency. The evaluator and reasoner steps are combined into one JSON
schema per row (`{verdict, explanation, adaptation}`) for efficiency, while
keeping `verdict` (the LLM-Assisted Evaluator's output) and
`explanation`/`adaptation` (the LLM-Assisted Reasoner's output) as separate
fields, so the two conceptual roles in Algorithm 1 remain distinguishable in
the output even though they share one HTTP round trip.

**Handling unstable outputs.**
- Up to `LLM_MAX_RETRIES` (default 4) retries with exponential backoff on
  timeouts or transport errors (`hybrid_evaluate.call_llm_json`).
- A response that isn't valid JSON after fence-stripping raises immediately
  rather than being silently reinterpreted.
- A row the model's JSON never mentions (`row_ref` missing from `results`)
  is recorded as `Gap` with an explicit "No LLM response returned for this
  row" explanation — it is never silently dropped or defaulted to `Match`.
- A verdict outside `{Match, Mismatch, Gap}` is coerced to `Gap`, not
  discarded.
- A chunk that exhausts all retries is recorded as `Error` for every row in
  that chunk, with the underlying exception message kept in the explanation
  column, so failures are visible in the output CSV rather than silently
  skipped.

---

## 7. Safeguards and limitations of the LLM-assisted stage (addresses: "No
## LLM safeguards in place")

**What exists today:**
- The verdict for every Deterministic-mode constraint is fixed by an
  explicit rule and is never seen, let alone overridable, by any LLM call
  (see `hybrid_evaluate.run_one_model`: Deterministic rows are copied from
  `verdict`/`explanation`/`adaptation` before any LLM is invoked).
- LLM output is constrained to a fixed JSON schema with a 3-way verdict
  enum; anything else is coerced to `Gap` (§6).
- `temperature=0.1` and a fixed system prompt reduce (but do not eliminate)
  run-to-run variance.
- Evidence given to the LLM is restricted to the specific metadata fields
  the constraint template declares as required (§6) — it is not handed the
  full model description, which limits (but does not prevent) it inventing
  connections between unrelated fields.

**What does not exist yet, stated plainly:**
- No retrieval-augmented grounding (RAG) against the source publications —
  the LLM reasons only over the extracted metadata fields, not the original
  text, so it cannot verify its own explanation against the source.
- No self-consistency / majority-voting across repeated calls to the same
  model — each verdict is a single sample per (model, constraint) pair.
- No independent verifier model or human-in-the-loop confirmation step
  before a verdict is recorded in the compatibility report.
- No confidence score is elicited or reported alongside the verdict.

These are exactly the kind of mitigations that are missing today (e.g.,
RAG-based verification); we list them here as explicit, scoped future work
rather than as an implicit gap, and the paper's discussion section should
state the same limitation rather than imply the current pipeline already
guards against LLM confabulation.

---

## 8. Ground-truth methodology for Table `detection_performance`

For Section 6.2.2's predictive evaluation, each method's prediction (from
the component models A, B and the pre-implementation Integration
Specification) is compared against the realized integration AB, which
serves as ground truth. **The ground truth is computed by the deterministic
rule engine applied to AB's own declared metadata (`merge2.py`,
`GT_SOURCE_COL = "result"`) — never by any LLM's opinion of AB.** This
matters because scoring an LLM method against that same LLM's own output on
a different input row is circular by construction; the deterministic,
LLM-independent alternative used here is not.

**Known residual limitation, not yet fixed:** three Deterministic templates
(Temporal Resolution, Spatial Resolution, and Dimensionality Compatibility)
compare the two component models' own declared values and additionally
check AB's declared `Resampling/Conversion Policies` field for evidence that
a discrepancy was intentionally bridged (`integration_bottleneckv6.py`,
`info_simple` loop). Where the AB metadata does not document a
resampling/conversion policy at all — currently the common case in
`modelsMetadataFullV3/` — the ground truth on these three templates remains
close to the rule-based method's own prediction mechanism, which still
inflates the rule-based baseline's apparent agreement with "ground truth" on
that subset. Curating the `Resampling/Conversion Policies` field more
completely for the 11 evaluated configurations, or replacing ground truth on
these three templates with expert annotation, would remove this residual
bias; until then, Table `detection_performance` numbers on those three
templates specifically should be read as provisional, and the paper should
say so rather than presenting the aggregate F1 as uniformly reliable across
all constraint templates.

---

## 9. Applying this to a different domain

The RM-ODP viewpoint structure, the two-stage instantiate/evaluate
procedure, and the Match/Mismatch/Gap verdict vocabulary are
domain-independent (Section 4). To instantiate the framework for a domain
other than environmental modeling:

1. Define your domain's metadata fields per viewpoint (Table
   `env_viewpoint_fields` equivalent) and add them to `FIELD_ALIASES` in
   `integration_bottleneckv6.py`.
2. Define your domain's `ConstraintTemplate`s in a new registry module with
   the same shape as `constraint_templates.py`: name, category
   (PatternAgnostic/InformationLevel/RuntimeLevel), viewpoint, and a
   **predefined** evaluation mode — decide Deterministic vs. LLMAssisted at
   this design stage, not at run time.
3. Implement one `check_*` function per Deterministic template (see §5 for
   the shape) and point `hybrid_evaluate.py`'s `EVIDENCE_COLS`/prompt at
   the same evidence fields for your LLMAssisted templates.
4. Re-derive ground truth (§8) using your domain's own deterministic,
   AB-grounded rule, or via expert annotation — do not reuse an LLM's own
   output as ground truth for evaluating that LLM.

---

## 10. Environment-variable reference

| Variable | Used by | Default |
|---|---|---|
| `LLM_BASE_URL` | `hybrid_evaluate.py` | *(required for Stage 2)* |
| `LLM_API_KEY` | `hybrid_evaluate.py` | *(required for Stage 2)* |
| `LLM_CHAT_ENDPOINT` | `hybrid_evaluate.py` | `{LLM_BASE_URL}/api/v0/chat/completions` |
| `LLM_MODELS` | `hybrid_evaluate.py` | the 3 models in `DEFAULT_MODELS` |
| `LLM_CHUNK_SIZE` | `hybrid_evaluate.py` | `20` |
| `LLM_TIMEOUT_S` | `hybrid_evaluate.py` | `600` |
| `LLM_MAX_RETRIES` | `hybrid_evaluate.py` | `4` |
| `REASON_DETERMINISTIC_VIA_LLM` | `hybrid_evaluate.py` | `0` (set `1` to also have the LLM rewrite Deterministic-row explanations; never changes their verdict) |
| `MATCH_REPORT_PATH` | `hybrid_evaluate.py` | `match_report.csv` |
| `ALL_LLM_MATCH_REPORT_PATH` | `hybrid_evaluate.py` | `allLLM_match_report.csv` |
