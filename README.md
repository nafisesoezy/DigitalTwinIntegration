# Digital Twin Model Integration — Compatibility Assessment

## 1. Overview and Scope

This repository provides the implementation and evaluation material for the multi-viewpoint, pattern-aware compatibility assessment framework presented in the paper. The framework supports **pre-implementation assessment of model compositions for a specified Digital Twin (DT) integration objective**.

The approach structures model and integration descriptions across the five RM-ODP viewpoints—**Domain, Information, Computational, Engineering, and Technology**—and represents compatibility requirements as reusable constraint templates. For a particular Integration Specification (IS), the selected integration pattern determines which constraints apply. These constraints are instantiated for the participating models and evaluated using either a predefined deterministic check or LLM-assisted assessment, producing **Match, Mismatch, or Gap** outcomes together with supporting explanations and candidate adaptations.

The repository contains the implementation of this assessment procedure, its environmental-modeling instantiation, the metadata used in the empirical evaluation, and the associated analysis scripts and outputs.

The current empirical evaluation is limited to **environmental modeling**. The underlying metamodel, viewpoint structure, integration-pattern representation, constraint-template mechanism, and compatibility outcomes are defined independently of environmental semantics; however, the concrete metadata fields and compatibility constraints used here were instantiated and validated for this domain. Applying the framework elsewhere therefore requires domain-specific instantiation and validation.

The framework assesses compatibility **before implementation**. Generation of executable integration configurations and runtime orchestration are outside the scope of this repository.

## 2. Repository Structure

```text
constraint_templates.py
    Environmental compatibility constraint templates and their predefined
    evaluation modes.

integration_bottleneckv6.py
    Pattern-aware constraint instantiation and deterministic compatibility
    assessment.

hybrid_evaluate.py
    LLM-assisted compatibility assessment and generation of structured
    explanations and candidate adaptations.

modelsMetadataFullV3/
    Environmental-model metadata used in the empirical evaluation.

Model repository/all/
    Source publications used to construct the evaluated model descriptions.

merge2.py / merge3.py
    Preparation of reference outcomes used in the empirical evaluation.

Results*.py
    Scripts for computing evaluation metrics and generating reported results.

figs/ and Figures/
    Generated evaluation figures and visualizations.

```

Earlier implementation and analysis scripts are retained for provenance. The files listed above represent the main components corresponding to the current framework and evaluation.

## 3. Requirements and Running the Framework

### Requirements

The implementation requires Python and the following packages:

```bash
python -m pip install pandas numpy pyyaml matplotlib seaborn requests
```

The deterministic compatibility checks run locally and do not require access to an external model.

The LLM-assisted assessment requires an OpenAI-compatible inference endpoint:

```bash
export LLM_BASE_URL="https://<your-inference-endpoint>"
export LLM_API_KEY="<your-key>"
```

The LLM model identifiers can be configured through `LLM_MODELS`. The models and default configuration used in the empirical study are documented in `hybrid_evaluate.py`.

### Run the compatibility assessment

Run the pattern-aware constraint instantiation and deterministic assessment over the provided environmental-model metadata:

```bash
python integration_bottleneckv6.py modelsMetadataFullV3
```

This loads the model and Integration Specification metadata, identifies the selected integration pattern and relevant data-flow dependencies, instantiates the applicable constraint templates, and evaluates constraints assigned to deterministic evaluation.

To subsequently evaluate constraints whose predefined evaluation mode is `LLMAssisted`, run:

```bash
python hybrid_evaluate.py
```

The resulting assessment records the applicable constraint, relevant metadata evidence, evaluation mode, **Match/Mismatch/Gap** verdict, explanation, and candidate adaptation.

The repository includes the metadata and generated evaluation artifacts used in the paper, allowing the implementation and reported analysis to be inspected without reconstructing the input corpus from scratch.

## 4. Applying the Framework to Another Domain

The framework separates its **generic integration structure** from its **domain-specific instantiation**. The RM-ODP viewpoint structure, integration-pattern representation, constraint-template mechanism, constraint categories, evaluation procedure, and Match/Mismatch/Gap outcomes can be retained when applying the approach to another domain.

A new domain requires the following elements to be instantiated:

1. **Define domain-relevant metadata.** Identify the model and integration metadata required for compatibility assessment and organize them across the five RM-ODP viewpoints.

2. **Define domain-specific constraint templates.** Specify the compatibility conditions relevant to the domain and associate each template with its viewpoint, constraint category, required metadata, and predefined evaluation mode (`Deterministic` or `LLMAssisted`).

3. **Implement deterministic checks.** For constraints that can be evaluated through explicit conditions, implement the corresponding deterministic compatibility functions.

4. **Configure LLM-assisted constraints.** For constraints requiring contextual or semantic interpretation, specify the metadata evidence provided to the LLM-assisted evaluator while retaining the structured Match/Mismatch/Gap output.

5. **Validate the domain instantiation.** The resulting metadata fields and constraints should be reviewed and empirically evaluated for the target domain before conclusions are drawn about their suitability.

The environmental implementation in this repository provides one concrete example of this instantiation process. Evidence from the current study establishes feasibility in environmental modeling; further domain-specific studies are required to assess how well the generic framework structure transfers to other DT domains.
