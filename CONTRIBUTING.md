# Contributing to Neural Continuity

Neural Continuity welcomes contributions that improve the measurement framework, diagnostics, reproducibility, and independently versioned experiments without weakening the evidentiary meaning of `PASS`, `FAIL`, or `INCONCLUSIVE`.

This repository treats experiment contracts and evidence as part of the scientific result. A technically successful run is not enough to establish continuity, and a failed continuity decision is not a defect to be tuned away after observation.

Before working on M1 diagnostics, read [`docs/M1_TRANSITION_B_V2_DIAGNOSTIC_PROTOCOL.md`](docs/M1_TRANSITION_B_V2_DIAGNOSTIC_PROTOCOL.md). For claim boundaries, also read [`docs/CLAIM_POLICY.md`](docs/CLAIM_POLICY.md).

## Useful contribution areas

Contributions are especially welcome in these areas:

- read-only implementation of the frozen M1-B diagnostic protocol;
- deterministic graph inventory, structural lineage, probe planning, and quantization-parameter inspection;
- instrumentation fidelity checks and diagnostic evidence packaging;
- model-free replay and deterministic metric reconstruction;
- tests, CI, reproducibility, documentation, and developer tooling;
- independently versioned future model-transition experiments;
- new measurement or uncertainty methods with explicit contracts and claim boundaries.

Small documentation, test, and tooling improvements can be proposed directly. Large changes to decision semantics, evidence governance, dataset roles, measurement envelopes, or experiment contracts should be discussed before implementation.

## Frozen evidence is not a development fixture

M0 and M1 v1 contain authoritative, versioned evidence. Historical contracts, candidates, observations, decisions, manifests, and frozen reports must remain auditable.

Do not:

- change thresholds, tolerances, calibration, or acceptance criteria after observing a result in order to obtain a different decision;
- regenerate or retune the closed M1-B v1 candidate and present it as the same experiment;
- modify frozen evidence packages or historical reports to match current code;
- reuse the consumed M1 v1 final holdout as an unseen holdout for a later claim;
- interpret a diagnostic hypothesis label as proof of causality;
- silently substitute models, tokenizers, preprocessing, runtime providers, datasets, or dependencies during qualifying execution;
- broaden a result beyond its declared contract, dataset, runtime, and measurement envelope.

If a historical artifact is defective or incomplete, preserve it and record the correction or adjudication separately.

## Current research frontier

The current M1-B research frontier is a read-only diagnosis of the closed static-QDQ ONNX INT8 candidate.

The frozen diagnostic question is:

> At which predeclared graph boundary does the frozen ONNX INT8 candidate first develop the material numerical divergence observed against the verified ONNX FP32 source?

The diagnostic protocol does **not** authorize a new candidate, new tolerances, new calibration, or a new Transition B decision. It may localize and characterize divergence under the frozen protocol.

### Static work

Static diagnostic work may include:

- complete authority verification before graph loading;
- deterministic ONNX graph inventory;
- structural FP32-to-INT8 lineage mapping;
- read-only quantization-parameter audit;
- deterministic probe planning and hashing.

These operations must not depend on observed activation values.

### Execution work

Qualifying diagnostic execution must preserve the protocol's runtime, dataset-role, ordering, masking, arithmetic, instrumentation-fidelity, and network restrictions. Missing or mismatched authority is `BLOCKED`, not something to repair implicitly.

No probe may be added, removed, or reordered after activation values are read.

## New experiments

A genuinely new candidate or transformation requires a separately versioned experiment boundary.

A new experiment should normally define, before target execution:

- a new contract ID;
- source and candidate identities;
- dataset and role identities;
- preprocessing and runtime semantics;
- measurement-null and uncertainty policy;
- materiality and functional limits;
- validation/frozen/holdout governance;
- evidence package and replay expectations;
- the exact claim the experiment is allowed to make.

Do not inherit a claim merely because two experiments share a model family or transformation type.

## Development workflow

1. Create a focused branch.
2. Keep one scientific or architectural responsibility per pull request where practical.
3. Add tests that exercise both the expected path and fail-closed behavior.
4. Run the relevant repository checks.
5. State clearly whether the change affects code, evidence, contracts, or only documentation.
6. Explain any new evidence fields, hashes, identities, or failure states.
7. Keep generated diagnostic artifacts separate from frozen M1 v1 evidence unless the protocol explicitly defines their relationship.

Repository checks currently include:

```bash
pytest -q
ruff check .
black --check .
mypy src
python -m compileall -q src tests
```

Some real-model or ONNX workflows require optional dependencies and locally preserved frozen artifacts. Absence of those inputs should fail or skip according to the relevant test/experiment contract; do not download or substitute qualifying artifacts implicitly.

## Pull request notes

Please include:

- the scientific or technical question addressed;
- the protocol/contract section that authorizes the work;
- files and evidence responsibilities affected;
- checks executed;
- whether any model execution occurred;
- whether any activation values were observed;
- whether any frozen artifact was read, and which identity checks preceded access;
- what remains intentionally unsupported or `BLOCKED`.

A contribution that exposes a reproducible failure or ambiguity can be valuable even when it does not produce a `PASS`. The project prioritizes defensible evidence over optimistic outcomes.
