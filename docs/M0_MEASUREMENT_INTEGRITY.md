# M0 Measurement Integrity

## Null measurements

Null measurements are repeated observations of the same frozen teacher.
They estimate natural variability by explicit sources:

- repeated inference runs,
- batch-size changes,
- hardware/runtime contexts,
- bootstrap resampling from the measured distribution.

No source of noise is merged into a single opaque number; each metric records a source tag.

## Noise source model

For each metric, deltas are computed from repeated-null comparisons against a baseline run.
Observed null deltas are bootstrap-resampled with a configurable confidence level and sample count.
Envelope files retain:

- bootstrap seed,
- sample count,
- confidence level,
- observed null distribution,
- lower/upper bounds.

## Controls

Three controls are required:

1. `exact-repeat` positive control: same model, repeated capture.
2. `material-negative` control: deterministic strong perturbation.
3. `boundary` synthetic control: explicit deterministic setup designed to remain inconclusive.

## Decision logic

The engine compares observed deltas to the metric-specific envelope.
`FAIL` requires:

- one or more frozen-set regressions (`source correct / candidate wrong`), or
- metric delta outside envelope.

`INCONCLUSIVE` is required when required evidence is missing, sample size is insufficient,
or explicit boundary control indicates overlap.

`PASS` is declared only when no fail condition is triggered and evidence is sufficient.

## Known limitations

- M0 is not a production-quality monitoring stack.
- Decision language is scoped to the configured fixture and declared experiment.
- CUDA-only metrics are optional and are skipped when CUDA is unavailable.

