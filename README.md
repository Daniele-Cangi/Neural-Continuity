# neural-continuity

`neural-continuity` is a small, explicit measurement repository for controlled
model-continuity milestones.

## Thesis

Neural continuity asks whether a model transition preserves declared operational properties
inside an explicit measurement envelope and measured noise model.  
It is **not** a claim of universal neural equivalence.

Outcomes are constrained to:

- `PASS`
- `FAIL`
- `INCONCLUSIVE`

## Current status

- M0 measurement integrity is complete and replay-verified.
- M1 Transition A (`PyTorch FP32 → ONNX FP32`) passed under its frozen contract.
- M1 Transition B evaluated one frozen static-QDQ ONNX INT8 candidate. The technical
  pipeline and model-free replay passed, but the scientific result is `FAIL` because
  functional limits were exceeded. This is evidence for that candidate and protocol,
  not a universal claim about INT8 quantization.

The failed candidate is not to be regenerated, retuned, or re-evaluated by changing
thresholds within the same evidence slice.  A further candidate requires a separately
versioned contract and experiment.

## M0-specific scope

M0 only implements `M0_MEASUREMENT_INTEGRITY`.
It proves the measurement system can distinguish:

1. normal variation from repeated observation of the same frozen model,
2. deliberate material degradation,
3. an explicit boundary condition that remains inconclusive.

## Why this is continuity

The framework compares declared experiments and evidence against noise envelopes.
It does not claim global equivalence, guaranteed safety, or certification.
It reports where evidence is insufficient and never conflates operational agreement with
universal invariance.

## Meaning of decisions

- `PASS`: observed material metrics stay within the measured noise envelope and no frozen-set functional regression occurred.
- `FAIL`: at least one non-negotiable frozen-set regression occurred, or material metric delta exceeds envelope.
- `INCONCLUSIVE`: evidence is ambiguous, boundary-overlapping, insufficient, or missing.

## Quick start

### Smoke experiment

```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1  # Windows PowerShell
pip install -e .
pytest -q
ruff check .
black --check .
mypy src
python -m neural_continuity.cli m0-run --config experiments/m0-smoke.yaml
```

If dependencies and model access are available, run:

```bash
python -m neural_continuity.cli m0-run --config experiments/m0-real-teacher.yaml
```

Real teacher execution may be skipped automatically when the model is not available offline.

## Reproducibility

- Deterministic local toy model and fixture for offline tests.
- Canonical JSON serialization with sorted keys and UTF-8.
- Artifact SHA-256 manifests.
- Stable fixture SHA-256 identity.

## Repository outputs

Each run creates `runs/<run-id>/` containing:

- `model-manifest.json`
- `dataset-manifest.json`
- `environment-manifest.json`
- `experiment-config.json`
- `raw-observations.parquet`
- `metrics.json`
- `noise-envelope.json`
- `comparison-report.json`
- `decision.json`
- `artifact-manifest.json`

## Non-claims (M0)

The project does not claim:

- universal equivalence,
- certification,
- guaranteed safety,
- production-readiness,
- complete OOD behavior coverage.
