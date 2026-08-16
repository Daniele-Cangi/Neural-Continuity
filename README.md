# Neural Continuity

**Evidence-driven framework for testing whether model transformations preserve declared operational behavior.**

Neural Continuity measures whether a model transition preserves a frozen set of operational properties inside an explicit measurement envelope and measured noise model. Each experiment produces replayable evidence and one of three decisions:

- `PASS`
- `FAIL`
- `INCONCLUSIVE`

> **A model can execute correctly and still fail continuity.**

That distinction is central to the project. Successful export, conversion, quantization, or runtime execution is not treated as proof that the transformed model still behaves within the declared contract.

## What Neural Continuity measures

The framework separates model execution from evidence about preserved behavior.

Current measurement and evidence machinery covers:

- frozen functional requirements and regression limits;
- embedding and representation drift;
- retrieval and ranking preservation;
- measured noise envelopes and repeatability;
- source / candidate identity and compatibility checks;
- deterministic evidence packaging with SHA-256 manifests;
- model-free replay of recorded decisions.

Experiments are evaluated against predeclared contracts. A candidate does not get to redefine its own baseline, tolerances, or acceptance criteria after the result is known.

## Current evidence

| Milestone | Transition / purpose | Result |
|---|---|---|
| **M0** | Measurement integrity | **PASS / replay-verified** |
| **M1-A** | `PyTorch FP32 → ONNX FP32` | **PASS** under the frozen contract |
| **M1-B** | `ONNX FP32 → static-QDQ ONNX INT8` | **FAIL** under the frozen contract |

### M0 — measurement integrity

M0 establishes the trust boundary for the measurement system. It verifies that the framework can distinguish:

1. normal variation from repeated observation of the same frozen model;
2. deliberate material degradation;
3. a boundary condition where the evidence remains `INCONCLUSIVE`.

Replay reconstructs the recorded aggregate decision and individual control outcomes without rerunning the original model execution.

### M1-A — FP32 export continuity

Transition A evaluates `PyTorch FP32 → ONNX FP32` separately from quantization so export/runtime effects are not conflated with quantization effects.

The transition passed under its frozen contract.

See [M1 Transition A evidence](docs/M1_TRANSITION_A_EVIDENCE.md).

### M1-B — INT8 candidate failure

Transition B evaluated one frozen static-QDQ ONNX INT8 candidate against the replay-verified ONNX FP32 source.

The technical pipeline completed successfully, measurement integrity remained valid, and model-free replay reproduced the recorded decision. The scientific result was nevertheless **`FAIL`** because the candidate exceeded the inherited functional limits across all required batch sizes (`1`, `16`, `64`) and materially degraded decision-bearing retrieval metrics.

This is exactly the separation Neural Continuity is intended to enforce:

```text
technical execution     PASS
measurement integrity   VALID
model-free replay       PASS
continuity decision     FAIL
```

The failed candidate remains frozen. It is not regenerated, retuned, or re-evaluated by changing thresholds inside the same evidence slice. A new attempt requires a separately versioned contract, candidate identity, and experiment.

See:

- [M1 Transition B decision](docs/M1_TRANSITION_B_DECISION.md)
- [M1 Transition B postmortem](docs/M1_TRANSITION_B_POSTMORTEM.md)
- [M1 v1 evidence archive](docs/M1_EVIDENCE_ARCHIVE.md)

## Current research frontier

The current frontier is a **read-only diagnosis of the closed M1-B v1 INT8 candidate**. The governing document is the [M1 Transition B v2 diagnostic protocol](docs/M1_TRANSITION_B_V2_DIAGNOSTIC_PROTOCOL.md).

The diagnostic asks where the frozen INT8 candidate first develops material numerical divergence relative to the verified ONNX FP32 source. It may localize and characterize divergence; it does not authorize a new candidate, new calibration, new tolerances, or a replacement Transition B decision.

Contributions are welcome around authority verification, deterministic graph inventory, structural lineage, quantization-parameter inspection, probe planning, instrumentation fidelity, diagnostic evidence packaging, and model-free replay. Historical M0/M1-v1 evidence remains frozen and auditable.

See [CONTRIBUTING.md](CONTRIBUTING.md) before changing experiment, diagnostic, evidence, or verification behavior.

## Why this is continuity

Model transformations can preserve executability while changing behavior that matters to the application. Neural Continuity asks a narrower and testable question:

> **Did this declared transition preserve the operational properties we froze before evaluation, inside the evidence envelope we actually measured?**

The framework therefore compares declared experiments and evidence against measured envelopes rather than treating conversion success, numerical closeness, or a single benchmark score as automatic equivalence.

## Meaning of decisions

- `PASS`: observed material metrics stay within the measured noise envelope and no frozen-set functional regression occurred.
- `FAIL`: at least one non-negotiable frozen-set regression occurred, or a material metric delta exceeds the declared envelope.
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

## Reproducibility and evidence

Neural Continuity treats the evidence package as part of the experiment rather than an afterthought.

The repository includes:

- deterministic local toy models and fixtures for offline tests;
- canonical JSON serialization with sorted keys and UTF-8;
- stable fixture identities;
- SHA-256 artifact manifests;
- frozen experiment contracts;
- compatibility checks between source and candidate observations;
- replay paths that reproduce decisions from recorded evidence without rerunning the models.

M1 v1 also records a tamper-evident inventory for its authoritative evidence packages. Tamper-evident does not mean physically immutable, and remote redundancy is not currently claimed as verified.

## Repository outputs

A measurement run can produce `runs/<run-id>/` artifacts including:

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

## Claim boundary

Neural Continuity measures continuity only inside declared experiments, frozen contracts, observed datasets, and measured uncertainty.

