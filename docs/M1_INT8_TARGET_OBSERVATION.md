# M1 ONNX INT8 target observation

## Scope

This package captures canonical ONNX INT8 target observations for the frozen
static QDQ candidate. It contains no source-target comparison and no
Transition B scientific decision.

## Captured evidence

- Directory: `C:\tmp\nc-m1-int8-observation-20260806T232512`
- Evidence manifest SHA-256:
  `4027c1edf9f24254e6174ca79bc722c98758c8f97f5ad175b380866f64063a80`
- Candidate manifest SHA-256:
  `d11888e48e24a9e29f5bdfac48ad7ace4204fb7b101e3531faa0f11190ad562c`
- ONNX INT8 SHA-256:
  `8b28688438e249c42b523e276333a3a009ca30d0754a3ba6fcbb10d76de873e5`
- Provider: `CPUExecutionProvider`
- Runs: batch sizes `1`, `16`, and `64`
- Evidence state: `CAPTURED_PENDING_SOURCE_COMPARISON`
- Transition B decision: `NOT_EVALUATED`

The package includes the frozen candidate, canonical target embeddings,
metadata, a replay bundle, and SHA-256 artifact entries. Its model-free replay
passed with three declared target runs and `model_execution_used: false`.

## Boundary

This target-only evidence cannot establish `PASS`, `FAIL`, or `INCONCLUSIVE`.
The next slice must capture compatible canonical ONNX FP32 source observations
before comparison, decision, and full Transition B replay can be implemented.
