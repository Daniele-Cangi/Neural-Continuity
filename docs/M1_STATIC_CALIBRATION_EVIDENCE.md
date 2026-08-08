# M1 static calibration input evidence

## Scope

This evidence package freezes the only inputs authorized for M1 static
quantization calibration. It is not a quantized model, an INT8 observation,
or a Transition B decision.

## Authority and boundaries

- Contract: `m1-transition-b-v1`
- Contract SHA-256: `ad8c04574b3121eb69028e89f98f81cd1a68c34f15ecc23f9dc85c66b45273b0`
- Dataset: `nc-m1-beir-scifact-v1`
- Authorized role: `quantization_calibration`
- Cardinality: `162` queries
- Order: query ID ascending by UTF-8 bytes
- Leakage behavior: `BLOCKED`
- Model execution: `false`

The package contains only canonical query IDs and texts. It excludes
`measurement_null`, `contract_development`, `validation`, `frozen_critical`,
and `final_holdout` observations. Any missing, altered, or undeclared artifact
fails verification closed through manifest artifact-integrity checks.

## Captured package

- Directory: `C:\tmp\nc-m1-static-calibration-20260806T224805`
- Manifest SHA-256: `3ac7d68e01976ee444217cd80c5b4b7338f870d8c0ab5a350a960495baef0778`
- Inputs SHA-256: `7aaecfdc1e4734ab360c66e12a43eea0896c53d3a3745346b5cf9edd46e8f522`
- Materialization manifest SHA-256: `beab716b9f322478ca3f2efd0e6e93e7d66a2b3483ed098941cd9f2275bcdcc2`
- Partition policy SHA-256: `43eb7bd3a805792897de35cebd995d3d5b93931f08fca260f8a8d4aa1883457d`

The package was built and verified with `model_execution_used: false`.

## Re-verification

```powershell
python -m neural_continuity.m1_b.calibration_data verify `
  --package C:\tmp\nc-m1-static-calibration-20260806T224805
```

Verification is model-free and validates the manifest hash, role, count,
unique IDs, canonical order, and text structure before later quantization can
consume the package.
