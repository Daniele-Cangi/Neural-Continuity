# M1 static QDQ candidate

## Scope

This package captures the frozen ONNX INT8 candidate for Transition B. It does
not contain target observations, a source-target comparison, replay evidence,
or a Transition B decision.

## Frozen configuration

- Contract: `m1-transition-b-v1`
- Contract SHA-256: `ad8c04574b3121eb69028e89f98f81cd1a68c34f15ecc23f9dc85c66b45273b0`
- Source ONNX FP32 SHA-256:
  `5c0d999bd6b5e64e36cad1f61a83ef8e7507d55be49086745780fabb7c648511`
- Calibration package SHA-256:
  `3ac7d68e01976ee444217cd80c5b4b7338f870d8c0ab5a350a960495baef0778`
- Calibration inputs SHA-256:
  `7aaecfdc1e4734ab360c66e12a43eea0896c53d3a3745346b5cf9edd46e8f522`
- Calibration queries: `162`
- Runtime: `onnxruntime 1.28.0`, `CPUExecutionProvider`
- Format: static `QDQ`
- Activations: `QUInt8`
- Weights: per-channel `QInt8`
- Calibration: `MinMax`, batch size `32`

## Captured candidate

- Directory: `C:\tmp\nc-m1-static-qdq-verified-20260806T230647`
- Candidate manifest SHA-256:
  `d11888e48e24a9e29f5bdfac48ad7ace4204fb7b101e3531faa0f11190ad562c`
- ONNX INT8 SHA-256:
  `8b28688438e249c42b523e276333a3a009ca30d0754a3ba6fcbb10d76de873e5`
- Candidate status: `CAPTURED_PENDING_OBSERVATION`

The generator verified the candidate with `onnx.checker` before publication.

## Technical notes

ONNX Runtime emitted preprocessing, degenerate-range, and LayerNorm
per-channel-axis warnings during calibration. They are recorded as technical
execution context, not a scientific `FAIL`, `PASS`, or `INCONCLUSIVE` result.
The next authorized slice must run and capture canonical ONNX INT8 observations
before any continuity claim can be evaluated.
