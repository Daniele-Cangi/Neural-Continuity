# M1 Transition B protocol

Transition B is strictly:

```text
verified ONNX FP32 -> ONNX INT8
```

It is independent from teacher export. The verified ONNX FP32 artifact from Transition A
is the only source artifact; the PyTorch teacher is not re-exported during B.

## Frozen prerequisites

- Transition A evidence and replay must be `PASS`.
- The ONNX FP32 source artifact SHA-256 must equal the identity pinned in
  `contracts/m1-transition-b-v1.json`.
- A dedicated ONNX FP32 measurement null must be captured before any INT8 candidate.
- The null must include three repeated source executions and batch-size variation at `1`,
  `16`, and `64`.
- Missing source observations, calibration identities, or replay artifacts are `BLOCKED`.

## Static calibration isolation

Only the materialized `quantization_calibration` role may provide static calibration data.
Its canonical query ordering is UTF-8 bytewise ascending query ID. The calibration reader
is limited to its frozen `162` query identities.

`measurement_null`, `contract_development`, `validation`, `frozen_critical`, and
`final_holdout` are prohibited calibration inputs. Leakage is a technical `BLOCKED` state,
not a scientific `FAIL`.

## Module boundaries

The implementation is deliberately split:

- `calibration_data`: split identity and deterministic batches;
- `quantization`: ONNX Runtime static QDQ transformation only;
- `onnx_observation`: one-artifact observation capture only;
- `comparison`: pure metrics and topology computation;
- `decision`: contract application only;
- `replay`: model-free reconstruction only;
- orchestration: thin CLI dispatch only.

No module may absorb another module's responsibility. In particular, quantization cannot
select thresholds, comparison cannot load calibration data, and replay cannot rerun either
ONNX model.

## Frozen INT8 method

The first candidate, after the ONNX FP32 null is available, is static QDQ quantization with
`QUInt8` activations, `QInt8` weights, per-channel weights, and MinMax calibration. A
different method requires a new contract version before candidate execution.
