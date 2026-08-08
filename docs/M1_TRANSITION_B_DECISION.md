# M1 Transition B decision evidence

Transition B was evaluated from the replay-verified ONNX FP32 source and ONNX
INT8 target observation packages under the unchanged `m1-transition-b-v1`
contract and inherited `m1-transition-a-v1` operational tolerances.

- Package: `C:\tmp\nc-m1-transition-b-decision-20260808T014916`
- Evidence manifest SHA-256:
  `eed7d7af553ae9aa77274104cc75f348de910df464d836272ab37e8760e78d4e`
- Technical pipeline status: `PASS`
- Measurement integrity: `VALID`
- Scientific Transition B status: `FAIL`
- Model-free replay: `PASS`
- Aggregate status match: `true`

All three required batches (`1`, `16`, `64`) fail the inherited functional
embedding limits. The `validation`, `frozen_critical`, and `final_holdout`
roles also fail applicable ranking and retrieval-metric limits. The remaining
roles are `OBSERVED_ONLY`, as required by the frozen policy.

This is an evidence-bounded result for the frozen candidate and protocol. It
does not claim universal quantization incompatibility and does not authorize
changing thresholds or regenerating the candidate within this evidence slice.
