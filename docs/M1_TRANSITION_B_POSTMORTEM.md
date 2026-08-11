# M1 Transition B v1 postmortem

## Authority and scope

This is a read-only analysis of the frozen `m1-transition-b-v1` result. It does
not change the candidate, calibration, contracts, operational tolerances,
dataset identities, preprocessing semantics, observations, or decision.

- Candidate: frozen static-QDQ ONNX INT8
- Execution provider: `CPUExecutionProvider`
- Required batches: `1`, `16`, `64`
- Measurement integrity: `VALID`
- Technical execution: `PASS`
- Model-free replay: `PASS`
- Scientific decision: `FAIL`

The claim remains evidence-bounded to this candidate and protocol. It is not a
universal claim about INT8 quantization.

## Functional evidence

The frozen functional tolerances were maximum absolute embedding delta
`<= 0.00001` and minimum cosine similarity `>= 0.99999`.

| Batch | Document max delta | Document min cosine | Query max delta | Query min cosine |
|---:|---:|---:|---:|---:|
| 1 | 0.3875682950 | -0.0423601381 | 0.2970198095 | 0.0770402253 |
| 16 | 0.3875681460 | -0.0423601717 | 0.2970198095 | 0.0770402402 |
| 64 | 0.3875681460 | -0.0423601717 | 0.2970198095 | 0.0770402402 |

Every batch independently exceeded both functional limits. The document and
query maximum deltas were approximately 38,757 and 29,702 times their allowed
limits, respectively.

## Retrieval evidence

Every query ranking changed in every data role at every batch. The ranking
change fraction was `1.0` for all six roles. The decision-bearing metric
decreases below were identical across batches.

| Role | Recall@10 decrease | MRR@10 decrease | NDCG@10 decrease | Applicable limit |
|---|---:|---:|---:|---:|
| `validation` | 0.3638613861 | 0.3283533711 | 0.3364687558 | 0.005 |
| `frozen_critical` | 0.3725000000 | 0.3155952381 | 0.3297972407 | 0.0 |
| `final_holdout` | 0.3606944444 | 0.3229811508 | 0.3342565183 | 0.005 |

The `measurement_null`, `quantization_calibration`, and
`contract_development` roles remained `OBSERVED_ONLY`; their measurements did
not become promotion decisions.

## Supported conclusions

- The outcome is not an `EXECUTION_ERROR`: source and target evidence completed.
- The outcome is not `BLOCKED`: integrity and compatibility gates passed.
- The regression is not isolated to one execution batch.
- The frozen candidate does not satisfy the inherited continuity contract.
- Replay reproduces the decision without rerunning either model.

The near-identical evidence at batches 1, 16, and 64 supports an inference that
the observed regression is candidate-level rather than batch-size instability.
It does not identify which quantized operator or calibration range caused it.

## Root-cause boundary

Root cause is not established by M1 v1. Any diagnostic work must treat the
following as hypotheses, not conclusions:

- sensitivity of normalization, attention, or matrix operations to full static QDQ;
- activation-range mismatch under the frozen MinMax calibration package;
- unsuitable quantization of one or more output-path operations;
- interaction between QUInt8 activations and per-channel QInt8 weights.

Compatibility evidence makes a silent dataset, tokenizer, preprocessing,
normalization, provider, batch, dimensionality, or role-order mismatch an
unsupported explanation unless new tamper-evident evidence demonstrates it.

## Governance for any v2 experiment

- `m1-transition-b-v1` and its candidate remain closed and unchanged.
- Diagnostics may use only roles authorized by a new predeclared protocol.
- Candidate or holdout results may not select new tolerances.
- A v2 attempt requires a new contract ID and a new candidate identity before
  target execution.
- The v1 `final_holdout` has been observed and is consumed. It cannot be
  represented as an unseen final holdout for v2.
- A qualifying v2 claim therefore requires a newly frozen holdout identity or
  an explicit governance decision defining a different repeated-evaluation
  claim; it must not silently repartition existing evidence.
- Spectra, distillation, pruning, routing, and later milestones remain outside
  this postmortem.

The next authorized coding slice should stop at read-only diagnostics on
`contract_development`; it must not generate a new INT8 candidate.

