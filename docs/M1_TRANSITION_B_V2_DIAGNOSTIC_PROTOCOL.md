# M1 Transition B v2 diagnostic protocol

## 1. Status and authority

This document freezes the pre-implementation protocol for a read-only diagnosis
of the closed `m1-transition-b-v1` candidate. It is not a Transition B v2
contract and does not authorize a new quantization candidate.

- Repository base: `6fef97dd0232a4d477c644c687c85a814886c902`
- Closed milestone tag: `m1-v1-evidence-closed`
- Frozen Transition B v1 result: scientific `FAIL`
- Frozen technical result: `PASS`
- Frozen measurement integrity: `VALID`
- Frozen model-free replay: `PASS`

The candidate, calibration package, contracts, operational tolerances, dataset
materialization, observations, and decision remain unchanged.

## 2. Diagnostic question

The only research question authorized by this protocol is:

> At which predeclared graph boundary does the frozen ONNX INT8 candidate first
> develop the material numerical divergence observed against the verified ONNX
> FP32 source?

The diagnostic may localize and characterize divergence. It may not optimize a
candidate, select tolerances, or claim that a suspected operator is causal
without additional evidence.

## 3. Frozen input authority

| Input | Required identity |
|---|---|
| ONNX FP32 source | `5c0d999bd6b5e64e36cad1f61a83ef8e7507d55be49086745780fabb7c648511` |
| ONNX INT8 candidate | `8b28688438e249c42b523e276333a3a009ca30d0754a3ba6fcbb10d76de873e5` |
| Candidate manifest | `d11888e48e24a9e29f5bdfac48ad7ace4204fb7b101e3531faa0f11190ad562c` |
| Calibration manifest | `3ac7d68e01976ee444217cd80c5b4b7338f870d8c0ab5a350a960495baef0778` |
| Paired FP32 evidence | `cf03882df0913e84b456b61f02a1c00a14ec151cd0fd9cc07f7d0bf04745b4df` |
| INT8 target evidence | `4027c1edf9f24254e6174ca79bc722c98758c8f97f5ad175b380866f64063a80` |
| Transition B decision | `eed7d7af553ae9aa77274104cc75f348de910df464d836272ab37e8760e78d4e` |
| Transition A contract | `772e0df5133de09f6108cb42144e9b2ee69e47c0694bdf5b60ca4d88c18ee5c4` |
| Transition B v1 contract | `ad8c04574b3121eb69028e89f98f81cd1a68c34f15ecc23f9dc85c66b45273b0` |

Every authority must be verified before model loading. Missing or mismatched
identity returns `BLOCKED`; evidence must not be silently repaired, regenerated,
or reconciled.

## 4. Data-role policy

### Authorized

- `contract_development`: the only role authorized for new model execution.
- `quantization_calibration`: existing recorded inputs and range metadata may be
  inspected, but not changed, resampled, or rerun to select a new configuration.

All `contract_development` queries must be used in canonical query-ID order. The
diagnostic execution batch size is `16`, with only the final incomplete batch
allowed to be smaller.

### Prohibited

- `measurement_null`: no new diagnostic execution;
- `validation`: no new diagnostic execution or configuration selection;
- `frozen_critical`: no new diagnostic execution or configuration selection;
- `final_holdout`: no access during diagnostics.

The v1 `final_holdout` is consumed. It cannot become an unseen v2 holdout.

## 5. Frozen runtime semantics

- Provider: `CPUExecutionProvider` only.
- Tokenizer and preprocessing: exact frozen v1 identities and semantics.
- Normalization: exact frozen v1 semantics.
- Input IDs, attention masks, ordering, padding, truncation, and output dtype:
  unchanged from the paired v1 capture.
- Network access: prohibited during qualifying diagnostic execution.
- Model downloads or dependency substitution: prohibited.

An unavailable dependency is `BLOCKED`, not evidence of incompatibility.
Unmeasured CUDA behavior remains `unverified`.

## 6. Predeclared diagnostic components

Implementation responsibilities must remain separate. No component may decide
or execute another component's responsibility.

### 6.1 Authority verifier

Verifies every input hash, manifest, dataset identity, role assignment, runtime
provider, tokenizer identity, and required artifact before loading either model.

### 6.2 Graph inventory

Produces a deterministic topological inventory for both frozen graphs:

- operator type and topological index;
- tensor names, shapes, and dtypes when declared;
- QDQ edges and quantized operator coverage;
- scale and zero-point initializer references;
- graph inputs and outputs;
- an explicit FP32-to-INT8 lineage map.

Lineage matching must use graph structure and tensor provenance, never observed
activation values. An ambiguous required match returns `BLOCKED`.

### 6.3 Quantization-parameter audit

Reports, without changing them:

- scale and zero-point dtype and shape;
- scale minimum, maximum, and finite-value status;
- zero-point minimum and maximum;
- per-tensor or per-channel structure;
- quantized-axis identity when declared;
- invalid, missing, non-finite, or non-positive scales.

This audit is descriptive. It cannot label a configuration acceptable or select
replacement values.

### 6.4 Probe planner

The complete probe plan must be written and hashed before activation capture.
It includes every unambiguously matched:

- boundary immediately before and after a quantized compute path;
- normalization boundary;
- attention or matrix-compute boundary;
- pooling or reduction boundary on the final embedding path;
- final embedding output.

Each boundary receives one or more structural family labels from
`QUANTIZED_COMPUTE`, `NORMALIZATION`, `ATTENTION_OR_MATMUL`, `OUTPUT_PATH`, and
`FINAL_OUTPUT`. Labels are derived only from operator type, graph topology, and
tensor lineage under these frozen rules:

- `QUANTIZED_COMPUTE`: the boundary crosses `QLinear*`, `MatMulInteger`,
  `ConvInteger`, or a `DequantizeLinear -> compute -> QuantizeLinear` path;
- `NORMALIZATION`: the boundary crosses `BatchNormalization`,
  `InstanceNormalization`, `LayerNormalization`, `LpNormalization`, or a
  structural normalization subgraph that computes a norm or variance and then
  divides or scales the same incoming lineage;
- `ATTENTION_OR_MATMUL`: the boundary crosses `Attention`,
  `MultiHeadAttention`, `MatMul`, `MatMulInteger`, `QLinearMatMul`, or `Gemm`;
- `OUTPUT_PATH`: the boundary is at or downstream of the first `ReduceMean`,
  `ReduceSum`, `ReduceMax`, `GlobalAveragePool`, or `GlobalMaxPool` that is an
  ancestor of the final embedding output;
- `FINAL_OUTPUT`: the boundary is the declared final embedding output.

A structural normalization subgraph must be recognized from operator and edge
patterns fixed in code before any activation capture; its exact matched nodes
and rule identifier are recorded in the inventory. A final output downstream
of an output-path reduction receives both `OUTPUT_PATH` and `FINAL_OUTPUT`.
If a boundary required to evaluate a predeclared
hypothesis cannot be matched unambiguously across the two graphs, the diagnostic
returns `BLOCKED`; it cannot omit the boundary and still become `COMPLETE`.

No probe may be added, removed, or reordered after any activation value is read.

### 6.5 Instrumented execution

Instrumentation operates on derived diagnostic graph copies. The frozen source
and candidate files remain byte-identical and are never overwritten.

For each planned probe, capture paired FP32 and INT8 tensors for all authorized
queries. Record:

- maximum absolute delta;
- mean absolute delta;
- relative L2 error;
- minimum and mean cosine similarity when dimensions permit;
- finite-value counts;
- INT8 saturation counts when representable from recorded quantization parameters.

These measurements are diagnostic observations, not operational decisions.

Paired numerical comparison uses the floating-point tensor after the applicable
`DequantizeLinear` at an INT8 boundary and its lineage-matched FP32 tensor. Raw
integer tensors are retained separately only for the saturation audit. A probe
without a unique same-semantics floating-point pair is `BLOCKED`.

Metric semantics are frozen as follows:

- corresponding tensors must have identical logical shape after structural
  lineage alignment; otherwise the diagnostic is `BLOCKED`;
- the batch axis is axis `0` and each query is an independent observation;
- for tensors that preserve a sequence axis, positions whose frozen
  `attention_mask` is `0` are excluded from every metric; inability to establish
  the sequence axis structurally is `BLOCKED`;
- maximum absolute delta is the maximum over all valid elements per query;
- mean absolute delta is the arithmetic mean over all valid elements per query;
- relative L2 error is `||target - source||_2 / ||source||_2` over all valid
  elements per query; it is `0` when both norms are zero and is recorded as
  `UNBOUNDED_ZERO_REFERENCE` with JSON value `null` when only the source norm is
  zero;
- cosine similarity is computed along the last feature axis for every valid
  token or row, then reported as per-query minimum and arithmetic mean; a pair
  of zero vectors has cosine `1`, exactly one zero vector has cosine `0`, and an
  absent or ambiguous feature axis is `BLOCKED`;
- every per-query value is retained in canonical query-ID order; package-level
  maxima, minima, and arithmetic means are derived from those values and never
  replace them;
- any unexpected NaN or infinity is `BLOCKED`; JSON artifacts contain no
  non-standard numeric literals.

For deterministic localization, each probe also records the bounded per-query
score
`symmetric_l2 = ||target - source||_2 / max(||source||_2, ||target||_2)`.
It is `0` when both norms are zero. The probe score is the arithmetic mean of
its per-query scores. This is a diagnostic ordering statistic, not a detection
limit or operational tolerance.

For a quantized activation tensor, saturation means an observed integer equals
the minimum or maximum value representable by its recorded dtype. Counts and
fractions are reported per query and in aggregate. If the integer tensor or its
quantization parameters are not observable, saturation is explicitly
`NOT_MEASURABLE`; it is never silently omitted.

### 6.6 Instrumentation fidelity control

Before probe evidence is accepted, each derived graph must reproduce the final
output of its corresponding frozen graph on the same inputs.

The required control is exact equality of shape, dtype, and values. Any mismatch
returns `BLOCKED` and invalidates all activation diagnostics from that derived
graph. The protocol must then be redesigned; tolerances must not be introduced
after observing the mismatch.

### 6.7 Diagnostic replay

Replay must recompute every reported metric from recorded probe tensors without
loading or executing either model. Missing probes, changed ordering, unmatched
identities, or artifact-hash failures return `BLOCKED`.

## 7. Predeclared hypotheses

The diagnostic may evaluate only these hypotheses:

- `H1_NORMALIZATION_SENSITIVITY`: divergence emerges at or immediately before a
  normalization boundary.
- `H2_ATTENTION_OR_MATMUL_SENSITIVITY`: divergence emerges across an attention
  or matrix-compute boundary.
- `H3_CALIBRATION_RANGE_MISMATCH`: recorded ranges exhibit saturation or scale
  structure consistent with the observed activation divergence.
- `H4_OUTPUT_PATH_QUANTIZATION`: divergence remains limited until pooling,
  normalization, or another final embedding-path boundary.

Allowed outcomes per hypothesis are `SUPPORTED`, `NOT_SUPPORTED`, and
`UNRESOLVED`. These are diagnostic labels, not scientific transition decisions.
Multiple hypotheses may remain supported or unresolved.

Hypothesis labels use this deterministic procedure:

1. Order matched probes by their frozen lineage order from graph input to final
   output. Set the predecessor score of the first probe to `0`.
2. For each probe, compute `growth = max(0, score - predecessor_score)`.
3. The dominant-onset set contains every probe whose growth equals the maximum
   observed growth. If the maximum is `0`, every hypothesis is `UNRESOLVED`.
4. For `H1`, `H2`, and `H4`, the relevant families are respectively
   `NORMALIZATION`, `ATTENTION_OR_MATMUL`, and `OUTPUT_PATH`. A hypothesis is
   `SUPPORTED` when every member of the dominant-onset set has its relevant
   family label, `NOT_SUPPORTED` when none does, and `UNRESOLVED` when the set
   mixes relevant and non-relevant boundaries.
5. `H3` is `NOT_SUPPORTED` when saturation is measurable at every planned
   quantized-activation boundary and all saturation counts are zero. It is
   `SUPPORTED` when saturation is measurable at every such boundary, at least
   one dominant-onset boundary has non-zero saturation, and the maximum observed
   saturation fraction occurs at a dominant-onset boundary. It is otherwise
   `UNRESOLVED`.

Equality comparisons use the replayed IEEE-754 values produced by the frozen
metric formulas; no epsilon or post-observation threshold may be introduced.
`SUPPORTED` means structurally consistent with the hypothesis under this
procedure, not proven causal.

## 8. Diagnostic package

The qualifying package must contain:

- `diagnostic-authority.json`;
- `fp32-graph-inventory.json`;
- `int8-graph-inventory.json`;
- `lineage-map.json`;
- `quantization-parameter-report.json`;
- `probe-plan.json`;
- `instrumentation-fidelity.json`;
- `probe-observations.npz`, containing numeric or fixed-width Unicode arrays
  loadable with NumPy `allow_pickle=False`;
- `activation-divergence-report.json`;
- `hypothesis-report.json`;
- `replay-bundle.json`;
- `evidence-manifest.json` with SHA-256 for every declared artifact.

The package states that evidence is tamper-evident, not physically immutable.

## 9. Diagnostic states

- `COMPLETE`: all authorities, controls, observations, reports, and replay pass.
- `BLOCKED`: required identity, lineage, artifact, observation, fidelity control,
  or replay evidence is missing or mismatched.
- `EXECUTION_ERROR`: valid authorized execution cannot complete technically.

This protocol does not emit `PASS`, `FAIL`, or `INCONCLUSIVE` for Transition B.
The frozen v1 scientific `FAIL` remains authoritative.

## 10. Non-scope

- no `m1-transition-b-v2` contract;
- no new INT8 candidate;
- no quantization configuration change;
- no calibration rerun or resampling;
- no tolerance change;
- no dataset repartition or new holdout selection;
- no tokenizer, preprocessing, normalization, or provider change;
- no validation, frozen-critical, or final-holdout execution;
- no performance optimization claim;
- no Spectra, distillation, pruning, routing, or later milestone.

## 11. Pre-implementation quality gate

- [ ] Every frozen authority is available and hash-verified.
- [ ] Only `contract_development` is scheduled for new model execution.
- [ ] The complete graph-derived probe plan is fixed before activation capture.
- [ ] Instrumented graphs are derived copies, never replacements.
- [ ] Exact final-output fidelity controls are implemented fail-closed.
- [ ] Raw probe observations support model-free metric replay.
- [ ] No diagnostic value can change v1 evidence or select a v2 tolerance.
- [ ] No prohibited data role is loaded by the diagnostic executor.
- [ ] Every declared missing artifact or observation returns `BLOCKED`.
- [ ] The implementation slice stops before any new candidate or v2 contract.

## 12. Stop condition

The diagnostic milestone ends when one package reaches `COMPLETE` and its
model-free replay reproduces the graph inventory, fidelity controls, activation
metrics, and hypothesis labels.

The result may inform a later governance decision. It does not authorize that
decision, create a v2 contract, or generate a candidate.
