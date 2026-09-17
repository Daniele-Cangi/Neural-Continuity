# M1 CUDA measurement-null qualification: draft v1

Status: `DRAFT_NOT_EXECUTABLE`. This document is a proposed, separate authority,
not an amendment to the frozen CPU measurement-null extension. No CUDA null
sentinel or qualifying full-corpus epoch has been authorized or captured under it.

## Purpose and claim boundary

Estimate source-only numerical detection limits for the verified Transition A
ONNX FP32 artifact under one declared hybrid CUDA runtime. The existing CPU
extension remains unchanged. Its 120/120 sentinel and any CPU envelope cannot
qualify a CUDA run or be pooled with CUDA observations. The bounded CUDA
preflight is technical evidence only; its source and INT8 measurements cannot
select this design, set a tolerance, or count as a null epoch.

The frozen Transition B v1 scientific `FAIL` remains in force. This protocol
does not run or compare the INT8 candidate, change quantization, select an
operational tolerance, release Stage 1, or emit a continuity decision.

## Frozen design proposed before new observations

| Item | Proposed rule |
|---|---|
| Source | Verified Transition A ONNX FP32 only; artifact SHA-256 `5c0d999bd6b5e64e36cad1f61a83ef8e7507d55be49086745780fabb7c648511`. |
| Data | Exact frozen SciFact materialization and partition policy; all 5183 document IDs in canonical order, all 81 `measurement_null` query IDs in canonical order, their roles and qrels. No holdout access. |
| Preprocessing | Exact verified teacher revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, tokenizer and preprocessing identities, normalized `float32` embeddings, 384 dimensions. No re-tokenization policy change. |
| Provider | Sessions request exactly `CUDAExecutionProvider`, then `CPUExecutionProvider`; no TensorRT session provider. Require actual CUDA activity and classify every CPU fallback by operator type. No node-name, tensor-name, index, or benchmark-specific exceptions. |
| Epoch layout | `batch_1_primary: 1`, `batch_16_primary: 16`, `batch_16_repeat: 16`, `batch_64_primary: 64`, in this order for every epoch. |
| Phase 1 | 120 independent-process CUDA sentinel epochs on the first 256 document IDs under the existing domain-separated SHA-256 ordering, plus all 81 measurement-null queries. Technical and non-qualifying only; the completed CPU sentinel is not a substitute. |
| Phase 2 | 120 independent-process qualifying full-corpus epochs, four complete source passes per epoch, only after Phase 1 integrity and replay pass and an independent pre-execution review. No early stopping or adaptive sample size. |

The fixed comparison families are `repeated_inference` (the two batch-16 runs
within each epoch), `batch_size_variation` (1 versus 16, 1 versus 64, and 16
versus 64 within each epoch), and `process_restart_variation` (the batch-16
primary run across disjoint adjacent epoch pairs: 1-2, 3-4, ..., 119-120).
Epoch maxima/minima are the units for the first two families; the 60 disjoint
pair extrema are the units for restart variation. Measure document/query maximum
absolute delta and minimum cosine, ranking changes, and absolute Recall@10,
MRR@10, and NDCG@10 deltas on measurement-null queries only. Preserve fixed
metric formulas and deterministic ranking tie-breaking from the frozen source
evidence. Do not pool query roles or use candidate/holdout observations.

Only a complete, replayed 120-epoch package may support a nonparametric
maximum-order-statistic statement about the 95th percentile at at least 95%
confidence for these declared units. Process isolation does not prove statistical
independence; the assumption and its limitations must be reported. No 99th
percentile, prediction-interval, universal-equivalence, or operational-tolerance
claim follows from this design.

## Authority and execution gates

Before loading any ONNX graph, model, or activation, a fail-closed static gate
must verify external SHA-256 roots and replay for the materialized dataset,
partition policy, contract, Transition A source, tokenizer/preprocessing record,
CPU extension plan (for historical separation), and the bounded CUDA preflight.
It must bind the exact document/query identities, roles, qrels, source artifact,
normalization, dimensionality, and this protocol's immutable version and config
hash. Missing, duplicate, path-escaping, malformed, or mismatched declarations
return `BLOCKED`; they are never regenerated or reconciled silently.

The technical preflight root is its externally recorded artifact-manifest SHA-256
`276ba5286ffb38ddab1aa2e9101142fdde5e8eb20db6d0a534d33c272274e073`;
its config SHA-256 is
`c050e99002558314b90a705360b9cc67ec45e1e6862e9703b3d0fea6cefd03d5`.
Neither root authorizes full-corpus execution. Do not read preflight benchmark or
candidate outputs to change the proposed layout or statistical rules.

Runtime identity is pinned to the preflight's NVIDIA GeForce RTX 2060,
GPU UUID `GPU-ae391850-862c-ae92-0bd8-5b68015c4cf5`, compute capability
`7.5`, driver `581.57`, and imported/distribution `onnxruntime-gpu` `1.28.0`.
Pin the exact Python and loaded CUDA/cuDNN versions before changing this document
to executable status. The observed package versions to bind are NumPy `2.3.3`,
Sentence Transformers `5.6.1`, tokenizers `0.22.0`, PyTorch `2.10.0+cpu`,
and transformers `4.56.2`. A runtime identity change requires a new authority
version, not an in-place exception.

Each epoch starts a fresh process, verifies all authority roots before creating
an ONNX Runtime session, checks the session's actual ordered providers, and
records runtime identity and provider/operator counts for each of the four runs.
CUDA must execute at least one operator in every run. Any undeclared provider,
unclassified CPU event, malformed profile, or missing run blocks that epoch.
Timings and GPU utilization are descriptive and cannot gate or tune detection
limits. A technical failure is `EXECUTION_ERROR`, never a scientific `FAIL`;
all attempts remain recorded, and only the same planned epoch may be resumed
after checkpoint integrity verification. Numerically inconvenient complete
epochs cannot be replaced.

## Evidence and replay

Write new packages outside the repository using atomic epoch checkpoints and
an externally retained SHA-256 for each checkpoint and the final manifest.
Retain canonical `float32` source embeddings for each declared run in
non-pickle numeric arrays, ordered IDs, runtime/provider records, rankings and
metrics for full-corpus epochs, comparison extrema, coverage/assumption
statements, and a replay bundle. Do not retain dataset text or candidate
embeddings. Account for several gigabytes of raw full-corpus arrays before
authorizing execution; replay must be memory-bounded.

Model-free replay checks exact artifact sets, external manifest roots,
checkpoint chains, every epoch/run and identity, shapes, dtypes, finiteness,
normalization and provider semantics. It recomputes rankings, metrics, family
extrema, and the finite-sample statement from raw observations without loading
a model. Any missing or altered declared observation is `BLOCKED`. Complete
capture is `CAPTURED_NOT_DECIDED`, not Transition B `PASS` or `FAIL`.
Artifacts are tamper-evident, not physically immutable. Detection limits remain
separate from operational tolerances and require independent review and a
separately frozen detection-limit authority before any Stage 1 release.

## Freeze gate

The static authority gate and its model-free replay verify frozen inputs but
never authorize ONNX graph loading. A separate, externally reviewed,
versioned source-only technical-preflight specification may grant a narrowly
scoped permission for that preflight only, after fresh live static verification
and exact runtime/DLL identity checks. This permission is distinct from the
freeze gate below: it cannot authorize a CUDA null sentinel, qualifying
full-corpus epochs, INT8, holdout access, or tolerance changes. The source-only
preflight specification is `experiments/m1-cuda-null-preflight-v1.yaml`; it is
currently specification-only and grants no permission by itself.

The machine-readable draft is
`experiments/m1-cuda-null-v1.yaml`. It pins the qualifying dataset to the
Transition A materialization manifest `0746d98f5e69c6a0ee48ca3f47b342de1d968a877c90df26ffe8f893437fd5de`.
The technical CUDA preflight used materialization manifest
`beab716b9f322478ca3f2efd0e6e93e7d66a2b3483ed098941cd9f2275bcdcc2`.
Their 14 declared artifact records, role records, counts, policies, and source
archive match, but their upstream source-manifest hashes differ. This is an
explicit provenance split, not an implicit dataset reconciliation. CUDA
qualification must use the `0746...` root; preflight cannot attest that root.

An isolated Python 3.12.10 GPU environment now exists at
`D:\neural-continuity-runtime-cuda-v1`. ORT 1.28.0, CUDA 13 and cuDNN 9
libraries were installed without modifying the historical CPU environment.
The draft config records the loaded DLL hashes separately from installed DLLs
not observed in the preload process. `ort.preload_dlls()` can return without
raising even when individual CUDA libraries fail to load; a successful return
alone must never satisfy the runtime gate. The gate must check actual mapped
DLL paths inside the isolated environment and their pinned hashes.

This rebuilt runtime has not created an ONNX session. Its equivalence to the
runtime of the historical bounded preflight is unverified. A new source-only
technical preflight under the rebuilt runtime is required after the complete
static authority gate, before any CUDA null sentinel. The historical preflight
remains non-qualifying and cannot substitute for that check.

This draft cannot authorize execution until code verifies all pinned dataset,
partition, contract, Transition A, tokenizer, Python and runtime-DLL identities;
its config SHA-256 is recorded outside the package; adversarial fail-closed and
model-free replay tests pass; the new source-only technical preflight passes;
storage and runtime budgets are checked; and an independent review approves the
unchanged design. Any subsequent design change requires a new version. Do not
start either CUDA phase or the CPU full-corpus phase as part of this slice.
