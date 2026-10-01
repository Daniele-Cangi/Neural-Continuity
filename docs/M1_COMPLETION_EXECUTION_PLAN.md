# M1 completion execution plan

Status: operational plan, not a scientific authority or an execution permit.
Prepared for the next implementation agent. Review this plan before starting a
scientific decision. It does not amend any frozen contract, protocol, artifact,
candidate, tolerance, or historical result.

## 1. Objective and current state

Finish the two open M1 tracks with the smallest coherent implementation:

1. Repair the contract-hash portability defect without changing frozen evidence.
2. Close the CUDA source-only metrology track at the claim level actually
   authorized by its frozen inputs and decision rules.
3. Complete the M1-B v2 diagnostic package and its model-free replay for the
   already failed v1 candidate.

The verified historical results are Transition A `PASS` and Transition B v1
scientific `FAIL`; the latter also has technical `PASS`, measurement integrity
`VALID`, and model-free replay `PASS`. The ONNX FP32 CPU measurement null was
captured and replayed. A `NOT_YET_CAPTURED` field in the original B contract
describes its state at freeze time; it is not the current evidence status.

The source-only CUDA sentinel has 120/120 isolated epochs and a completed
technical gate with `TECHNICAL_GATE_PASS_NO_SCIENTIFIC_RELEASE`. The qualifying
full corpus has 120/120 epochs, final model-free replay and external anchors;
its status is `CAPTURED_NOT_DECIDED`. The separate detection-limit authority is
frozen with `scientific_decision: NOT_EVALUATED` and
`stage1_release_authorized: false`. No CUDA INT8 candidate was executed.

The historical `docs/M1_CUDA_NULL_QUALIFICATION_PROTOCOL.md` still calls its
proposal a draft. Use the later, hash-bound full-corpus execution authority and
the completed evidence for current state. Do not rewrite that historical draft
to make it appear that execution had been authorized at its original freeze.

## 2. Authority map and claim limits

Use these anchors for read-only orientation; verify the actual bytes and replay
records before deriving or publishing a new claim:

| Role | Location or SHA-256 |
|---|---|
| Full-corpus execution authority | `experiments/m1-cuda-null-full-corpus-authority-v1.json`; SHA-256 `8a474cd6dbce76515f99715368da4ad873b3e576c8cd106c1b98d3be958392ea` |
| Full-corpus evidence | `D:\neural-continuity-evidence\nc-m1-cuda-full-corpus-v1-20260919\full-corpus-package` |
| External final manifest | `024e69d7cc55be0e25b8cc3da95775279fad5cbd827cd102672632e4fabfe7ed` |
| Package artifact manifest | `0e935bacfe9c5ffdce1a1eeae47fe78379202c9e2f8965169277ada715ee0fb29` |
| Corpus summary | `8f97e8e3dce75c44edb3ee9de45576c28328c52b6356e5f64e69754149e6b4ea` |
| Full-corpus replay bundle | `d30b75883f29ab4e6b04bf0d1aa7f91e54032cc3ed65489bfe9c29454690a1bd` |
| Detection-limit authority | `experiments/m1-cuda-detection-limit-authority-v1.json` |
| Frozen B v1 decision | `docs/M1_TRANSITION_B_DECISION.md`; evidence manifest SHA-256 `eed7d7af553ae9aa77274104cc75f348de910df464d836272ab37e8760e78d4e` |
| Diagnostic protocol | `docs/M1_TRANSITION_B_V2_DIAGNOSTIC_PROTOCOL.md` |

Keep three questions separate. The CUDA source null estimates measurement
resolution for the declared runtime and dataset. Operational tolerances judge a
candidate under its own frozen contract. The M1-B v2 diagnostic describes where
the already recorded INT8 divergence appears; its H1-H4 labels do not revise
the v1 decision or establish causation.

## 3. Start-of-work inventory (read only)

1. Identify the checkout, branch, pending changes, existing PRs and current
   committed versions. Preserve unrelated work. Use one coherent branch for
   each independently reviewable change; avoid micro-PRs for one fix.
2. Confirm the authorities above, their external anchors and recorded replay
   outcomes. Do not recapture an epoch, regenerate a package, or run an ONNX
   model merely to inspect status. If an anchor or required artifact is absent
   or mismatched, record `BLOCKED` and the exact missing identity.
3. Inspect the existing hash helpers, replay paths, static diagnostic packages,
   tests, and `.github/workflows/offline-validation.yml`. Reuse them where they
   already enforce the invariant. Current CI already runs pytest, Ruff, scoped
   Black, mypy and compileall on Windows in offline mode.
4. Keep a short evidence ledger of verified input path, expected SHA-256,
   observed SHA-256, declared role, and verification result. Do not duplicate
   these identities into another scientific authority without a trust reason.

Exit: the agent can point to the exact frozen input for each subsequent
operation. A missing input blocks the dependent operation only; it is never
silently replaced.

## 4. Slice A: contract-hash portability

### Observed defect

In the current Windows checkout, the Transition A contract has raw SHA-256
`0bb881c5379b155d13cfafc731299ae67e1843addb0b2357a5ce6a5354f72848`.
Its LF-normalized SHA-256 is the frozen inheritance pin
`772e0df5133de09f6108cb42144e9b2ee69e47c0694bdf5b60ca4d88c18ee5c4`.
The content difference is only CRLF versus LF. The Transition B contract has
Windows raw SHA-256 `0acd1b0218b513ebbb6f9ab480f9305c746591b49a84577d09cb2e29c881c795`
and LF SHA-256 `ad8c04574b3121eb69028e89f98f81cd1a68c34f15ecc23f9dc85c66b45273b0`.
`m1_b/decision_package.py` hashes A as raw bytes but compares it with the
LF-content pin in B; that check rejects the current checkout. Other code pins
the Windows raw B hash, so a blanket hash replacement would break an existing
authority or historical replay.

### Implementation sequence

1. Inventory every use of contract hashing in
   `src/neural_continuity/m1_b/decision_package.py`,
   `src/neural_continuity/m1_b/calibration_data.py`,
   `src/neural_continuity/m1_diagnostics/authority.py`,
   `src/neural_continuity/m1_diagnostics/cuda_null_authority.py`, and the
   associated replay verifiers. Classify each digest as a logical frozen
   contract-content pin or a raw artifact-byte integrity check.
2. Reuse the existing LF-normalized contract verifier for content pins, or
   centralize exactly that algorithm in one small helper if duplication is the
   demonstrated problem. Normalize line endings only. Do not normalize JSON
   field order, whitespace, Unicode, numbers or arbitrary altered content.
3. Keep raw SHA-256 checks for archived evidence files and manifests whose
   frozen identity is explicitly byte based. Do not rewrite historical bundles,
   pins, contracts or manifests. If a pinned raw contract hash cannot be made
   portable while preserving the frozen verification rule, stop and prepare a
   separately versioned authority correction for review.
4. Add focused tests using LF and CRLF copies of the same contract. Both must
   satisfy the content pin; a changed value, missing field or altered non-newline
   byte must fail closed. Exercise the B tolerance-inheritance path and any
   CUDA authority path touched by the repair. Prove historical replay still
   accepts its exact original bytes where those packages are available.
5. Change only the demonstrated hash call sites. Avoid a general serialization
   rewrite and avoid changing a scientific threshold or decision outcome.

Exit: contract-content verification gives the same result for LF and CRLF;
artifact-byte verification still rejects altered archived evidence; the
historical M1-A `PASS` and M1-B v1 `FAIL` remain unchanged.

## 5. Slice B: CUDA Stage 1 scientific boundary

1. Verify the full-corpus manifest, checkpoint tip, summary and replay bundle
   against their externally retained roots. Consume the completed model-free
   replay result. A fresh 120-epoch capture is not part of this slice. Repeat a
   full memory-bounded replay only when a concrete missing verification requires
   it; report its cost and outcome distinctly from model execution.
2. Recompute the descriptive comparison of the frozen CUDA null extrema with
   the *unchanged* operational tolerances. Preserve family separation:
   `repeated_inference` (120 units), `batch_size_variation` (120 units), and
   `process_restart_variation` (60 units). Report the maximum embedding delta,
   minimum cosine, ranking changes and retrieval-metric deltas by family.
   The observed maximum embedding delta is about `1.94e-7`, while the existing
   functional limit is `1e-5`; observed topology deltas are zero, exactly at
   the frozen critical zero tolerance. This is a scale comparison, not a
   candidate `PASS` or proof of universal equivalence.
3. Audit the frozen sources for an actual Stage 1 release rule: authorized
   inputs, claim, required checks, state vocabulary and approval. The current
   detection-limit authority explicitly says `stage1_release_authorized: false`.
   Do not infer release from a small null or from the old M1-B result.
4. If no frozen rule authorizes that release, write a short, separately
   versioned *proposal* that identifies the source-only claim, exact pinned
   inputs, decision procedure, missing-evidence behavior and review/owner
   approval. Do not choose an operational tolerance from the observed null,
   candidate or consumed holdout. Keep the proposal non-executable until its
   scientific authority is explicitly frozen. If the proposal would revise a
   contract or the meaning of a prior result, stop that transition and state
   the conflict rather than issuing a decision.
5. If an applicable frozen rule is found or subsequently approved, implement
   one pure evaluator over the already verified evidence. Its replay must
   reproduce the released status and every relied-upon input hash without
   creating an ONNX Runtime session. Missing declared evidence returns
   `BLOCKED`; an execution failure is `EXECUTION_ERROR`, never scientific
   `FAIL`. Keep detection-limit results separate from materiality judgments.

Exit: either a replayable Stage 1 decision under an applicable frozen rule, or
a reviewable authority proposal with Stage 1 still unreleased. The latter is a
valid stop; it must not be described as scientific qualification.

## 6. Slice C: M1-B v2 diagnostic closure

The frozen v1 candidate and its `FAIL` are inputs. Use the existing activation
capture, descriptive analysis, structural clusters and causal-plan
preregistration. The only qualifying new model execution role is
`contract_development` with `CPUExecutionProvider`, batch size 16 and the
frozen tokenizer, preprocessing and normalization semantics. No final-holdout
access is authorized.

### C1. Static authority and probe inventory

1. Verify every input in section 3 of
   `docs/M1_TRANSITION_B_V2_DIAGNOSTIC_PROTOCOL.md` before graph loading.
   Reuse `authority.py`, `graph_inventory.py`, `quantization_audit.py`,
   `probe_plan.py`, `instrumentation.py` and the existing fidelity controls.
2. Confirm that the inventory and lineage map cover every required structural
   boundary. Preserve the already frozen probe plan and its hash. No rule may
   depend on an observed node name, index, tensor name or benchmark-specific
   exception. Ambiguous required lineage returns `BLOCKED`.
3. Confirm the derived graph copies match frozen final outputs exactly in
   shape, dtype and values. A mismatch is `BLOCKED`; do not introduce a
   post-observation fidelity tolerance.

### C2. One canonical metric implementation

1. Examine `activation_analysis_metrics.py`, `stage0_metrics.py` and the
   current evidence builders before coding. Reuse or refactor the appropriate
   code; do not build a parallel metric system with different aggregation.
2. Implement the protocol's per-query metrics over recorded tensor pairs:
   maximum and mean absolute delta, relative L2, cosine minimum and mean,
   finite counts, and `symmetric_l2`. Apply the frozen attention mask to all
   relevant elements. Preserve canonical query-ID and C-contiguous element
   order, explicit binary64 conversion, fixed pairwise reduction and
   hexadecimal binary64 serialization. Preserve the specified zero-vector,
   zero-reference and non-finite behavior exactly.
3. Compute saturation per query and the H3 aggregate from all valid integer
   elements; report `NOT_MEASURABLE` when the required integer tensor or
   quantization parameters are unavailable. Never interpret a missing value
   as zero saturation.
4. Retain each per-query value. Derive aggregates from those values. Record the
   metric algorithm version, implementation hash and runtime versions so that
   replay can reject an incompatible implementation.

### C3. Dominant onset and H1-H4

1. Build predecessor edges from the nearest upstream *matched* probe on each
   tensor-lineage path. Keep attention and residual branches separate; a
   topological list neighbor is not automatically a predecessor. Use a
   synthetic zero-score graph-input predecessor where needed.
2. Score each edge with `max(0, child_score - predecessor_score)` using the
   canonical arithmetic. Retain all maximum-growth ties by exact decoded
   binary64 bit pattern. Zero maximum growth makes every hypothesis
   `UNRESOLVED`.
3. Apply the frozen family-set rule for H1, H2 and H4, and the all-boundaries
   measurable plus aggregate-saturation rule for H3. Emit only `SUPPORTED`,
   `NOT_SUPPORTED` or `UNRESOLVED`; labels are structural diagnostics and never
   causal proof or a new Transition B result.

### C4. Package and replay

1. Emit the exact required artifacts from section 8 of the diagnostic
   protocol, including `probe-observations.npz`, `activation-divergence-report.json`,
   `hypothesis-report.json`, `replay-bundle.json` and a manifest hashing every
   declared artifact. Numeric arrays must load with NumPy `allow_pickle=False`.
2. Replay from recorded probe tensors and frozen static artifacts only. Do not
   load or run either model. Recompute every per-query metric, aggregate,
   predecessor edge, dominant-onset tie and hypothesis label; compare exact
   canonical values and input/artifact hashes.
3. Add synthetic tests for masks, parallel branches, tied onset, zero growth,
   non-finite inputs, zero vectors, missing probes, changed order, corrupted
   hashes and missing required artifacts. Each absence or mismatch must be
   `BLOCKED`; a valid technical interruption is `EXECUTION_ERROR`.
4. Only after these tests and the static gate pass, run the frozen diagnostic
   execution on `contract_development`, package it, and independently replay
   it. Record the external manifest SHA-256. Stop at `COMPLETE` when inventory,
   fidelity, observations, metrics, H1-H4 and replay all agree.

Exit: one tamper-evident M1-B v2 diagnostic package with `COMPLETE` and
model-free replay. The v1 `FAIL` remains the scientific Transition B result.

## 7. Validation and delivery rules

For each code slice, run the focused new and existing tests that cover its
changed invariant. Before presenting a final stacked implementation head, run
the repository's offline gates in this order:

```powershell
python -m pytest -q
ruff check .
python -m black --check --workers 1 -- src tests
python -m mypy src
python -m compileall -q src tests
```

Run historical model-free replay only where the exact frozen inputs are
available and a specific compatibility question requires it. Do not label an
unavailable replay as passed. CI is an engineering check, not qualifying
scientific evidence. Keep changes modular; do not grow a single capture,
decision or replay file into a new monolith.

For each slice, record changed files, exact command results, artifact roots,
verified hashes, status and claim boundary. Commit coherent completed work;
preserve unrelated working-tree changes. Push, open a PR or publish a new
scientific authority only under the repository owner's applicable authorization.

Stop immediately at a scientific boundary requiring a new authority, a failed
frozen identity, a violated data role, or missing replay evidence. Record
`BLOCKED` with the exact reason. Stop a technical attempt as `EXECUTION_ERROR`
without changing or replacing numerically inconvenient complete observations.
