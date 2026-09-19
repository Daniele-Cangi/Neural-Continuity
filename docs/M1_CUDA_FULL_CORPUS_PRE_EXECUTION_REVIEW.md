# CUDA full corpus: independent pre-execution review

Status: `PRE_EXECUTION_REVIEW_PASS` (2026-09-18). This is an artifact-based
review of the frozen design and the repository state at commit `4032d07`. The
review was performed in a separate Codex task from the implementation work,
without relying on prior implementation reasoning. It is not an executable
authority, a claim of human or organizational review, or a scientific release.

## Reviewed identities

| Item | SHA-256 |
|---|---|
| CUDA qualification protocol, `docs/M1_CUDA_NULL_QUALIFICATION_PROTOCOL.md` | `eb3db027c73d989349bf3a11a8c734a7c4c175701f214acb288e39247676c7d3` |
| Frozen CUDA config, `experiments/m1-cuda-null-v1.yaml` | `df1a171d93e7fa07909e3f24b552baccd592a7239c5e6a5c4e3f972c475551d3` |
| Full-corpus proposal, `experiments/m1-cuda-null-full-corpus-proposal-v1.json` | `4e78cd0497becbde1c974369c6fbac82cda190cffae006c2e1892790a161859c` |
| Sentinel authority, `experiments/m1-cuda-null-sentinel-authority-v2.json` | `987e038164916549e9bda3f1ce362c7f8b6490ab45875b73c4228863621259eb` |
| Sentinel epoch 120 checkpoint, `D:\neural-continuity-evidence\nc-m1-cuda-sentinel-v2-20260917\chain\epoch-0120.json` | `ded69e251d15dd6360deeeaaf970435ddb6b1e004086ea30274b53766fbc48ca` |
| Sentinel technical-gate manifest, `D:\neural-continuity-evidence\nc-m1-cuda-sentinel-technical-gate-20260918\artifact-manifest.json` | `331411a9523690afaf0e56678141e4b2358151fa817e91b6366d811edd12a1b3` |
| Budget-preflight v2 spec, `experiments/m1-cuda-full-corpus-budget-preflight-v2.json` | `5ecd713388e1947c0aa88c45ae67e62c0792f4783f21b0e9b75ad0b3736dc60c` |
| Budget-preflight manifest, `D:\neural-continuity-evidence\nc-m1-cuda-full-corpus-budget-preflight-v2-20260918\artifact-manifest.json` | `93aa797cebad75892bc4771ef24d5c4893978d8fc1f4cab45c40fddbb897ef6a` |

The proposal, sentinel authority, technical gate, and budget preflight agree on
the qualifying SciFact materialization manifest
`0746d98f5e69c6a0ee48ca3f47b342de1d968a877c90df26ffe8f893437fd5de`
and source ONNX FP32 artifact
`5c0d999bd6b5e64e36cad1f61a83ef8e7507d55be49086745780fabb7c648511`.
The historical preflight's different materialization root is explicitly
declared and is not substituted for the qualifying root.

## Review conclusion

No blocking design mismatch was found. The proposal retains 120 separate
process epochs, all 5,183 documents and 81 measurement-null queries, the four
fixed source-only passes, all three fixed comparison families, and no early
stopping or adaptive sample size. It leaves INT8, holdout access, operational
tolerance changes, and scientific release disabled. The sentinel technical gate
replays as `TECHNICAL_GATE_PASS_NO_SCIENTIFIC_RELEASE`; the budget package
replays as `TECHNICAL_TIMING_CAPTURED_NOT_QUALIFYING`. The latter's 13.383-hour
linear projection is explicitly not an upper bound or a stopping rule. Its
failed v1 profiling attempt is disclosed.

The subsequent executable authority must implement the existing protocol's
attempt retention, hash-anchored same-epoch resume, complete raw observation
retention, actual-run provider/operator accounting, and memory-bounded model-free
replay of rankings, metrics, extrema, and coverage. The budget preflight's
64-item provider profiles are technical samples; they cannot stand in for
provider accounting on the qualifying full-input passes. The sentinel runner is
sentinel-only and has no qualifying resume path. These are requirements for the
future implementation, not changes to the reviewed design. This review does
not certify an executor that does not yet exist or authorize a full-corpus epoch.

## Reproduction

In this repository, verify the file hashes above, then run:

```powershell
python -m neural_continuity.m1_diagnostics.cuda_null_full_budget_replay replay `
  --bundle D:\neural-continuity-evidence\nc-m1-cuda-full-corpus-budget-preflight-v2-20260918\replay-bundle.json `
  --manifest-sha256 93aa797cebad75892bc4771ef24d5c4893978d8fc1f4cab45c40fddbb897ef6a
```

The observed result was `replay_status: PASS`, `model_loaded: false`, and
`full_corpus_qualification_execution_authorized: false`. The budget replay
recomputes the sentinel technical gate and checks the pinned source, dataset,
query, and qrels identities. Relevant offline tests passed:
`tests/test_m1_cuda_null_full_budget.py`,
`tests/test_m1_cuda_null_sentinel_postgate.py`, and
`tests/test_m1_cuda_null_sentinel_chain_replay.py`.
