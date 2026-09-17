# M1 CUDA sentinel: pre-execution review packet

Status: `REVIEW_REQUIRED_NOT_EXECUTABLE` (2026-09-17). This note is not an
authority, a protocol amendment, or permission to load an ONNX graph.

## Evidence already available

- The source-only rebuilt-runtime preflight is technical `PASS`, with externally
  recorded artifact-manifest SHA-256
  `9340a7a52ff6502d9ee6639dc3c15d41325c3521c77629778e44f8d7043bd8d1`.
  It is not a measurement-null epoch or a qualification result.
- The single-epoch package writer, checkpoint chain, and combined model-free
  replay have synthetic fail-closed tests (PRs #56-#58). They do not establish
  live-runtime or 120-epoch correctness.
- At inspection, D: reported 286,400,974,848 free bytes (266.73 GiB). This is
  a mutable capacity snapshot, not a reserved execution budget.
- The frozen layout's raw `float32` embeddings alone require 248,463,360 bytes
  (0.231 GiB) for 120 sentinel epochs and 3,881,041,920 bytes (3.615 GiB) for
  120 full-corpus epochs. These figures exclude checkpoint duplication, profiles,
  manifests, temporary files, filesystem overhead, and other evidence.

## Gates still open

1. The frozen config remains `DRAFT_NOT_EXECUTABLE`; its independent review,
   fail-closed replay, and fresh-preflight freeze fields remain `false`. Do not
   edit the pinned config or infer approval from merged plumbing PRs. A reviewed
   new authority version must explicitly grant only the intended next phase.
2. The qualifying materialization root is `0746d98f5e69c6a0ee48ca3f47b342de1d968a877c90df26ffe8f893437fd5de`.
   The historical technical preflight observed a different materialization root.
   The provenance split must be reviewed, not silently reconciled.
3. A fresh static authority and runtime/DLL verification must pass before any
   graph load. The exact source, tokenizer, partition, IDs, roles, qrels, and
   provider policy must be bound to the reviewed authority.
4. Storage headroom must be rechecked against an explicit peak-space allowance
   at launch. The 64-item source-preflight timings are descriptive only; they do
   not establish a safe upper bound for 120 fresh-process epochs. A runtime
   window and interruption/resume procedure require review.
5. An independent reviewer must approve the unchanged 120-epoch sentinel design,
   attempt retention, external hash anchoring, replay coverage, and fail-closed
   behavior. The sentinel is technical and non-qualifying; full corpus needs a
   separate pre-execution review after sentinel integrity and replay pass.

Until these gates close, no CUDA sentinel, full-corpus epoch, INT8 run, holdout
access, tolerance change, or scientific continuity decision is authorized.
