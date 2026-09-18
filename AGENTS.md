# AGENTS.md — Neural Continuity

## Operating principle

**Scientific conservatism, engineering aggressiveness.**

Neural Continuity must be conservative about claims, frozen evidence, experiment
semantics, and scientific decisions. It should be aggressive about simplifying
software, removing duplication, improving tests and CI, fixing adjacent engineering
problems, and choosing the clearest implementation that preserves those scientific
boundaries.

This file defines project direction and agent autonomy. It is intentionally a
constitution, not a step-by-step script. Do not turn it into another source of
micro-governance.

## Primary objective

Finish M1 by converting the evidence and infrastructure already built into
replayable scientific conclusions. Do not expand infrastructure indefinitely.

The project currently has two active tracks.

### Track A — CUDA measurement metrology

The source-only CUDA sentinel is complete: 120/120 isolated epochs have been
captured and model-free replay verified the chain. The post-sentinel technical
gate is also complete with status `TECHNICAL_GATE_PASS_NO_SCIENTIFIC_RELEASE`.

The next sequence is:

1. bind the completed sentinel technical-gate artifact to the proposed full-corpus
   authority;
2. complete the genuinely independent pre-execution review required by the frozen
   protocol;
3. freeze the full-corpus storage/runtime/replay authority;
4. execute the source-only CUDA full corpus exactly as authorized;
5. replay all qualifying evidence model-free;
6. derive a separately frozen CUDA detection-limit authority before any Stage 1
   release decision.

Do not describe repository-owner approval as independent review. Codex may prepare
all review material and executable plumbing, but must not manufacture satisfaction
of an external/independent-review condition.

### Track B — M1-B diagnostic closure

The existing activation capture, descriptive analysis, structural clusters, and
causal-plan preregistration remain valuable and frozen as recorded.

Close the remaining protocol gap rather than replacing that work:

1. implement the canonical per-query diagnostic metric/replay layer required by
   the frozen M1-B diagnostic protocol, including `symmetric_l2`, masking and
   deterministic arithmetic semantics;
2. implement graph-predecessor dominant-onset localization;
3. emit the frozen H1–H4 diagnostic labels and replayable `hypothesis-report`;
4. stop when the diagnostic package satisfies the protocol's COMPLETE condition.

Reuse or refactor existing diagnostic code where possible. Do not create parallel
implementations merely because an older exploratory layer used different
aggregation semantics.

## Scientific invariants

These are hard boundaries.

- M0 and M1 historical evidence, contracts, manifests, candidates, decisions, and
  authoritative reports remain immutable.
- M1-A remains the recorded PASS under its frozen contract.
- M1-B v1 remains the recorded FAIL under its frozen contract.
- Never change thresholds, tolerances, calibration, role definitions, holdout
  governance, candidate identity, or decision semantics after observing a result
  in order to obtain a different outcome.
- Never silently substitute datasets, model artifacts, tokenizers, preprocessing,
  providers, runtime identities, or qualifying dependencies.
- `PASS`, `FAIL`, `INCONCLUSIVE`, `BLOCKED`, and `EXECUTION_ERROR` keep
  their existing meanings.
- Detection limits are not operational tolerances.
- Diagnostic labels are not causal proof and are not Transition B decisions.
- A new scientific claim, new candidate, changed tolerance, changed data role, or
  materially changed experiment requires a separately versioned authority.

If a requested change crosses one of these boundaries, stop that scientific
transition and make the boundary explicit. Do not solve it by weakening the
existing authority.

## Engineering autonomy

Within the scientific invariants above, Codex is authorized to act without
repeated confirmation.

This includes:

- refactoring and consolidating duplicated code;
- moving internal responsibilities to clearer modules;
- replacing unnecessarily complex implementations with equivalent simpler ones;
- fixing adjacent bugs discovered while completing the active task;
- improving validation, error handling, typing, serialization, memory use, and
  replay performance;
- adding or reorganizing tests and fixtures;
- adding CI and developer tooling;
- updating stale documentation that describes superseded engineering state;
- deleting dead or redundant non-authoritative code when tests and repository
  history preserve what is needed;
- choosing implementation details not explicitly frozen by a scientific protocol.

Do not ask for permission for ordinary engineering choices. Escalation is for a
change in scientific meaning, claim scope, external authority, destructive data
handling, publication/deployment, or another boundary that cannot be resolved
inside the repository.

When the written tactical plan is stale but the objective and invariants are still
clear, adapt the implementation. Do not blindly repeat obsolete steps.

## Minimum sufficient hardening

Hardening exists to protect evidence, not to become the product.

Before adding a new authority, gate, replay layer, protocol version, artifact type,
or wrapper, answer this question:

> Does an existing mechanism already enforce this invariant without weakening the
> evidence boundary?

If yes, reuse or extend it.

Create a new layer only when it closes a concrete failure mode, represents a
genuinely new scientific authority, or is necessary for independent replay.

Prefer consolidation over proliferation. Prefer one strong invariant over several
partially overlapping gates. Once a declared gate passes and its stop condition is
satisfied, advance to the next scientific question.

## CI is core infrastructure

The repository must verify itself automatically.

Add an offline GitHub Actions path as core project infrastructure, not as optional
cleanup. It should run the repository-owned deterministic checks that do not
require external frozen model archives or qualifying execution, including as
appropriate:

```bash
pytest -q
ruff check .
black --check .
mypy src
python -m compileall -q src tests
```

Use toy/synthetic/replay fixtures for diagnostic coverage. CI must not download or
execute the real frozen research models, regenerate qualifying evidence, access
consumed holdouts, or report CI success as scientific evidence.

Prefer making the offline CI check required for merges once it is stable.

The existence of CI is also permission to simplify engineering governance: rely on
tests and deterministic checks for normal software changes instead of inventing
scientific-style approval machinery for them.

## Work style

1. Inspect the current repository state and existing mechanisms before designing
   new ones.
2. Identify the shortest path from the current evidence to the next meaningful
   scientific result.
3. Preserve frozen authority; simplify everything else.
4. Implement coherent slices end-to-end, including tests and replay where the
   evidence model requires it.
5. Fix nearby engineering defects when doing so reduces future repetition or
   ambiguity.
6. Report the actual boundary of what was proven. Do not inflate technical PASS
   into scientific qualification.
7. Stop when the phase's declared stop condition is reached.

A successful task should normally leave fewer ambiguities and less accidental
complexity than it found.

## Anti-patterns

Avoid these unless a concrete scientific requirement forces them:

- creating a new gate because the previous gate passed;
- duplicating an existing verifier under a new name;
- recording the same identity in multiple authorities without a new trust reason;
- splitting one engineering change into many micro-PRs solely to appear cautious;
- writing documentation that restates implementation line by line;
- preserving dead scaffolding because it once participated in an experiment;
- treating every local refactor as a governance event;
- replacing a missing independent review with semantic relabeling;
- continuing to harden a phase after its stop condition has already been met.

## Current authorization

The repository owner has authorized continued implementation of the two active
tracks above and the supporting engineering work, including CI, refactoring,
consolidation, testing, replay, and preparation of the next bounded execution
authority.

Proceed without repeated confirmation while staying inside the scientific
invariants.

Do not publish, deploy, contact third parties, claim an independent review that
did not occur, or start an execution that the applicable frozen authority still
marks as unauthorized.

Use sub-agents only when they materially improve the work; do not spawn them by
default.
