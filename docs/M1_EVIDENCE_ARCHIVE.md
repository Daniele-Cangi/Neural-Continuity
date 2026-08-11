# M1 v1 evidence archive

## Status

- Repository merge commit: `77f31586ccc47cdbbe00afdcba149a8bd78c5a02`
- Local archive:
  `C:\dev\neural-continuity-evidence\m1-v1-77f31586ccc47cdbbe00afdcba149a8bd78c5a02`
- Tracked inventory: `docs/m1-v1-archive-manifest.json`
- Archive-manifest SHA-256:
  `2b01debd62b6503251b4f7755ef1a5c7a84834aebe721d5c90d653d46ad8d064`
- Authoritative packages: `11`
- Package files: `53`
- Package bytes: `451868451`
- Frozen contracts: `2`
- Copy verification: every source and destination size and SHA-256 matched.
- Remote redundancy: `NOT_VERIFIED`

This archive is tamper-evident through SHA-256 inventories. It is not claimed to
be physically immutable. The repository records the complete file inventory so
that loss or alteration of the local archive can be detected.

## Package authorities

| Package | Authority | SHA-256 |
|---|---|---|
| `nc-m1-scifact-v1-canonical-20260806` | `materialization-manifest.json` | `0746d98f5e69c6a0ee48ca3f47b342de1d968a877c90df26ffe8f893437fd5de` |
| `nc-m1-scifact-v1-materialized-20260806` | `materialization-manifest.json` | `beab716b9f322478ca3f2efd0e6e93e7d66a2b3483ed098941cd9f2275bcdcc2` |
| `nc-m1-teacher-baseline-py312-v2-20260806` | `evidence-manifest.json` | `9caae6846b59c40bea3fe08eff5d2f2edb00edd1b5d9ef10bc8f5c3854370c51` |
| `nc-m1-source-null-20260806T163954` | `evidence-manifest.json` | `f3250f96577c1594b356c89252e3482e914f8b891d1bedbe84040206743a2f3d` |
| `nc-m1-transition-a-20260806T193853` | `evidence-manifest.json` | `12566ccbcc7f3f74a799abca2189a9b0906efd44a0f038ce0dc7c44b7b87fc3a` |
| `nc-m1-onnx-null-20260806T211943` | `evidence-manifest.json` | `506ab742aac5abbb8558ce20e714d4d5baf2c3785bb17de0d9fbdb22ad84c123` |
| `nc-m1-static-calibration-20260806T224805` | `calibration-manifest.json` | `3ac7d68e01976ee444217cd80c5b4b7338f870d8c0ab5a350a960495baef0778` |
| `nc-m1-static-qdq-verified-20260806T230647` | `candidate-manifest.json` | `d11888e48e24a9e29f5bdfac48ad7ace4204fb7b101e3531faa0f11190ad562c` |
| `nc-m1-int8-observation-20260806T232512` | `evidence-manifest.json` | `4027c1edf9f24254e6174ca79bc722c98758c8f97f5ad175b380866f64063a80` |
| `nc-m1-fp32-paired-source-20260807T101905` | `evidence-manifest.json` | `cf03882df0913e84b456b61f02a1c00a14ec151cd0fd9cc07f7d0bf04745b4df` |
| `nc-m1-transition-b-decision-20260808T014916` | `evidence-manifest.json` | `eed7d7af553ae9aa77274104cc75f348de910df464d836272ab37e8760e78d4e` |

## Explicit exclusions

The archive does not promote these non-authoritative intermediates:

| Directory | Reason |
|---|---|
| `nc-m1-fp32-observation-20260807T002914` | Superseded by the canonical paired FP32 source package. |
| `nc-m1-static-qdq-20260806T230404` | Unverified intermediate candidate. |
| `nc-m1-cpu-py312` | Execution environment, not qualifying evidence. |

## Restore and replay boundary

Frozen replay bundles contain absolute paths. Preservation therefore does not
make those bundles path-portable and does not authorize rewriting them.

1. Never edit an archived package or its tracked inventory.
2. Restore a package only when its recorded original path is absent; never
   silently merge with or overwrite an existing directory.
3. Restore the exact package directory name under its recorded `C:\tmp` path.
4. Verify the tracked inventory and each package authority before replay.
5. Use the frozen contracts from the repository only when their SHA-256 values
   match the archived contract records.

For the final Transition B replay, the decision package, paired FP32 source,
INT8 target, and both contracts must be available at their recorded paths.

