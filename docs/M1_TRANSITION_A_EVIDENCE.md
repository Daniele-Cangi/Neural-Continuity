# M1 Transition A evidence

## Scope

This record covers only Transition A:

```text
PyTorch FP32 real teacher -> ONNX FP32
```

It is evidence-bounded continuity evidence, not a claim of universal equivalence. It does
not include ONNX INT8, calibration, or any Transition B execution.

## Captured package

The real CPU capture completed on `2026-08-06`:

```text
C:\tmp\nc-m1-transition-a-20260806T193853
```

- evidence manifest SHA-256:
  `12566ccbcc7f3f74a799abca2189a9b0906efd44a0f038ce0dc7c44b7b87fc3a`;
- contract: `m1-transition-a-v1`, SHA-256
  `772e0df5133de09f6108cb42144e9b2ee69e47c0694bdf5b60ca4d88c18ee5c4`;
- materialized SciFact manifest SHA-256:
  `0746d98f5e69c6a0ee48ca3f47b342de1d968a877c90df26ffe8f893437fd5de`;
- teacher: `sentence-transformers/all-MiniLM-L6-v2` at
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`;
- environment: Python `3.12.10`, PyTorch `2.10.0+cpu`, ONNX `1.22.0`, ONNX Runtime
  `1.28.0`, Sentence Transformers `5.6.1`, NumPy `2.5.1`;
- provider: `CPUExecutionProvider` only;
- corpus: `5183` documents and `1109` queries;
- paired source/target runs: batch sizes `1`, `16`, and `64`.

The ONNX graph has `596` nodes, inputs `input_ids`, `attention_mask`, and
`token_type_ids`, and output `embeddings`. The exported graph SHA-256 is:

```text
5c0d999bd6b5e64e36cad1f61a83ef8e7507d55be49086745780fabb7c648511
```

## Decision

`transition_a_status` is `PASS` for every paired run. The final aggregate is `PASS` and
therefore satisfies the prerequisite that permits, but does not itself implement,
Transition B.

Maximum observed functional differences:

| Batch | Document max abs delta | Query max abs delta | Minimum cosine |
|---:|---:|---:|---:|
| 1 | `2.4586915969848633e-07` | `2.4586915969848633e-07` | `0.9999998807907104` |
| 16 | `2.384185791015625e-07` | `2.4959444999694824e-07` | `0.9999998807907104` |
| 64 | `2.4028122425079346e-07` | `2.4959444999694824e-07` | `0.9999998807907104` |

For all three runs and all six data roles, ranking changes are `0` and decreases in
`Recall@10`, `MRR@10`, and `NDCG@10` are `0.0`. The blocking `frozen_critical` role and
the one-shot `final_holdout` role both pass at every required batch size.

## Replay and integrity

Replay of the exact generated bundle completed `PASS` without source or target model
execution. It reproduced the aggregate `PASS`, all role outcomes, rankings, comparison
report, and decision. The external package is deliberately not committed because it
contains the SciFact-derived observations and the ONNX model; its integrity is
tamper-evident rather than physically immutable.

Artifact SHA-256 values declared by the manifest:

- `comparison-report.json`:
  `5ae1b95b1d9ef33028aca98955c0ad592e8a3b731bc4497d243e6ae1b8ddca90`;
- `decision.json`:
  `9d37990e05c3d3ce0a1fef878d8472fcd638b16dcdbafe22b5b55a9ce98fa7ec`;
- `onnx-manifest.json`:
  `51788194d0d0e22b6e730f4b1163954071948c8f5e8aac474758b75952063f6e`;
- `replay-bundle.json`:
  `f3b07edb75dbd3d4e469b08ea086156d6421735da47eec818b871a5d9196c1cb`;
- `source-rankings.jsonl`:
  `630e167c0f629edcb585a3a348838135879e0438c1b78f06d5c0c7a3e7abdf97`;
- `target-rankings.jsonl`:
  `de49645d82c6bbf47873da38b1a753b79892e651e59f0c2c97ad6375381a8dee`;
- `teacher-manifest.json`:
  `c7a4548f10bcf8229c70d5fa7ef8676c7f9b7b03e5a1a515f2dad093f43ed724`;
- `teacher.onnx`:
  `5c0d999bd6b5e64e36cad1f61a83ef8e7507d55be49086745780fabb7c648511`;
- `transition-observations.npz`:
  `a386a1a42c6342b71898d4b55c661ffc8f6cf63e1c56138189c7934dc30f2cda`.

## Technical caveats

The verified run used the legacy TorchScript-based PyTorch ONNX exporter with dynamic
axes. PyTorch emitted a deprecation warning for that exporter, and Transformers emitted
trace warnings about Python boolean branches and advanced indexing. The graph passed
`onnx.checker`, loaded with the frozen CPU provider, and completed all required variable
batch captures. These warnings are recorded technical follow-up items; they are neither
scientific regressions nor a basis for claiming portability to other models, providers,
hardware, or exporter implementations.
