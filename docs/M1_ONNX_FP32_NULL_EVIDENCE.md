# M1 ONNX FP32 source measurement null

The verified Transition A ONNX FP32 graph was measured as the source instrument for
Transition B. No INT8 artifact was created.

- package: `C:\tmp\nc-m1-onnx-null-20260806T211943`;
- evidence manifest SHA-256:
  `506ab742aac5abbb8558ce20e714d4d5baf2c3785bb17de0d9fbdb22ad84c123`;
- source ONNX SHA-256:
  `5c0d999bd6b5e64e36cad1f61a83ef8e7507d55be49086745780fabb7c648511`;
- source Transition A manifest SHA-256:
  `12566ccbcc7f3f74a799abca2189a9b0906efd44a0f038ce0dc7c44b7b87fc3a`;
- provider: `CPUExecutionProvider`;
- six source-only runs, `5183` documents, `81` `measurement_null` queries;
- model-free replay: `PASS`.

Repeated inference at batch `64` produced zero embedding, ranking, and retrieval-metric
delta across two comparisons. Batch variation at `1`, `16`, and `64` produced maximum
document delta `1.7881393432617188e-07`, maximum query delta
`4.470348358154297e-08`, zero ranking changes, and zero `Recall@10`, `MRR@10`, and
`NDCG@10` delta. Minimum observed cosine was `0.9999997615814209` for documents and
`0.9999998211860657` for queries.

These are empirical detection limits, not operational tolerances and not a Transition B
decision. The static INT8 candidate remains prohibited until this evidence is reviewed.
