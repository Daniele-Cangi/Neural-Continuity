# M1 Teacher Baseline Evidence Record

## Scope

Questo record identifica la prima baseline reale M1 per
`sentence-transformers/all-MiniLM-L6-v2`. Qualifica il teacher PyTorch FP32 e
il suo pacchetto di osservazioni replayabile; non produce alcuna decisione per
la transizione A (`PyTorch FP32 -> ONNX FP32`) o B (`ONNX FP32 -> ONNX INT8`).

Gli artifact non sono versionati nel repository: includono dati SciFact ed
embedding derivati. Sono identificati da hash SHA-256 e devono essere forniti
separatamente per il replay.

## Identita degli artifact

| Artifact | SHA-256 |
|---|---|
| Materialization manifest SciFact v1 | `0746d98f5e69c6a0ee48ca3f47b342de1d968a877c90df26ffe8f893437fd5de` |
| Teacher evidence manifest | `9caae6846b59c40bea3fe08eff5d2f2edb00edd1b5d9ef10bc8f5c3854370c51` |
| Canonical observations | `8882c98fcd21e0413da41b38fdc11a6cfd3694bae02b955aca5b161985f2f56c` |
| Rankings | `cb5e1884411c910976e9d303528ee0ab69192ddd9e13bb299f2002d55d818fde` |
| Replay bundle | `0779cc2a7bfedda7d8491fb587bcffc19992373ab4b41d71b13135b155524e88` |

Il manifest elenca cinque artifact e una verifica indipendente ha rilevato
`0` hash mismatch.

## Teacher e ambiente congelati

- Model ID: `sentence-transformers/all-MiniLM-L6-v2`
- Revision: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- Device: `cpu`
- Output: `float32`, normalizzazione L2 post-encode
- Python: `3.12.10`
- PyTorch: `2.10.0+cpu`
- Sentence Transformers: `5.6.1`
- Snapshot files hashati: `11`
- CUDA: non misurato, quindi `unverified`

Python 3.13 non e stato usato per questa evidence: PyTorch Windows stabile e
qualificato e stato installato nel virtual environment isolato Python 3.12.

## Copertura e risultati baseline

- Corpus: 5.183 documenti
- Query: 1.109
- Qrel: 1.258
- Ranking: top-10, pareggi risolti per `document_id` crescente

| Ruolo | Query | Recall@10 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| `measurement_null` | 81 | 0.758436 | 0.611160 | 0.642486 |
| `quantization_calibration` | 162 | 0.772634 | 0.608451 | 0.643459 |
| `contract_development` | 364 | 0.810668 | 0.656119 | 0.690646 |
| `validation` | 202 | 0.761551 | 0.592272 | 0.625867 |
| `frozen_critical` | 60 | 0.710000 | 0.568373 | 0.596290 |
| `final_holdout` | 240 | 0.801667 | 0.613813 | 0.657280 |

Questi valori sono baseline descrittive del teacher. Non sono soglie, non
autorizzano tuning sul holdout e non sono un claim di equivalenza universale.

## Replay

Il replay ha restituito:

```text
status               PASS
replay_verified      true
model_execution_used false
document_count       5183
query_count          1109
```

Il replay ricalcola ranking e metriche dall'NPZ e dai qrel inclusi nel bundle;
non costruisce e non invoca il teacher. Artifact dichiarati mancanti, hashati
diversamente o semanticamente non coerenti devono restare `BLOCKED`.

## Stato successivo

La baseline teacher e ora congelata e replay-verificata. Il prossimo lavoro
autorizzato e definire il contratto della transizione A prima di qualunque
export ONNX; quantizzazione, Spectra, distillazione, pruning e routing restano
fuori scope.
