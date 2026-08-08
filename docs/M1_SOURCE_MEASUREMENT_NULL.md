# M1 source measurement null

Questa slice misura esclusivamente la variabilita osservabile del teacher PyTorch FP32
reale gia congelato. Non esegue export ONNX, non costruisce un candidato e non emette una
decisione scientifica sulla transizione A.

## Evidenza catturata

La cattura reale e stata eseguita il `2026-08-06` sul pacchetto SciFact materializzato
canonico e sul teacher CPU congelato. Il bundle esterno, non versionato nel repository per
non includere dataset o embedding, e:

```text
C:\tmp\nc-m1-source-null-20260806T163954
```

- SHA-256 `evidence-manifest.json`:
  `f3250f96577c1594b356c89252e3482e914f8b891d1bedbe84040206743a2f3d`
- SHA-256 configurazione:
  `46583620f91e7156d0b9f22e1c1c9135675016717c96f00e4731fd3c900e35aa`
- dataset materializzato:
  `0746d98f5e69c6a0ee48ca3f47b342de1d968a877c90df26ffe8f893437fd5de`
- ambiente: Python `3.12.10`, PyTorch `2.10.0+cpu`, Sentence Transformers `5.6.1`,
  NumPy `2.5.1`;
- teacher: `sentence-transformers/all-MiniLM-L6-v2` alla revisione
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`;
- `5183` documenti, `81` query `measurement_null`, `6` esecuzioni sorgente;
- replay senza modello: `PASS`.

Envelope `repeated_inference` (`2` confronti, batch `64`): delta assoluto embedding
documenti/query `0.0`, cambi ranking `0`, delta `Recall@10`/`MRR@10`/`NDCG@10` `0.0`.
Il minimo coseno documenti osservato e `0.9999998211860657`; quello query e `1.0`.

Envelope `batch_size_variation` (`3` confronti, batch `1`, `16`, `64`): massimo delta
assoluto embedding documenti `1.043081283569336e-07`, query
`9.73232090473175e-08`, cambi ranking `0`, delta
`Recall@10`/`MRR@10`/`NDCG@10` `0.0`. Il minimo coseno documenti e
`0.9999998211860657`; quello query e `1.0`.

Artifact registrati nel manifest:

- `comparison-report.json`:
  `4b20290d2d57e36d1f3307a3aed244b80ca3ceafef0fa886db8546f189d0aaad`;
- `replay-bundle.json`:
  `3d2ed6f79890aa3e0cd0da2120757a9ee26d0a8a41f858f456221418b2ef2293`;
- `source-null-observations.npz`:
  `72ff7de979e0d65a79363c83dd8f85e9cd2a890b46591dd86a1952c038c35412`;
- `source-null-rankings.jsonl`:
  `7776910383c65a828bd51e7ba2700f36d085791cc87c76c0bdb847d74f9a8fb3`;
- `teacher-manifest.json`:
  `de700bba010ed9225a4dcb2dc5481b647588c8166ba4518620c8a145a9e195b5`.

## Disegno

- corpus completo stabile, con ranking contro tutte le `5183` evidenze SciFact;
- sole query del ruolo `measurement_null`;
- tre esecuzioni complete con batch `64`;
- esecuzioni complete aggiuntive con batch `1`, `16` e `64`;
- embedding `float32` normalizzati L2, ranking deterministico e metriche `Recall@10`,
  `MRR@10`, `NDCG@10`;
- envelope empirici separati per `repeated_inference` e `batch_size_variation`.

Gli envelope sono limiti di rilevabilita osservati, non tolleranze operative. Il report
registra quindi `measurement_null_status: CAPTURED_NOT_DECIDED`,
`transition_a_decision: NOT_APPLICABLE` e `operational_tolerance: NOT_SELECTED`.

## Cattura e replay

Con il runtime isolato che contiene il teacher congelato:

```powershell
C:\tmp\nc-m1-cpu-py312\Scripts\python.exe -m neural_continuity.m1_measurement_null capture `
  --config experiments\m1-transition-a-measurement-null.yaml `
  --dataset C:\tmp\nc-m1-scifact-v1-canonical-20260806 `
  --output C:\tmp\nc-m1-source-null-<timestamp>

C:\tmp\nc-m1-cpu-py312\Scripts\python.exe -m neural_continuity.m1_measurement_null replay `
  --bundle C:\tmp\nc-m1-source-null-<timestamp>\replay-bundle.json
```

Il replay non carica il modello. Verifica SHA-256 di ogni artifact dichiarato, tutte le
esecuzioni sorgente richieste, ranking, metriche e envelope ricostruiti. Una esecuzione
sorgente dichiarata ma assente termina `BLOCKED` con
`MISSING_DECLARED_SOURCE_OBSERVATION`.

## Limiti di scope

Il pacchetto non autorizza ancora soglie del contratto A: queste richiedono gli envelope
reali catturati e una decisione di governance separata, senza usare risultati di candidati
o `final_holdout`. Il contratto rimane percio esclusivamente in `contracts/` e non viene
creato o modificato da questo runner.
