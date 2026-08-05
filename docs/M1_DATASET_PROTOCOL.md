# M1 Dataset Protocol - SciFact v1

## 1) Stato e autorita

Questo documento definisce il protocollo dati per la prima evidence qualificante M1.
Il manifest machine-readable autoritativo e
`experiments/m1-scifact-v1.manifest.json`.

Stato corrente: `specified_not_materialized`.

Questo stato non abilita claim M1. Il dataset diventa utilizzabile come evidence
qualificante solo dopo materializzazione, audit e congelamento degli artifact previsti
dal quality gate di questo documento.

## 2) Claim boundary

Il dataset misura continuita di retrieval su claim scientifici in lingua inglese contro
un corpus di abstract scientifici. Non misura equivalenza universale, robustezza fuori
dominio, qualita generativa o accuratezza clinica.

La fixture locale a cinque query resta esclusivamente un preflight di plumbing e non e
parte di questo protocollo.

## 3) Sorgente congelata e licenze

Sorgente selezionata: SciFact nella distribuzione retrieval BEIR.

- Archivio: `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip`
- Pagina BEIR: `https://github.com/beir-cellar/beir/wiki/Datasets-available`
- Repository originario: `https://github.com/allenai/scifact`
- Licenza claim e annotazioni: CC BY 4.0.
- Licenza abstract del corpus: ODC-By 1.0, tramite S2ORC.
- Licenza del codice SciFact: Apache-2.0; il codice upstream non entra nel dataset M1.

BEIR e un distributore del formato e non sostituisce l'autorita delle licenze
originarie. La redistribuzione futura dei dati deve preservare attribuzione e termini
applicabili; il manifest e gli hash non trasferiscono diritti sui contenuti.

Identita dello snapshot acquisito il 2026-08-05:

| Artifact | SHA-256 |
|---|---|
| `scifact.zip` | `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165` |
| `corpus.jsonl` | `dec31c8182f3d744c7d2c09423756fd1d17cbef75808db13ba01cc0aab4d1ac6` |
| `queries.jsonl` | `8ff84a7c903f722981cd8d595c022660140c51867b27608a6d4910db86080313` |
| `qrels/train.tsv` | `a53f2114831916c096b6c37d9e54da68cef4efdcdbd5ed46533601af972acf1d` |
| `qrels/test.tsv` | `0864bb985e0ca2367ba217977e72004d549054b2b06666ed9d4825ac7c21284c` |

L'MD5 upstream dell'archivio, mantenuto solo come controllo di compatibilita con BEIR,
e `5f7d1de60b170fc8027bb7898e2efca1`. SHA-256 resta l'autorita locale.

## 4) Semantica di materializzazione

- Encoding sorgente: UTF-8.
- Gli identificatori query e documento sono stringhe opache.
- Il testo query e il valore JSON `text`, senza lowercase o normalizzazione whitespace.
- Il testo documento e `title + "\n" + text` quando `title` non e vuoto, altrimenti `text`.
- Non sono ammessi deduplicazione, stemming, traduzione o riscrittura.
- Una relazione e rilevante quando lo score qrel e strettamente maggiore di zero.
- Il corpus candidato comprende i 5.183 documenti per ogni query di valutazione.
- I pareggi di ranking sono risolti per `document_id` crescente.
- Record mancanti, ID duplicati, qrel orfani o valori non finiti devono fallire chiusi.

Le scelte sopra sono preprocessing scientifico e devono essere incluse nell'identita
del pacchetto materializzato. Un cambiamento richiede una nuova versione del dataset.

## 5) Partizionamento deterministico

L'unita di assegnazione e la query. Gli ID query dei ruoli sono disgiunti; il corpus e
un universo retrieval condiviso e read-only, non un campione di calibrazione.

Per ogni split upstream, calcolare:

```text
sha256(utf8("neural-continuity:m1:scifact:v1:" + upstream_split + ":" + query_id))
```

Ordinare per digest crescente e, in caso di collisione, per `query_id` crescente come
byte UTF-8. Applicare quindi gli intervalli fissati nel manifest, senza rebalance o
fallback dipendente dai dati.

| Ruolo | Split upstream | Intervallo ordinato | Query | Uso |
|---|---:|---:|---:|---|
| `measurement_null` | train | `[0, 81)` | 81 | Ripetibilita e detectability source-only |
| `quantization_calibration` | train | `[81, 243)` | 162 | Input query per calibrazione futura, mai decisione |
| `contract_development` | train | `[243, 607)` | 364 | Contratto e diagnostica source-only |
| `validation` | train | `[607, 809)` | 202 | Dry-run candidato dopo freeze delle tolleranze |
| `frozen_critical` | test | `[0, 60)` | 60 | Regressioni query-level bloccanti |
| `final_holdout` | test | `[60, 300)` | 240 | Decisione aggregata finale |

La calibrazione usa soltanto i testi query assegnati a
`quantization_calibration`. Se la futura quantizzazione statica richiedera testi
documento, dovra essere introdotto un corpus di calibrazione separato e versionato
prima di aprire qualsiasi risultato holdout.

## 6) Regole anti-contaminazione

- `measurement_null` non partecipa alla promozione.
- `quantization_calibration` non definisce contratto, tolleranze o decisioni.
- `contract_development` puo usare solo evidence del teacher e null measurement.
- `validation` non puo modificare le tolleranze numeriche gia congelate.
- `frozen_critical` e `final_holdout` non possono guidare export, quantizzazione o tuning.
- L'identita del candidato, la trasformazione e il contratto devono essere congelati
  prima di eseguire i ruoli derivati dallo split upstream `test`.
- Un risultato tecnico `EXECUTION_ERROR` o `BLOCKED` non e un `FAIL` scientifico.
- Ogni accesso anticipato a evidence holdout invalida la qualifica e richiede una nuova
  versione del protocollo e una nuova assegnazione.

La separazione e governata e tamper-evident, non fisicamente immutabile o segreta.

## 7) Quality gate di materializzazione

- SHA-256 dell'archivio e dei quattro file sorgente verificati.
- Conteggi sorgente verificati: 1.109 query, 809 query train e 300 query test.
- Conteggi qrel verificati: 919 righe train e 339 righe test.
- Schema, unicita ID, riferimenti qrel e valori finiti verificati fail-closed.
- Ruoli derivati con algoritmo, domain separator e intervalli esatti del manifest.
- Query membership disgiunta ed esaustiva rispetto alle 1.109 query con qrel.
- Ogni file derivato registrato con path relativo, byte size e SHA-256.
- Licenze e citazioni incluse nel pacchetto dati.
- Identita teacher e preprocessing congelati prima della cattura.
- Pacchetto di osservazioni canonico replayabile senza rerun del modello.
- Nessuna tolleranza derivata da candidato, validation o holdout.

Se un gate manca, lo stato resta `BLOCKED` e non viene prodotta evidence qualificante.

## 8) Materializzatore offline

Il materializzatore e implementato in
`src/neural_continuity/dataset_materialization.py`. Non effettua download e non
sovrascrive directory esistenti.

```powershell
neural-continuity-materialize `
  --manifest experiments/m1-scifact-v1.manifest.json `
  --archive <PATH_LOCALE>\scifact.zip `
  --output <NUOVA_DIRECTORY_OUTPUT>
```

Un'esecuzione valida produce `materialized_unqualified`, i sei ruoli, il corpus
canonico, attribution metadata e `materialization-manifest.json`. Non eseguire ancora
export ONNX o quantizzazione.
