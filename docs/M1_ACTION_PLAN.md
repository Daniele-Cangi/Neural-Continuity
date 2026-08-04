# M1 Action Plan — Verified Quantization

## 1) Scope e vincoli

La milestone M1 definisce il percorso per misurare due transizioni reali:

- `PyTorch FP32 → ONNX FP32`
- `ONNX FP32 → ONNX INT8`

M1 non include implementazioni operative, tuning automatico o ricerca ad hoc.
Il piano è documentale e non modifica comportamento M0 prima di un ciclo esplicito di implementazione separato.

## 2) Stato iniziale e non-scope

Non-scope espliciti:

- integrazione con Spectra;
- distillazione;
- potatura;
- stitching;
- routing adattivo;
- early exit;
- abstention runtime;
- conformal prediction;
- ricerca OOD;
- sostituzione provider;
- web UI;
- MLflow;
- certificazione regolatoria;
- claims di equivalenza universale;
- tuning automatico di soglie sul holdout finale.

## 3) Fase M1.0 — Environment and dependency qualification

Obiettivo: raccogliere lo stato ambientale senza modifiche.

Tabelle previste:

| Dipendenza | Valutazione corrente | Stato |
|---|---|---|
| Python | `3.13.5` | available |
| PyTorch | mancante (`ModuleNotFoundError`) | missing |
| ONNX | mancante (`ModuleNotFoundError`) | missing |
| ONNX Runtime | `1.22.1` | available |
| Sentence Transformers | mancante (`ModuleNotFoundError` dipendente da torch) | missing |
| CPU arch | `AMD64` | available |
| Execution providers | `AzureExecutionProvider`, `CPUExecutionProvider` | available |
| CUDA runtime | non disponibile (torch assente) | incompatible |
| teacher cache (`sentence-transformers/all-MiniLM-L6-v2`) | assente | missing |
| ONNX quantization API importabili | non disponibile (`onnx` assente) | missing |

Output della fase:

- elenco verifiche;
- decisione di acquisizione dipendenze da parte di infrastruttura di esecuzione (non qui implementato);
- elenco delle aree bloccate fino a disponibilità pacchetti.

## 4) Fase M1.1 — Real teacher qualification

Proposta:

- identificare modello `sentence-transformers/all-MiniLM-L6-v2`;
- acquisire identità modello:
  - `model_id`;
  - hash file modello/tokenizer;
  - configurazione tokenizer;
  - normalizzazione output;
  - dimensione embedding;
  - lunghezza massima sequenza;
  - comportamento di inferenza deterministico;
  - modalità cache-only nel piano di run;
  - prova di smoke execution.

Regola chiave: nessun claim di idoneità finché qualifica completa.

## 5) Fase M1.2 — Dataset design

Il fixture a 5 query rimane fixture di plumbing e non deve diventare training/evidenza definitiva.

Struttura dataset proposta:

- split `calibration` (estimare variabilità e nullo);
- split `validation` (tuning non decisivo, solo diagnostica);
- split `frozen_critical` (regressioni bloccanti);
- split `final_holdout` (decisione finale).

Gli split includono:

- identificatori stabili di query e documento;
- judgments di rilevanza;
- distractor difficili;
- strati per lunghezza input;
- strati lessicali e semantici.

Regola: holdout non usato per scegliere trasformazioni.

## 6) Fase M1.3 — PyTorch FP32 baseline

L’adapter di baseline deve emettere osservazioni canoniche:

- embeddings query/documenti;
- ranking e score per query;
- observability per query;
- campionamenti di latenza e warm-up;
- throughput;
- RSS/process;
- allocazione CUDA dove disponibile;
- identificazioni runtime/modello;
- batch size e sequenza osservata.

Invarianti richieste:

- bundle replayabile per i test;
- nessuna modifica ai principi di fail-closed M0.

## 7) Fase M1.4 — ONNX FP32 transition

Transizione A:

- `PyTorch FP32 → ONNX FP32`

Punti di piano:

- esportazione ONNX e configurazioni richieste;
- axes statici/dinamici;
- ownership tokenizer/preprocessing;
- normalizzazione/pooling coerenti;
- hashing di graph e artifact;
- metadata runtime e provider;
- confronto funzionale e topologico;
- rappresentazione replay;
- confronto sistemi.

L’errore di export o operatore non supportato deve rendere decisione FAIL-closed.

Nessuna logica di quantizzazione in questa fase.

## 8) Fase M1.5 — ONNX INT8 transition

Transizione B:

- `ONNX FP32 → ONNX INT8`

Prerequisito: transizione A con pacchetto evidenze valida e verificata.

Modalità supportate dove possibili:

- quantizzazione dinamica;
- quantizzazione statica con split calibrazione separato.

Evidenze da catturare:

- metodo e config di quantizzazione;
- origine calibrazione;
- mutazioni graph;
- parametri quantization per pesi/attivazioni;
- operatori esclusi;
- dimensione artifact;
- provider hardware;
- latenza, throughput, memoria;
- evidence funzionale e topologica.

## 9) Fase M1.6 — Detectability and materiality

Definizioni richieste nel contratto:

- `measurement envelope`: differenze distinguibili dalla variazione misurata;
- `acceptance contract`: differenze operativamente non accettabili.

Regola: nessuna soglia operativa può derivare direttamente solo dal null envelope.
Il futuro contratto versionato M1 deve poter rappresentare:

- regressioni a tolleranza zero sul set critico;
- tolleranze direzionali di qualità;
- limiti di warning topologico;
- metriche systems non bloccanti;
- aspettative performance per hardware;
- comportamento quando manca evidence.

## 10) Fase M1.7 — Evidence and replay

Entrambe le transizioni devono essere ricostruibili in replay come unità indipendenti.

Identità transizioni:

- `transition A`: `PyTorch FP32 → ONNX FP32`
- `transition B`: `ONNX FP32 → ONNX INT8`

Il fallimento della transizione A deve impedire la promozione B.

Le evidenze devono includere:

- manifest immutabili per manifest, osservazioni, decisioni e controlli;
- evidenze per transizione;
- hash canonici e tracciamento seed/config.

## 11) Fase M1.8 — Tests and kill conditions

Test da prevedere:

- correttezza adapter;
- parità tokenizer;
- parità pooling;
- parità normalizzazione;
- ordine deterministico;
- integrità export ONNX;
- provider EP mancante;
- input grafico mancante;
- output malformed;
- NaN/inf embeddings;
- mutazione artifact;
- equivalenza replay;
- osservazione dichiarata mancante;
- regressione su critical-set;
- intervallo candidato interno, esterno e crossing boundary;
- miglioramento direzionale.

Test regressione performance come non bloccante salvo contratto esplicito.

Kill conditions predefinite:

- `PyTorch FP32 → ONNX FP32` non riproduce M0 entro contratto difendibile;
- export cambia semantica preprocessing;
- evidence non distingue effetti export da effetti quantizzazione;
- contaminazione validation/holdout da calibrazione;
- replay non ricostruisce la decisione;
- risultati dipendenti da stato runtime non dichiarato.

## 12) Piano cambio file (M1 planning map)

| Path | Azione | Responsabilità | Motivo | Dipendenze | Test | Artifact |
|---|---|---|---|---|---|---|
| `src/neural_continuity/models.py` | modify | manifest e wrapper embedding | Estendere con adapter ONNX specifici | `metrics`, `observations` | adapter correctness, deterministic output | candidati di osservazioni |
| `src/neural_continuity/observations.py` | modify | separare metadati e metriche per run transizione | Differenziare transizioni e controlli | `models`, `datasets` | NaN/inf, missing outputs | observation rows/parquet |
| `src/neural_continuity/cli.py` | modify | orchestrazione transizioni A/B, control plan espliciti | Punto di comando unico e replay coerente | `evidence`, `metrics` | m0 replay parity, boundary synthetic | decisioni, control outcomes |
| `src/neural_continuity/decisions.py` | modify | policy acceptance per detectability/materiality | Distinguere envelope e materialità | `metrics`, `contracts` | interval boundary logic | reason payload |
| `src/neural_continuity/evidence.py` | modify | manifest separati per transizione | Immutabilità e audit | `observations`, `bootstrap` | hash mutation | artifact-manifest |
| `src/neural_continuity/contracts/` | create | contract M1 metric policy esteso | Versionamento regole e non-blocchi | nessuna dipendenza interna forte | load/validate schema | manifest di contratto |
| `src/neural_continuity/adapters/` | create | adapter ONNX/quantization | Isolare conversion e inferenza | `onnx`, `onnxruntime` | export integrity | export artifacts |
| `src/neural_continuity/transitions/` | create | orchestrare A e B come transizioni separate | separare causa export/quantizzazione | `models`, `adapters`, `decisions` | transizione A/B smoke | decision boundary package |
| `experiments/` | modify | configurazioni M1 A/B + boundary controls dedicati | Configurazione esplicita per run | `contracts`, `docs` | run smoke docs | run configs |
| `tests/test_measurements.py` | modify | aggiungere test transizione e replay fail-closed specifici | protezione regressioni M1 | `fixtures`, `cli` | suite M1.0 | golden assertions |
| `tests/fixtures/` | modify | dataset M1 con split e split ids stabili | supporto misurazione separata | `datasets` | fixture validation | fixture hashes |
| `docs/M1_ACTION_PLAN.md` | modify | piano operativo | Artefatto guida vivo per M1 | nessuna | n/a | documentazione |

## 13) Piano dipendenze e split (dichiarazione finale)

### Dependency qualification table

| Item | Available | Missing | Incompatible | Unverified |
|---|---:|---:|---:|---:|
| Python runtime | `3.13.5` |  |  |  |
| PyTorch |  |  | `3.13.5 environment (non installed)` |  |
| ONNX |  | `M1 environment` |  |  |
| ONNX Runtime | `1.22.1` |  |  |  |
| Sentence Transformers |  | `Dipendente da torch` |  |  |
| CUDA EP |  |  | non disponibile senza torch |  |
| Teacher cache |  | `missing` |  |  |
| ONNX quantization API |  |  | assente senza ONNX |  |

### Transition decomposition

1. Transition A
   - Input: modello PyTorch/torch-simile FP32.
   - Output: artifact ONNX FP32.
   - Gate: pass/fail del controllo M0-equivalente.
2. Transition B
   - Input: ONNX FP32.
   - Output: ONNX INT8 quantizzato.
   - Gate: B richiede A passata.

### File-level change map

Vedi sezione 12.

### Dataset and split plan

- calibrazione: `calibration`;
- validazione: `validation`;
- congelamento funzionale: `frozen_critical`;
- conclusione: `final_holdout`.

### Evidence artifact plan

- raw observations parquet;
- metadata manifest;
- decisione per run e per controllo;
- noise envelope;
- replay bundle;
- comparison report;
- artifact manifest con hash SHA-256 canonici.

### Proposed contract additions

- identificatori transizione (`A` e `B`);
- policy di materialità separata per metric family;
- sezione `expected_materiality` per limiti asimmetrici e directional;
- sezione `topology_warning` per limiti non bloccanti;
- regole per comportamento in caso di evidence mancante.

### Test matrix

- adapter correctness;
- tokenizer/parsing parity;
- pooling parity;
- normalization parity;
- deterministic sample ordering;
- ONNX export integrity;
- missing execution provider;
- missing graph input;
- malformed output;
- NaN/Inf handling;
- artifact mutation;
- replay equivalence;
- declared observation missing;
- critical-set regression;
- interval inside / outside / crossing boundary;
- directional beneficial change;
- performance non-blocking unless contract says otherwise;
- smoke full M0 pipeline with M1 configuration.

### Kill conditions

- fallimento transizione A;
- change preprocessing durante export;
- inseparabilità effetti export/quantizzazione;
- contaminazione holdout da calibrazione;
- replay non ricostruibile;
- dipendenza da stato runtime non dichiarato;
- mancanza di evidenza critica non classificata.

### Unresolved concrete decisions

- scelta tra `PyTorch` reale e baseline simulata per primo run in ambienti senza torch;
- provider ONNX prioritario in ambiente CPU-only;
- schema finale di `acceptance contract` (numeri e famiglie);
- soglia hardware per promozione transizione INT8.

### Recommended first implementation slice

`Load the cached real teacher, capture a canonical PyTorch FP32 observation, and prove that the resulting evidence can be replayed.`

## 14) Stato della guida e passaggio a implementazione

- `M1_RESEARCH_GUIDE.md` allineata con terminologia M0 (`negative` vs `material-negative`).
- Nessun codice implementativo introdotto.
- Prossima fase: revisione M1 plan dopo disponibilità dipendenze `torch/onnx/sentence-transformers`.
