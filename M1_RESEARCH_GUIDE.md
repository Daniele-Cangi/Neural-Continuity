# Guida generale M1 — Neural Continuity

## 1) Scopo del progetto

Neural Continuity mira a **misurare e documentare la continuità** di cambi di modello in modo ripetibile e verificabile.

Obiettivo operativo:

- verificare se una transizione di modello (stesso dominio dichiarato, stessa contract di comportamento) preserva proprietà operative entro incertezza misurata;
- produrre una decisione in tre stati:
  - `PASS`
  - `FAIL`
  - `INCONCLUSIVE`.

La piattaforma non dichiara equivalenza neurale assoluta: dichiara continuità entro i limiti del contratto esplicito e dell’evidenza osservata.

## 2) Stato attuale: Milestone M0 (conclusa)

La milestone M0 è l’hardware di fiducia del sistema di misurazione.

- Controlli implementati e validati:
  - `exact_repeat`
  - `negative` (alias testuale di `material-negative` nella documentazione M0)
  - `boundary`
- Decisione aggregate: `measurement_integrity_status = PASS`.
- Replay verificato: riesecuzione ricostruita senza rerun del modello e match su:
  - stato aggregato
  - outcome dei controlli individuali.
- Fail-closed: un controllo dichiarato mancante deve fallire in modo deterministico.
- Artifacts e manifest:
- `artifact-manifest.json` con hash SHA-256 completo
- run bundle, decisione, report e metrica serializzati in modo verificabile (tamper-evident) tramite hash canonici.

## 3) Regole base (invarianti M0 da preservare)

Le seguenti proprietà **non devono essere degradate** da M1:

1. separazione chiara tra outcome del controllo e salute del sistema;
2. `negative` può essere `FAIL` pur mantenendo integrità M0;
3. `boundary` valido resta `INCONCLUSIVE` quando attraversamento significativo;
4. regressioni critiche freeze-set hanno priorità;
5. orientamento metriche esplicito (higher-is-better / lower-is-better / two-sided / informational);
6. miglioramenti non devono essere penalizzati come regressioni;
7. envelope null separato per sorgente di rumore;
8. no null envelope a larghezza zero da evidenza mancante;
9. incertezza candidato esplicitata;
10. il candidato non definisce la propria baseline null;
11. identità source/candidato allineate;
12. replay riproduce stato aggregato e outcome individuali;
13. fail-closed su controllo dichiarato mancante;
14. SHA-256 completo nel manifest.

## 4) Transizione prevista per M1

M1 introduce la prima transizione reale sotto controllo:

1. `PyTorch FP32 -> ONNX FP32`
2. solo dopo, `ONNX FP32 -> ONNX INT8`

Questo evita di confondere effetti di export/runtime con effetti di quantizzazione.

Modello candidato iniziale di riferimento:

- `sentence-transformers/all-MiniLM-L6-v2`

## 5) Criteri di misura per M1

Per ogni transizione valutare (entro il contratto dichiarato):

- **Funzionale**
  - metriche retrieval
  - regressioni su set critico/freeze-set
- **Topologico/Rappresentativo**
  - drift embedding
  - preservazione ranking e neighbourhood
- **Sistemi**
  - latenza
  - throughput
  - memoria/RSS
  - dimensione artifact e risorse
- **Metrologia**
  - incertezze bootstrap/query
  - ripetibilità
  - sufficienza campione
- **Immutabilità prova**
  - manifest, hash, seeds, config, comando, versioni.
- **Replay**
  - verificare decisione e outcome da bundle registrato.

## 6) Rischi tecnici principali da gestire

1. attribuzione errata tra regressione da export e regressione da quantization;
2. non-determinismo ONNX/runtime che altera riproducibilità;
3. confusione tra significatività statistica e materialità operativa;
4. inadeguatezza campionamento fixture/benchmarks rispetto alla variabilità reale;
5. incoerenze nella serializzazione artifact (seed, versioni runtime, op-set, device).

## 7) Moduli esistenti da riutilizzare

- `src/neural_continuity/models.py`
- `src/neural_continuity/datasets.py`
- `src/neural_continuity/observations.py`
- `src/neural_continuity/noise.py` (legacy / non-authoritative: usa `abs(delta)` e non rispecchia direttamente gli invarianti M0 attuali)
- `src/neural_continuity/metrics.py`
- `src/neural_continuity/evidence.py`
- `src/neural_continuity/decisions.py`
- `src/neural_continuity/perturbations.py`
- `src/neural_continuity/cli.py`
- `tests/test_measurements.py` e `tests/fixtures`

## 8) Componenti nuove previste per M1

- Adapter e runner ONNX (export/inferenza) con manifest esplicito (versione runtime, op-set, provider, device);
- Layer specifico di quantizzazione (export INT8) e tracciabilità del processo di quantizzazione;
- Config schema per confronto “export-only” vs “quant-only”;
- Pipeline policy per soglie di materialità operative (separate da soglie statistiche).

## 9) Cosa NON deve entrare in M1

- integrazione con Spectra come oracolo primario;
- adaptive routing completo;
- distillazione (M2);
- regole speculative non contrattualizzate;
- cambi di semantica decisionale/format o dei vincoli M0.

## 10) Processo consigliato (operativo)

1. Fissare contratto del candidato e dominio dichiarato.
2. Implementare suite M1 separata in configurazione esperimento.
3. Eseguire baseline di export (`PyTorch -> ONNX`) e bloccare eventuali deviazioni tecniche.
4. Eseguire step quantization (`ONNX FP32 -> ONNX INT8`) con controllo incrociato.
5. Raccogliere artefatti completi, manifest e hash.
6. Verificare:
   - gate generali (pytest/ruff/black/mypy/compileall),
   - replay,
   - stato control + fail-closed.
7. Rilasciare solo con risultati coerenti con gli invariants M0 e senza promesse oltre scope.

## 11) Stato repository e contesto operativo

- Repo locale: `C:\dev\neural-continuity`
- M0 è stato integrato in `main` tramite PR #1.
- Merge commit M0 + pianificazione iniziale M1:
  `3617d5e9d402c886e70b868cfa03af59229e84f0`
- Branch originaria di pianificazione M1:
  `codex/m1-verified-quantization`
- Branch correttiva documentale:
  `codex/m1-plan-freeze-fix`
- Base M0 verificata:
  `3418b7d612adfd2bd81045dd4843cf5904192b5f`
- Nessuna implementazione M1 è ancora iniziata.
