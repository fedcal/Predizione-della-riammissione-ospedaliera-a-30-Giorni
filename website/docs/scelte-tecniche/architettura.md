---
sidebar_position: 1
title: "Architettura del progetto"
description: Capitolo: Architettura del progetto. Progetto Hospital Readmission 30d.
---

# Architettura del progetto

## Indice

---

## 1. Layout

```
hospital-readmission-30d/
|-- README.md                    Quick-start, risultati attesi, badge.
|-- LICENSE                      MIT.
|-- pyproject.toml               Build config + dipendenze + entry points CLI.
|-- requirements.txt             Lock approssimativo per chi non usa pyproject.
|-- mkdocs.yml                   Configurazione del sito documentazione.
|
|-- src/readmit_pipeline/        Codice sorgente (pip install -e .)
|   |-- __init__.py
|   |-- config.py                Path, costanti, iperparametri di default (frozen).
|   |-- data.py                  load_raw + cleaning (drop colonne, drop expired).
|   |-- icd9.py                  Mapping ICD-9 -> 9 macro-categorie (Strack 2014).
|   |-- features.py              ReadmissionFeatureEngineer (BaseEstimator+TransformerMixin).
|   |-- preprocessing.py         build_preprocessor -> ColumnTransformer.
|   |-- splits.py                GroupShuffleSplit + StratifiedGroupKFold (group-aware).
|   |-- models.py                LogReg + RandomForest [+ XGBoost opzionale].
|   |-- tuning.py                GridSearchCV con scoring=AUC-PR.
|   |-- threshold.py             Ottimizzazione soglia su matrice costi asimmetrica.
|   |-- evaluation.py            Metriche binary classification + plot ROC/PR.
|   |-- fairness.py              Fairlearn MetricFrame + DP/EO gaps.
|   |-- interpretability.py      Coefficienti LR, feature importance, SHAP opzionale.
|   |-- inference.py             predict_readmission_risk() + CLI readmit-predict.
|   `-- pipeline.py              Orchestrator + CLI readmit-train.
|
|-- notebooks/                   Documentazione esecutiva (rigenerata da script).
|   |-- 01_eda_demographics_target.ipynb
|   |-- 02_preprocessing_icd9_grouping.ipynb
|   |-- 03_models_logreg_vs_ensemble.ipynb
|   |-- 04_fairness_audit.ipynb
|   `-- 05_interpretability_and_errors.ipynb
|
|-- data/                        Dataset (gitignored)
|   |-- raw/                     diabetic_data.csv + IDS_mapping.csv (UCI 296)
|   |-- processed/               Output preprocessing (cache)
|   `-- external/                Eventuali dataset di arricchimento
|
|-- reports/
|   |-- figures/                 PNG (ROC, PR, confusion matrix, feature importance)
|   |-- models/                  Modelli serializzati .joblib (gitignored)
|   |-- cv_summary.csv           Tabella CV scores (run output)
|   |-- holdout_metrics.json     Metriche holdout (run output)
|   |-- fairness_summary.csv     Gap DP/EO per attributo (run output)
|   `-- fairness_report.csv      Metriche per sottogruppo (run output)
|
|-- scripts/
|   |-- build_notebooks.py       Genera i 5 notebook da sorgente Python.
|   `-- run_full.sh              Pipeline completa (training + notebook + smoke test).
|
|-- tests/
|   |-- test_data.py             Smoke test cleaning + binarizzazione target.
|   |-- test_icd9.py             Test del mapping ICD-9 (Strack 2014, Tabella 2).
|   |-- test_splits.py           Test group-aware split: zero overlap.
|   `-- test_fairness.py         Test calcoli Fairlearn su mini-DataFrame.
|
|-- docs/
|   |-- index.md                 Hero + quickstart + risultati.
|   |-- teoria/                  7 file Markdown didattici.
|   |-- scelte_tecniche/         Documenti di design (architettura, modelli).
|   `-- stylesheets/extra.css    Custom CSS (mobile-first, hero banner).
|
`-- venv/                        Virtual environment (gitignored).
```

## 2. Principi di design

### 2.1 Separazione `src/` vs `notebooks/`

Il **codice riutilizzabile** vive in `src/readmit_pipeline/`. I **notebook** sono presentazione: importano funzioni da `src/`, mostrano output, raccontano la storia.

Vantaggi:
- Modifiche al codice riflesse automaticamente nei notebook (riavvia il kernel).
- Niente duplicazione: la logica esiste una sola volta.
- Test unitari possibili sul codice in `src/` (i notebook non sono test-friendly).

### 2.2 Notebook generati da script

I 5 notebook sono prodotti da `scripts/build_notebooks.py`. Per modificarli si edita lo script e si rilancia. Vantaggi:

- **Diff Git puliti**: niente cambi spuri di metadata, output cells, kernel hash.
- **Riproducibilita'**: chiunque rigenera notebook identici.
- **Sorgente in formato testuale**: il `.py` e' grep-able come ogni Python file.

### 2.3 Pipeline come oggetto sklearn

Tutte le trasformazioni che dipendono da statistiche del dataset stanno dentro un `sklearn.pipeline.Pipeline`. Vantaggi:

- **No leakage** (vedi `docs/teoria/03_preprocessing_dati_clinici.md`).
- **Persistenza**: un singolo `joblib.dump(pipeline)` salva preprocessing + modello.
- **API uniforme**: `fit/predict/predict_proba` standard sklearn.

La pipeline finale ha 3 step:

```
[ feature_engineer ]  ->  [ preprocessor ]  ->  [ model ]
ReadmissionFeatureEngineer  ColumnTransformer    LogReg | RandomForest
- n_med_prescribed          - num branch:
- prior_healthcare_use         imputer median
- diag_*_cat                   StandardScaler
- A1C_measured              - cat branch:
                               imputer 'Unknown'
                               OneHotEncoder
                               (min_frequency=10)
```

### 2.4 Configurazione centralizzata

Tutti i path, le costanti, le grid di iperparametri vivono in `config.py`. Niente magic number sparsi nel codice.

`PipelineConfig` e' un `@dataclass(frozen=True)`: ogni esperimento crea un proprio config immutabile, nessuna mutazione accidentale (rule "immutability" del coding style).

### 2.5 Group-aware split: contratto invariante

Lo stesso `patient_nbr` non deve mai apparire sia in train sia in test. Implementato in `splits.group_aware_train_test_split` con `GroupShuffleSplit`, e validato da `assert_no_group_leakage()` (chiamato esplicitamente nella pipeline e testato in `test_splits.py`).

Per la cross-validation: `StratifiedGroupKFold` (sklearn>=1.0) — group-aware *e* stratificato sul target.

### 2.6 Fairness audit come step della pipeline

Il fairness audit non e' un'aggiunta opzionale ma uno **step esplicito** della pipeline (`pipeline.run_full_pipeline`). Output:

- `reports/fairness_summary.csv`: gap demographic_parity ed equalized_odds per attributo.
- `reports/fairness_report.csv`: metriche per sottogruppo (selection_rate, TPR, FPR, precision).

Questi file sono il deliverable principale della Fase 6 del project work.

### 2.7 Soglia decisionale: parametro di prima classe

La soglia (default 0.5) e' un parametro della `PipelineConfig`. La pipeline cerca anche la **soglia ottima** sotto matrice costi asimmetrica (`threshold.find_optimal_threshold`). Output:

- `holdout_metrics.json` contiene metriche sia alla soglia 0.5 sia alla soglia ottima.

## 3. Punti di estensione

### 3.1 Aggiungere un nuovo modello

1. Implementare la pipeline in `models.py` (es. `lightgbm_pipeline`).
2. Aggiungerla al dizionario in `get_all_pipelines`.
3. Definire la grid in `config.py` (`LIGHTGBM_PARAM_GRID`).
4. Aggiungere il caso in `tuning.tune_all_models`.

Il resto del codice (evaluation, fairness, inference) funziona senza modifiche grazie al polimorfismo sklearn.

### 3.2 Cambiare il rapporto costi FN/FP

In `config.py`:

```python
COST_FN_OVER_FP: float = 5.0   # default conservativo
```

Range tipico [5, 20]. Con `COST_FN_OVER_FP = 20`, la soglia ottima si abbassa drasticamente (~0.15-0.25), aumentando recall a costo di precision (piu' alert).

### 3.3 Aggiungere SMOTE come strategia di sbilanciamento

1. Aggiungere `imbalanced-learn>=0.12` come dipendenza opzionale in `pyproject.toml`.
2. Sostituire `sklearn.pipeline.Pipeline` con `imblearn.pipeline.Pipeline` in `models.py`.
3. Inserire `SMOTE(random_state=42)` come step prima del `model`.
4. Disabilitare `class_weight='balanced'` (i due meccanismi si combinano male).

### 3.4 Mitigazione di fairness post-processing

Per applicare `fairlearn.postprocessing.ThresholdOptimizer`:

```python
from fairlearn.postprocessing import ThresholdOptimizer
mitigator = ThresholdOptimizer(
    estimator=trained_model,
    constraints="equalized_odds",
    prefit=True,
)
mitigator.fit(X_train, y_train, sensitive_features=race_train)
y_pred = mitigator.predict(X_test, sensitive_features=race_test)
```

Lasciato come estensione esplorabile: in produzione bisogna chiedere alla governance se l'uso esplicito di `race` al momento della decisione e' accettabile.

## 4. Trade-off espliciti

| Scelta | Pro | Contro |
|---|---|---|
| `class_weight='balanced'` come default | Semplice, robusto, no overfit | Meno potente di SMOTE su feature continue |
| `min_frequency=10` su OneHotEncoder | Riduce dimensionalita', robusto al drift | Categorie rare aggregate possono perdere segnale |
| `StratifiedGroupKFold` | Group-aware + bilanciato | Piu' lento di KFold semplice |
| `average_precision` come scoring | Robusto su sbilanciamento | Meno familiare di accuracy/ROC |
| LogReg come baseline | Interpretabile, calibrato | Meno potente su interazioni non lineari |
| RandomForest come ensemble | Cattura interazioni | feature_importance impurity-based ha bias |
| Fairlearn come libreria | Standard di settore (Microsoft) | API in evoluzione, breaking changes occasionali |
| Soglia ottima alla fine | Ricalibrabile senza retraining | Ottimizzata su test (potenziale piccolo overfit) |

## 5. Testing strategy

Smoke test inclusi in `tests/`, eseguibili con `pytest tests/`:

| Test | Cosa verifica |
|---|---|
| `test_data.py` | Binarizzazione target, drop colonne zero-variance, drop expired |
| `test_icd9.py` | Mapping ICD-9 corretto su tutte le 9 macro-categorie + V/E/missing |
| `test_splits.py` | Zero overlap di pazienti fra train/test e fra fold CV |
| `test_fairness.py` | Calcoli Fairlearn su mini-DataFrame (skip se fairlearn non installato) |

Esecuzione veloce: `pytest tests/ -q`.

Per produzione si dovrebbero aggiungere:
- Property-based test su `ReadmissionFeatureEngineer` (proprieta': `n_med_prescribed >= 0` sempre).
- Integration test sull'intera pipeline con seed fisso (RMSE entro tolleranza).
- Drift detection test (KS-test fra training e test set).

## 6. CLI: due entry point

| Comando | Funzione |
|---|---|
| `readmit-train [--quick] [--include-xgb]` | Esegue tutta la pipeline; salva modello + report |
| `readmit-predict [--input record.json] [--threshold 0.3]` | Inferenza singola da JSON; output JSON |

Implementati in `pipeline.main_train` e `inference.main_predict`. Registrati come `[project.scripts]` in `pyproject.toml`.