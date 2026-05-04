---
sidebar_position: 3
title: "Preprocessing dati clinici"
description: "Capitolo: Preprocessing dati clinici. Progetto Hospital Readmission 30d."
---

# Preprocessing dei dati clinici

## Indice

---

## 1. Il marker `?` non e' un caso particolare

Il dataset Diabetes-130 codifica i missing come **`?`** invece di NaN. Senza una conversione esplicita, pandas legge i `?` come stringa: la colonna diventa `object` invece di `float`, le statistiche descrittive sono inutili, e i modelli sklearn falliscono al fit.

Soluzione (in `data.load_raw`):

```python
df = pd.read_csv(path, na_values=["?"])
```

`na_values` accetta una lista di stringhe da convertire in NaN. Una linea, ma e' la **prima decisione di pulizia obbligatoria**.

## 2. Le tre colonne da rimuovere a priori

| Colonna | Problema | Decisione |
|---|---|---|
| `weight` | ~97% missing (i clinici non lo registravano sistematicamente) | **Drop** — l'imputazione di una colonna 97% missing e' rumore |
| `examide` | Un solo valore ('No') | **Drop** — zero variance, nessuna informazione |
| `citoglipton` | Un solo valore ('No') | **Drop** — zero variance |
| `payer_code` | ~40% missing + scarsa rilevanza clinica | **Drop** — informazione assicurativa, non clinica |

Documentate in `config.COLUMNS_TO_DROP`. Una scelta esplicita e motivata batte un automatismo "rimuovi colonne con >X% missing".

## 3. Drop dei pazienti "Expired/Hospice"

I codici `discharge_disposition_id` corrispondenti a decesso o ricovero in hospice (11, 13, 14, 19, 20, 21) identificano pazienti che **non possono essere riammessi**: sono morti o in cure palliative. Lasciarli nel training distorce la stima del modello, perche' il loro `readmitted = NO` non e' un esito di qualita' assistenziale ma una **conseguenza biologica**.

Effetto su Diabetes-130: rimuove ~1.600 record (~1.5% del dataset). Pulizia pre-split, prima dell'estrazione di `groups`.

## 4. Strategia per i missing residui

Dopo aver droppato `weight`, `payer_code`, etc., restano missing in:

| Colonna | % missing | Strategia |
|---|---|---|
| `race` | ~2% | Imputazione 'Unknown' (categoria nuova) |
| `medical_specialty` | ~49% | Imputazione 'Unknown' (gia' implicita: `OneHotEncoder`) |
| `diag_1`, `diag_2`, `diag_3` | &lt;1% | Mappa a categoria 'Missing' nel ICD-9 grouping |

**Perche' imputare con una categoria 'Unknown' invece di droppare le righe**:

- Droppare il 49% del dataset (per il `medical_specialty`) buttando via informazione utile.
- Il missing **e' informativo**: medici di base senza specialita' formale documentata sono spesso quelli di reti rurali, che hanno pattern di readmission diversi. Tenere il missing come categoria fa imparare al modello l'eventuale segnale.

Implementato dentro il `ColumnTransformer`:

```python
SimpleImputer(strategy="constant", fill_value="Unknown")
```

## 5. ICD-9 grouping: dalla cardinalita' 700+ a 9 macro-categorie

Le tre colonne `diag_*` hanno >700 valori unici ciascuna. **Mai** un OneHotEncoder grezzo su questo: produce 2000+ colonne sparse, di cui la maggior parte hanno meno di 10 occorrenze nel training.

Soluzione: macro-grouping clinico (Strack 2014, Tabella 2), implementato in `src/readmit_pipeline/icd9.py`:

```python
from readmit_pipeline.icd9 import map_icd9_to_category

map_icd9_to_category("250.83")  # 'Diabetes'
map_icd9_to_category("428")     # 'Circulatory'
map_icd9_to_category("V58.67")  # 'Other' (codici V/E)
map_icd9_to_category(None)      # 'Missing'
```

Risultato: 10 categorie totali (9 cliniche + 'Missing'), gestibili da OHE senza esplosione di dimensionalita'.

## 6. Group-aware split: il piu' importante

Punto critico. Lo stesso `patient_nbr` puo' comparire piu' volte nel dataset. Senza precauzioni, lo split casuale **disperde** i ricoveri di uno stesso paziente fra train e test.

Conseguenza pratica: il modello "memorizza" pattern del paziente specifico (combinazione eta'-razza-medico) e li riapplica in test. Le metriche misurano la **memorizzazione**, non la generalizzazione a pazienti nuovi.

Soluzione (in `src/readmit_pipeline/splits.py`):

```python
from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
(train_idx, test_idx), = splitter.split(X, y, groups=patient_nbr)
```

E per la cross-validation `StratifiedGroupKFold` (sklearn>=1.0): bilancia la classe positiva *e* mantiene i gruppi disgiunti.

Differenza misurata sperimentalmente:

- Split casuale ingenuo: AUC-ROC test ~0.70.
- Group-aware split: AUC-ROC test ~0.64.

I 6 punti di gap sono il **leakage** della prima formulazione. Ed e' rilevante: in produzione il modello vede solo pazienti **mai visti prima**, quindi 0.64 e' la stima onesta delle performance.

## 7. Encoding dei farmaci antidiabetici

Le 23 colonne dei farmaci (`metformin`, `insulin`, `glyburide`, ...) hanno valori `{No, Steady, Up, Down}`. Tre opzioni di encoding:

| Strategia | Pro | Contro |
|---|---|---|
| OneHot (ogni colonna -> 4 dummy) | Modello flessibile | Esplosione: 23 * 3 = 69 colonne |
| Ordinal (No=0, Steady=1, Up=2, Down=2) | Compatto | Imposta ordine arbitrario |
| **Aggregazione** (`n_med_prescribed`, `n_med_changed`) | Interpretabile, parsimoniosa | Perde dettaglio per-farmaco |

Scelta in `features.ReadmissionFeatureEngineer`: **aggregazione** per le pipeline finali, ma **tenere anche le colonne grezze** se il modello e' un ensemble (lascia che decida lui).

```python
out["n_med_prescribed"] = (df[meds] != "No").sum(axis=1)
out["n_med_changed"]    = df[meds].isin(["Up", "Down"]).sum(axis=1)
```

## 8. Imputer + Scaler: posizione esatta nella pipeline

```
[ feature_engineer ]  ->  [ ColumnTransformer ]  ->  [ Model ]
                          |    num branch:           |
                          |       imputer (median)   |
                          |       StandardScaler     |
                          |    cat branch:           |
                          |       imputer ('Unknown')|
                          |       OneHotEncoder      |
                          |       (min_frequency=10) |
```

**Perche' lo scaler PRIMA del modello e DOPO il preprocessing**:

- Le feature numeriche ammettono mediana + scaling diretto.
- Le feature categoriche post-OHE sono gia' 0/1: scalarle aggiungerebbe rumore. Quindi `StandardScaler` sta dentro il branch numerico, non a valle del `ColumnTransformer`.

**Perche' `min_frequency=10` su OneHotEncoder**:

Il OneHotEncoder di sklearn>=1.1 supporta `min_frequency`: categorie con &lt;10 occorrenze nel training set vengono aggregate in un'unica categoria `infrequent_sklearn`. Effetto:

- Riduce dimensionalita' (categorie ultra-rare scartate).
- Robusto al **drift**: se in produzione arriva una categoria nuova, finisce in `infrequent_sklearn` invece di crashare.

## 9. Cosa NON fare

- **Imputare la mediana globale** dell'intero dataset prima dello split. Genera leakage.
- **Standardizzare** prima dello split. Stesso problema.
- **OneHotEncoder con `handle_unknown='error'`**. In produzione una categoria nuova fa crashare il sistema. Usa `'ignore'` o aggrega via `min_frequency`.
- **Scalare le feature 0/1 (post-OHE)**. Inutile e introduce numeri non interpretabili.
- **Eliminare i pazienti con missing su `race`**. Il missing e' informativo (vedi paragrafo 4).

## 10. Riferimenti

- **Strack, B. et al.** (2014). Tabella 2: classificazione ICD-9.
- **Pedregosa et al.** (2011). *Scikit-learn: Machine Learning in Python*. JMLR — `ColumnTransformer`, `OneHotEncoder`, `SimpleImputer`.
- **scikit-learn user guide** sezione [Group-based cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#group-cv).