---
sidebar_position: 2
title: "Prima esplorazione (EDA)"
description: "Hands-on con il notebook 01: distribuzione del target, demografia, missing values codificati come '?', pazienti con encounter multipli. Cosa cercare e cosa annotare."
---

# Prima esplorazione (EDA)

:::tip Obiettivo della lezione
Aprire i dati per la **prima volta** e sviluppare *l'istinto* di guardarli prima di modellarli. Non c'è modello che salvi un EDA fatto male.
:::

## Notebook di riferimento

`notebooks/01_eda_demographics_target.ipynb`

Aprilo con:

```bash
jupyter lab notebooks/01_eda_demographics_target.ipynb
```

## Step 1 — Caricamento e prima occhiata

```python
import pandas as pd

df = pd.read_csv("../data/raw/diabetic_data.csv")
print(df.shape)
df.head()
```

Domande da farti **prima** di leggere la cella successiva:

- Quante righe? (~101.766)
- Quante colonne? (~50)
- Quali tipi di dato? (`df.dtypes`)
- Quali colonne hanno valori `?` invece di `NaN`?

```python
# Conta i '?' per colonna (sono i missing "veri")
question_counts = (df == "?").sum()
question_counts[question_counts > 0].sort_values(ascending=False)
```

Cosa dovresti scoprire:

| Colonna | % `?` |
|---|---|
| `weight` | ~97% |
| `medical_specialty` | ~49% |
| `payer_code` | ~40% |
| `race` | ~2% |

:::warning Insight #1
`weight` è quasi inutilizzabile (97% mancante). Va **eliminato**, non imputato. Imputare il 97% dei valori significa inventare un dataset, non analizzarlo.
:::

## Step 2 — Distribuzione del target

```python
df["readmitted"].value_counts(normalize=True)
```

Atteso:

```
NO     0.539
>30    0.349
<30    0.111   ← la classe positiva
```

Crea la versione binarizzata:

```python
df["readmitted_30d"] = (df["readmitted"] == "<30").astype(int)
df["readmitted_30d"].mean()  # ~0.11
```

:::tip Insight #2 — sbilanciamento
**Solo l'11%** dei ricoveri risulta in readmission a 30 gg. Tieni a mente questo numero: è la **baseline naturale** della classe positiva, e diventerà il riferimento per le metriche (un classificatore casuale ha AUC-PR ≈ 0.11).
:::

## Step 3 — Demografia e target

Guarda la readmission rate per sottogruppo:

```python
df.groupby("race")["readmitted_30d"].mean().sort_values(ascending=False)
df.groupby("age")["readmitted_30d"].mean().sort_values(ascending=False)
df.groupby("gender")["readmitted_30d"].mean()
```

Domande critiche:

- Il tasso varia significativamente fra gruppi? (Spoiler: sì, soprattutto per età.)
- C'è un gruppo molto rappresentato (Caucasian ~75%)?
- C'è un gruppo poco rappresentato? (Es. `Asian` < 1%): attenzione alla **varianza** nelle stime di fairness.

:::tip Insight #3 — fairness pre-modello
Le **disparità di tasso esistono già nei dati**. Il modello non le crea da zero, le *propaga*. Questa è la motivazione per il fairness audit: misurare se il modello *amplifica* le disparità esistenti.
:::

## Step 4 — Encounter multipli

```python
encounters_per_patient = df.groupby("patient_nbr").size()
encounters_per_patient.describe()

# Quanti pazienti hanno più di 1 ricovero?
(encounters_per_patient > 1).sum()
# Quanti record appartengono a pazienti con >1 ricovero?
df.duplicated(subset="patient_nbr", keep=False).sum() / len(df)
```

Atteso: **~30% dei record appartiene a pazienti con encounter multipli**.

:::warning Insight #4 — group leakage
Se fai un `train_test_split` casuale, ricoveri **dello stesso paziente** finiranno in train e in test. Il modello vedrà "pattern del Sig. Rossi" in training e li ritroverà nel test → metriche gonfiate. Soluzione: **group-aware split** su `patient_nbr` (vedi *Teoria → 04 Modelli*).
:::

## Step 5 — Colonne a varianza zero

```python
for col in df.columns:
    if df[col].nunique() == 1:
        print(col, "→", df[col].iloc[0])
```

Trovi: `examide`, `citoglipton` — un unico valore per ogni record. **Da eliminare** prima della modellazione.

## Step 6 — Diagnosi (ICD-9)

```python
df["diag_1"].nunique()       # ~700
df["diag_1"].value_counts().head(20)
```

Oltre 700 codici diversi nella diagnosi primaria. Non puoi one-hot encodare 700 colonne: vedrai nel capitolo *Preprocessing* come **raggruppare** in macro-categorie cliniche.

## Step 7 — Casi da escludere

```python
# Dimissioni "Expired" (deceduti) e "Hospice"
discharge_ids_to_drop = [11, 13, 14, 19, 20, 21]  # mapping da IDS_mapping.csv
df_clean = df[~df["discharge_disposition_id"].isin(discharge_ids_to_drop)]
print(f"Rimossi {len(df) - len(df_clean)} record")
```

Perché? Un paziente deceduto **non può essere riammesso**, quindi la sua etichetta `readmitted=NO` è triviale e non informativa.

## Esercizi guidati

**E1.** Calcola la readmission rate per fascia d'età. Quale fascia ha il rischio più alto? È coerente con quello che ti aspettavi clinicamente?

**E2.** Calcola la correlazione fra `number_inpatient` (ricoveri nell'anno precedente) e `readmitted_30d`. Cosa ti dice?

**E3.** Quanti record vengono **rimossi** se elimini contemporaneamente i deceduti, gli hospice e le righe con `gender == "Unknown/Invalid"`? Esprimi il numero in valore assoluto e in percentuale.

**E4.** Crea una visualizzazione (heatmap o tabella pivot) che mostri la readmission rate per **race × age**. Cosa noti?

## Cose da annotare per il report finale

A fine sessione, scrivi mezza pagina con:

1. **Forma del dataset** (righe, colonne, periodo coperto).
2. **Prevalenza della classe positiva** (~11%).
3. **Top-5 colonne con più missing** (e la decisione: eliminare / imputare / mantenere come `Unknown`).
4. **% di pazienti con encounter multipli** (e la decisione di splittaggio: group-aware).
5. **Colonne a varianza zero** identificate.
6. **Categorie da escludere** (dimissioni terminali).
7. **Una osservazione clinica** che ti sorprende (es. "i pazienti `[80-90)` hanno readmission rate doppia rispetto a `[40-50)`").

Queste 7 voci diventeranno la sezione **"Esplorazione dati"** del tuo report.

## Prossimo passo

[**Primo modello**](./03-primo-modello.md): trasforma le scelte di EDA in una pipeline che addestra LogReg e Random Forest.
