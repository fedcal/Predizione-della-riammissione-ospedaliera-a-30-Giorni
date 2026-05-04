---
sidebar_position: 2
title: "EDA clinica"
description: "Capitolo: EDA clinica. Progetto Hospital Readmission 30d."
---

# EDA clinica

## Indice

---

## 1. Le tre cose da sapere prima di guardare un grafico

Prima di lanciare un singolo plot, fissa tre numeri di riferimento sul dataset Diabetes 130-US Hospitals:

| Quantita' | Valore | Implicazione |
|---|---|---|
| Numero di righe | 101.766 | Sufficienti per LogReg/RF; XGBoost non strettamente necessario |
| Numero di pazienti unici (`patient_nbr`) | ~71.500 | ~30% dei pazienti ha >=2 ricoveri |
| Tasso di `readmitted = '<30'` | ~11.16% | Classe positiva minoritaria; AUC-PR > AUC-ROC come metrica |

Senza questi tre numeri, ogni grafico e' privo di scala interpretativa.

## 2. Distribuzione del target (3 classi vs binaria)

```python
df['readmitted'].value_counts(normalize=True).round(4) * 100
# NO     53.91 %
# >30    34.93 %
# <30    11.16 %
```

Cosa leggere:

- La classe `<30` (positiva binaria) e' al **~11%**. Su 100 pazienti, ne riammettiamo 11 entro 30 giorni.
- La classe `>30` (~35%) e' la piu' insidiosa: questi pazienti sono stati riammessi, ma fuori dalla finestra HRRP. La binarizzazione li tratta come "negativi" ma clinicamente non lo sono.
- Un classificatore costante "predici sempre 0" raggiunge **89% accuracy** senza imparare nulla. Per questo l'accuracy e' inutile come metrica primaria.

## 3. Distribuzione demografica e disparita' nel target

E' lo step diagnostico chiave per l'audit di fairness in fase di valutazione. Calcola il **tasso di readmission per sottogruppo**:

```python
df.groupby('race')['readmitted_30d'].agg(['mean', 'count'])
```

Pattern tipici osservati su questo dataset (Strack 2014, Tabella 4):

| Race | Tasso readmit 30d | n |
|---|---|---|
| Caucasian | ~10.9% | ~76.000 |
| AfricanAmerican | ~11.7% | ~19.000 |
| Hispanic | ~9.8% | ~2.000 |
| Asian | ~8.6% | ~640 |
| Other | ~9.1% | ~1.500 |
| Missing (NaN) | ~10.2% | ~2.300 |

La differenza fra Caucasian e AfricanAmerican (10.9% vs 11.7%) e' **statisticamente significativa** dato $n$ elevato, ma operativamente piccola (~0.8 percentage point). E' il **base rate** che dovremo confrontare con il **predicted rate** del modello in fase di fairness audit.

Sull'asse `age` la differenza e' molto piu' marcata: pazienti $\geq$70 anni hanno tassi quasi doppi rispetto agli under-50. Conseguenza: un modello che usa `age` come feature predittiva di per se' separera' bene, ma generera' alert sproporzionati sugli anziani.

## 4. Pattern di utilizzo sanitario pregresso

Le tre feature `number_outpatient`, `number_emergency`, `number_inpatient` (visite ambulatoriali, accessi in PS, ricoveri nell'anno precedente) sono **fra i predittori piu' forti** in letteratura (Kansagara et al. 2011).

Distribuzione tipica:

- `number_inpatient`: zero per il ~70% dei pazienti, ma con coda lunga fino a 21+.
- `number_emergency`: zero per il ~80%.
- Pazienti con `number_inpatient >= 3` hanno tasso di readmission >25% (vs ~9% per chi ha 0).

Implicazione per il feature engineering:

```python
prior_healthcare_use = (
    number_outpatient + 2 * number_emergency + 3 * number_inpatient
)
```

Pesi crescenti perche' un ricovero pregresso e' clinicamente piu' grave di un accesso al PS, che e' a sua volta piu' grave di una visita ambulatoriale.

## 5. Diagnosi primarie: la cardinalita' di ICD-9

Le tre colonne `diag_1`, `diag_2`, `diag_3` contengono codici ICD-9 (formato `XXX.YY`). Caratteristiche:

- ~700 valori unici per `diag_1`.
- Code "V" e "E" (status / external causes) presenti ma rari.
- I codici della famiglia 250.xx sono il **diabete con varianti** (250.83, 250.6, ecc.).

**Mai usare un OneHotEncoder grezzo su `diag_*`**: produrrebbe ~700 colonne con la maggior parte sparse, esplodendo dimensionalita' e rumore.

Soluzione: **macro-grouping** in 9 categorie (Strack 2014, Tabella 2):

| Range ICD-9 | Categoria |
|---|---|
| 390-459, 785 | Circulatory |
| 460-519, 786 | Respiratory |
| 520-579, 787 | Digestive |
| 250.xx | Diabetes |
| 800-999 | Injury / Poisoning |
| 710-739 | Musculoskeletal |
| 580-629, 788 | Genitourinary |
| 140-239 | Neoplasms |
| altro | Other |

Implementato in `src/readmit_pipeline/icd9.py`.

## 6. HbA1c: l'ipotesi centrale di Strack 2014

Il paper originale ha mostrato che la **misurazione dell'HbA1c durante il ricovero** (a prescindere dal valore) e' associata a un tasso di readmission ridotto. Razionale clinico: misurare l'HbA1c indica un'attenzione strutturata al controllo glicemico, che si traduce in piano di dimissione piu' robusto.

Distribuzione di `A1Cresult`:

| Valore | % | Interpretazione |
|---|---|---|
| `None` | ~83% | Test non eseguito |
| `>8` | ~8% | HbA1c elevata |
| `Norm` | ~5% | HbA1c normale |
| `>7` | ~4% | HbA1c moderatamente elevata |

Feature derivata utile: `A1C_measured = (A1Cresult != 'None')` come flag binario, che cattura l'**effetto Strack** indipendentemente dal valore esatto.

## 7. Pazienti con encounter multipli

```python
n_per_patient = df.groupby('patient_nbr').size()
print((n_per_patient > 1).sum())   # ~22.000 pazienti
print(n_per_patient.max())         # ~40 ricoveri di un singolo paziente!
```

Implicazione metodologica fondamentale: **lo split casuale e' sbagliato**. Senza un group-aware split, ricoveri dello stesso paziente finirebbero sia in train sia in test. Il modello "riconosce" il paziente attraverso pattern individuali (eta', razza, prior_inpatient) e ottiene metriche over-ottimistiche che non si replicano in produzione.

Soluzione: `GroupShuffleSplit` per l'holdout iniziale + `StratifiedGroupKFold` per la cross-validation, entrambi con `groups=patient_nbr`. Vedi `src/readmit_pipeline/splits.py`.

## 8. Distribuzione `time_in_hospital` e numero di farmaci

Due feature continue informative:

- `time_in_hospital` $\in$ [1, 14]: distribuzione approssimativamente esponenziale, mediana ~3 giorni. Pazienti con permanenza >7 giorni hanno tassi di readmission ~2x rispetto a permanenze brevi (gravita' clinica maggiore al baseline).
- `num_medications` $\in$ [1, 81]: distribuzione skewed-right. Pazienti con >20 farmaci sono i **multimorbidi**: alta complessita' clinica, alto rischio.

Feature derivata `n_med_changed` (numero di farmaci antidiabetici con dosaggio modificato) e' un proxy della **fragilita' farmacologica**: un dosaggio modificato significa che il clinico sta cercando un equilibrio non ancora raggiunto, segnale di paziente complesso.

## 9. Cosa NON vedere nei dati

Il dataset Diabetes-130 manca di:

- **Condizioni socio-economiche** (income, education, housing).
- **Supporto familiare** (vivi solo? caregiver disponibile?).
- **Aderenza terapeutica** (il paziente ha effettivamente preso i farmaci?).
- **Eventi acuti post-dimissione** (caduta, infezione, complicanza).
- **Eventi ambulatoriali post-dimissione** (visite di follow-up effettuate o no).

Questi sono **predittori potenzialmente molto forti** della readmission. La loro assenza e' la ragione principale per cui le AUC-ROC tipiche su questo dataset sono moderate (0.60-0.67): non e' un fallimento del modello, e' un limite del dataset.

## 10. Riferimenti

- **Strack, B. et al.** (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates*. BioMed Research International, 2014:781670 — fonte della classificazione ICD-9 e dei base rate per sottogruppo.
- **Kansagara, D. et al.** (2011). *Risk prediction models for hospital readmission: a systematic review*. JAMA, 306(15), 1688-1698.
- **CMS HRRP overview**: https://www.cms.gov/medicare/quality/value-based-programs/readmissions-reduction-program