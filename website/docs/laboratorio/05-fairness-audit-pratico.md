---
sidebar_position: 5
title: "Fairness audit pratico"
description: "Hands-on con Fairlearn: calcolo delle metriche per sottogruppo demografico (race, age), differenze fra demographic parity / equalized odds / predictive parity, lettura di fairness_summary.csv."
---

# Fairness audit pratico

:::tip Obiettivo della lezione
Imparare a **misurare** se il modello sbaglia in modo diverso fra gruppi demografici, **interpretare** le tre principali definizioni di equità e **scegliere** quella più rilevante per il contesto clinico.
:::

## Notebook di riferimento

`notebooks/04_fairness_audit.ipynb`

## Quale problema stiamo affrontando

Un modello può avere AUC-PR alta **a livello aggregato** e nascondere comportamenti sistematicamente peggiori su un sottogruppo. Esempio (numeri puramente didattici):

| Gruppo | Recall | Precision |
|---|---|---|
| Caucasian | 0.72 | 0.16 |
| AfricanAmerican | 0.58 | 0.14 |
| Hispanic | 0.65 | 0.15 |

Il modello "manca" il 42% dei riammessi afroamericani contro il 28% dei caucasici. Se l'output del modello decide chi riceve un follow-up, **stiamo amplificando una disparità sanitaria pre-esistente**.

## Step 1 — Setup Fairlearn

Fairlearn è già nelle dipendenze. Importa:

```python
from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.metrics import (
    recall_score, precision_score,
    false_positive_rate, false_negative_rate,
)

metrics = {
    "selection_rate":     selection_rate,
    "recall":             recall_score,
    "precision":          precision_score,
    "false_positive_rate": false_positive_rate,
    "false_negative_rate": false_negative_rate,
}
```

## Step 2 — Calcola metriche per sottogruppo

```python
mf_race = MetricFrame(
    metrics=metrics,
    y_true=y_test,
    y_pred=y_pred,                # alla soglia ottima
    sensitive_features=df_test["race"],
)
mf_race.by_group
```

Output (esempio):

| race | selection_rate | recall | precision | FPR | FNR |
|---|---|---|---|---|---|
| AfricanAmerican | 0.31 | 0.58 | 0.14 | 0.27 | 0.42 |
| Asian          | 0.28 | 0.69 | 0.18 | 0.24 | 0.31 |
| Caucasian      | 0.34 | 0.72 | 0.16 | 0.30 | 0.28 |
| Hispanic       | 0.30 | 0.65 | 0.15 | 0.27 | 0.35 |
| Other          | 0.29 | 0.61 | 0.14 | 0.27 | 0.39 |

Stessa cosa per `age`:

```python
mf_age = MetricFrame(
    metrics=metrics,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=df_test["age"],
)
```

## Step 3 — Le tre definizioni di equità

### Demographic parity (statistical parity)

> *La probabilità di essere classificati come "a rischio" è uguale fra i gruppi.*

$$P(\hat{Y} = 1 \mid A = a) = P(\hat{Y} = 1 \mid A = b)$$

Misura: differenza di **selection_rate** fra gruppi.

```python
from fairlearn.metrics import demographic_parity_difference
demographic_parity_difference(y_test, y_pred, sensitive_features=df_test["race"])
```

**Quando ha senso**: in scenari dove vogliamo *intervenire ugualmente* su tutti i gruppi (es. un programma di outreach universale).

**Quando NON ha senso**: in sanità, se il tasso di malattia **vero** è diverso fra gruppi, forzare la parità di selezione introduce errori.

### Equalized odds

> *Dato lo stesso vero stato (riammesso o no), la probabilità di essere classificati correttamente è uguale fra i gruppi.*

$$P(\hat{Y} = 1 \mid Y = y, A = a) = P(\hat{Y} = 1 \mid Y = y, A = b), \quad \forall y$$

Misura: gap di **TPR (recall)** e **FPR** fra gruppi.

```python
from fairlearn.metrics import equalized_odds_difference
equalized_odds_difference(y_test, y_pred, sensitive_features=df_test["race"])
```

**Quando ha senso**: nella maggioranza dei contesti clinici. Vogliamo che, *fra i pazienti realmente a rischio*, tutti i gruppi abbiano la stessa probabilità di essere intercettati.

### Predictive parity

> *Dato che il modello predice "a rischio", la probabilità di essere realmente riammesso è uguale fra i gruppi.*

$$P(Y = 1 \mid \hat{Y} = 1, A = a) = P(Y = 1 \mid \hat{Y} = 1, A = b)$$

Misura: gap di **precision (PPV)** fra gruppi.

**Quando ha senso**: in scenari di allocazione di risorse limitate (chi riceve la telefonata di follow-up). Vogliamo che il "tasso di vero positivo fra i flagged" sia uguale.

## Step 4 — Il teorema di impossibilità

:::warning Chouldechova 2017 e Kleinberg et al. 2016
**Se le prevalenze sono diverse fra gruppi** (qui lo sono — la readmission rate varia con `race` e `age`), **non puoi soddisfare contemporaneamente demographic parity, equalized odds e predictive parity**. È un teorema, non un'opinione.
:::

Devi **scegliere** quale definizione ottimizzare. La scelta è etica e operativa, non tecnica.

## Step 5 — Leggi `fairness_summary.csv`

Il file generato da `readmit-train`:

| sensitive_feature | demographic_parity_diff | equalized_odds_diff | predictive_parity_diff |
|---|---|---|---|
| race | 0.06 | 0.14 | 0.04 |
| age  | 0.21 | 0.18 | 0.05 |

Lettura:

- **Su `race`**: il gap più grande è in equalized odds (0.14). Recall/FPR differiscono fra gruppi.
- **Su `age`**: il gap più grande è in demographic parity (0.21). Il modello "alerta" molto di più gli anziani.

### È un problema?

- Per `age`: clinicamente ragionevole (gli anziani hanno **vero** rischio più alto). Demographic parity NON è la metrica giusta qui.
- Per `race`: 14 punti di gap in equalized odds **richiede attenzione**. Significa che il modello intercetta meglio i riammessi di un gruppo rispetto a un altro.

## Step 6 — Mitigazione (cenni)

Fairlearn offre algoritmi di mitigazione. Esempi:

### Pre-processing: reweighting

Dai pesi diversi agli esempi in training per bilanciare i gruppi.

### In-processing: ExponentiatedGradient

```python
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds

mitigator = ExponentiatedGradient(
    estimator=LogisticRegression(max_iter=1000),
    constraints=EqualizedOdds(),
)
mitigator.fit(X_train, y_train, sensitive_features=A_train)
```

### Post-processing: ThresholdOptimizer

Sceglie soglie **diverse per gruppo** per equalizzare TPR/FPR.

:::warning Trade-off
Tutte queste tecniche **sacrificano performance aggregata** (AUC-PR globale) per migliorare equità. La domanda non è "se sacrificare", è "**quanto** sacrificare". Va deciso con stakeholder clinici e di governance.
:::

## Esercizi guidati

**E1.** Calcola `MetricFrame.by_group` per `gender`. Ci sono disparità rilevanti? Aspettative cliniche?

**E2.** Costruisci una tabella per `race` con: prevalenza vera (proporzione di reali riammessi nel gruppo), selection_rate del modello (frazione di flagged). Se selection_rate ≫ prevalenza vera, il modello "sovra-alerta" quel gruppo.

**E3.** Applica `ThresholdOptimizer` con `EqualizedOdds` e ricalcola le metriche per gruppo. Quanto migliora l'equalized odds gap? Di quanto cala l'AUC-PR globale?

**E4.** Scrivi mezza pagina che risponde a: *"In un programma di follow-up post-dimissione con capacità limitata, quale fra demographic parity, equalized odds e predictive parity ha più senso? Motiva."*. Non c'è una risposta unica giusta — sarà la tua **discussione etica**.

## Cose da annotare per il report finale

1. **Tabella `MetricFrame.by_group`** per race **e** age.
2. **Tre gap** (demographic_parity, equalized_odds, predictive_parity) per ciascun attributo.
3. **Identificazione del gruppo più penalizzato** e di quello più "iper-alertato".
4. **Scelta motivata** della definizione di fairness di riferimento (e perché le altre due *non* sono adatte).
5. **(Opzionale)** Risultato di una tecnica di mitigazione applicata.

## Prossimo passo

[**Esercizi proposti**](./06-esercizi-proposti.md): cinque esercizi graduati per consolidare quello che hai imparato, dall'introduzione di una nuova feature al re-tuning della soglia su una nuova matrice costi.
