---
sidebar_position: 4
title: "Leggi le metriche"
description: "Come interpretare correttamente AUC-PR, AUC-ROC, recall, precision, F-beta e confusion matrix in un contesto clinico con classi sbilanciate."
---

# Leggi le metriche

:::tip Obiettivo della lezione
Allenare l'**occhio clinico** sulle metriche: capire perché lo stesso modello può sembrare ottimo o pessimo a seconda di cosa misuri, e perché il default 0.5 quasi sempre è la scelta sbagliata.
:::

## Premessa: cosa stiamo misurando davvero

Su 100 ricoveri:

- 11 finiranno in readmission a 30 giorni (positivi).
- 89 no (negativi).

Il classificatore "stupido" che predice sempre `0` ha **89% di accuracy**. Inutile. Quindi la prima regola è:

> **Mai usare accuracy come metrica primaria su classi sbilanciate.**

## AUC-ROC: utile, ma tradisce

La **ROC curve** plotta TPR vs FPR al variare della soglia. L'**area sotto la curva** (AUC-ROC) misura quanto bene il modello **ordina** i positivi rispetto ai negativi.

- AUC-ROC = 0.5 → casuale.
- AUC-ROC = 1.0 → perfetto.

Su questo dataset, modelli decenti ottengono **AUC-ROC ≈ 0.65–0.67**.

:::warning Trappola
Su classi molto sbilanciate, l'AUC-ROC può essere alto (0.7+) anche se la precision è bassissima (< 20%). Motivo: la ROC valuta il ranking di **tutti** i negativi, e con 89 negativi su 100 c'è "abbondanza" di confronti facili.
:::

## AUC-PR: la metrica giusta per noi

La **precision-recall curve** plotta precision vs recall. L'AUC-PR è l'area sotto questa curva.

- **Baseline** (classificatore random): AUC-PR = **prevalenza della classe positiva** = ~0.11.
- **Modelli decenti** sul dataset: 0.20–0.30.
- **Perfetto**: 1.0.

:::tip Perché AUC-PR è "meglio" qui
Perché si concentra **solo sulla classe positiva** (riammessi). Ignora i veri negativi, che qui sono troppi e mascherano il problema. Un guadagno da 0.11 a 0.23 in AUC-PR è **2× sopra il random** — è quello che conta clinicamente.
:::

### Esempio numerico

Due modelli con stesso AUC-ROC = 0.66:

| Modello | AUC-ROC | AUC-PR | Lettura |
|---|---|---|---|
| A | 0.66 | 0.13 | Appena meglio del random sui positivi. Non utile. |
| B | 0.66 | 0.27 | 2.5× meglio del random. **Quello che vogliamo.** |

L'AUC-ROC li equipara. L'AUC-PR li distingue.

## Recall, precision, F1, F-β

Date le quattro celle della confusion matrix:

|                | Predetto = 0 | Predetto = 1 |
|---             |---           |---           |
| **Reale = 0**  | TN           | FP           |
| **Reale = 1**  | FN           | TP           |

### Recall (sensibilità)

$$\text{Recall} = \frac{TP}{TP+FN}$$

Frazione di **pazienti riammessi correttamente intercettati**. Alto recall = pochi falsi negativi.

### Precision (PPV)

$$\text{Precision} = \frac{TP}{TP+FP}$$

Frazione di **allarmi corretti**. Alta precision = pochi falsi positivi.

### F1

$$F_1 = 2 \cdot \frac{\text{P} \cdot \text{R}}{\text{P} + \text{R}}$$

Media armonica. Bilancia precision e recall **alla pari**.

### F-β

$$F_\beta = (1+\beta^2) \cdot \frac{\text{P} \cdot \text{R}}{\beta^2 \cdot \text{P} + \text{R}}$$

- $\beta = 1$ → F1.
- $\beta = 2$ → enfatizza la **recall** 4 volte di più della precision.
- $\beta = 0.5$ → enfatizza la **precision**.

:::tip Quale F usare?
Nel nostro contesto, **F2** è ragionevole: ci interessa intercettare i pazienti riammessi (alta recall), accettando qualche falso allarme in più.
:::

## La soglia di decisione: il parametro più potente

Tutte le metriche binarie (recall, precision, F1, confusion matrix) dipendono dalla **soglia** scelta. Il modello produce probabilità; la soglia decide chi è "flagged".

### Soglia 0.5 vs soglia ottima

Esempio reale dal progetto:

| Soglia | TP | FP | TN | FN | Precision | Recall | F2 |
|---|---|---|---|---|---|---|---|
| 0.50 (default) | 612 | 2728 | 12871 | 836 | 0.18 | 0.42 | 0.34 |
| 0.27 (ottima) | 1028 | 6302 | 9297 | 420 | 0.14 | **0.71** | **0.43** |

Lettura clinica:

- A 0.5 il modello **perde 836 pazienti riammessi** su 1.448 totali (58%).
- A 0.27, ne **perde solo 420** (29%).
- Costo: 3.500 falsi positivi in più (telefonate non strettamente necessarie).

Se una telefonata costa 15 $ e un ricovero perso 15.000 $, **il calcolo è scontato**.

### Come si trova la soglia ottima

```python
import numpy as np
from sklearn.metrics import confusion_matrix

def cost(tau, y_true, y_proba, cost_fn_over_fp=5):
    y_pred = (y_proba >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp + cost_fn_over_fp * fn

taus = np.linspace(0.05, 0.95, 91)
costs = [cost(t, y_val, proba_val) for t in taus]
tau_star = taus[np.argmin(costs)]
```

## Curva precision-recall: il diagramma da guardare

La PR curve mostra **tutti i possibili compromessi** fra precision e recall.

Cose da osservare:

1. **Posizione della curva**: più "in alto a destra", meglio è.
2. **Baseline orizzontale** = prevalenza positiva (~0.11). Se la curva ci tocca o sta sotto, il modello è inutile.
3. **Pendenza**: una curva ripida indica che spostare la soglia di poco cambia molto le metriche. Importante per la scelta di $\tau$.
4. **Il punto operativo scelto** ($\tau^*$): segnalo sulla curva, è il "dove sei" del tuo modello in produzione.

## Esercizi guidati

**E1.** Apri `reports/figures/pr_curve.png`. Annota:
- AUC-PR del modello.
- Coordinata (recall, precision) alla soglia 0.5.
- Coordinata alla soglia ottima.
- Distanza verticale fra il punto ottimo e la baseline 0.11.

**E2.** Cambia il rapporto costi in `config.py` (es. 10× invece di 5×). Come cambia la soglia ottima? È più bassa o più alta? Perché?

**E3.** Calcola F-β per β = 1, 2, 3 alla soglia ottima. Quale β massimizza F? Quale ha più senso clinicamente?

**E4.** Costruisci una tabella che mostri, per τ ∈ {0.1, 0.2, 0.3, 0.5, 0.7}, i valori di precision e recall. Cosa succede agli estremi?

## Errori comuni da non fare

| Errore | Conseguenza |
|---|---|
| Usare accuracy come metrica primaria | Modelli inutili sembrano eccellenti. |
| Confrontare modelli su AUC-ROC senza guardare AUC-PR | Si sceglie il modello "che ordina meglio i negativi", non quello utile. |
| Tunare la soglia sul **test set** | Data leakage. La soglia va scelta in CV o su un validation set, poi **applicata** al test. |
| Riportare solo precision senza recall (o viceversa) | Numeri "decorativi". I clinici vogliono entrambi + curva PR. |
| Definire la matrice costi **dopo** aver visto i risultati | Bias di conferma. La matrice costi va fissata in fase di **problem framing**. |

## Prossimo passo

[**Fairness audit pratico**](./05-fairness-audit-pratico.md): scopri se il modello sbaglia in modo diverso fra sottogruppi demografici, e cosa fare al riguardo.
