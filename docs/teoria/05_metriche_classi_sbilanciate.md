---
layout: default
title: Metriche classi sbilanciate
parent: Teoria
nav_order: 5
math: mathjax
description: >-
  Metriche di valutazione su classi sbilanciate: confusion matrix,
  precision, recall, AUC-ROC vs AUC-PR, F-beta e ottimizzazione della
  soglia su matrice dei costi asimmetrica.
---

# Metriche di valutazione su classi sbilanciate
{: .no_toc }

## Indice
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 1. Confusion matrix: tutto parte da qui

In classificazione binaria, qualunque modello + qualunque soglia produce una **confusion matrix**:

|  | Predetto 0 | Predetto 1 |
|---|---|---|
| Reale 0 | TN (true negative) | FP (false positive) |
| Reale 1 | FN (false negative) | TP (true positive) |

Tutte le metriche derivano da queste 4 quantita'.

## 2. Accuracy: perche' NON usarla

$$
\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}
$$

Su Diabetes-130 con base rate 11%, un classificatore che predice **sempre 0** ottiene:

$$
\text{Accuracy} = \frac{0 + 89}{100} = 89\%
$$

Senza imparare nulla. **Non usare accuracy come metrica primaria** su classi sbilanciate.

## 3. Recall (Sensitivity, TPR)

$$
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = P(\hat{y}=1 \mid y=1)
$$

Frazione di pazienti realmente riammessi che il modello ha intercettato. **Metrica primaria nel contesto clinico**: il costo del FN e' alto, quindi vogliamo recall elevata.

Su Diabetes-130, una recall di 0.6 significa: "su 100 pazienti riammessi, ne intercettiamo 60". Soglia di lavoro tipica: >= 0.5.

## 4. Precision (PPV)

$$
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = P(y=1 \mid \hat{y}=1)
$$

Fra i pazienti che il modello segnala come "ad alto rischio", quanti sono effettivamente riammessi. **Metrica operativa**: indica quanto e' "spreco" il programma di follow-up.

Esempio: precision = 0.30 significa "su 100 alert, 30 sono veri positivi e 70 sono FP che ricevono follow-up senza necessita'".

## 5. Trade-off precision-recall

Aumentando la **soglia decisionale** (passa da 0.5 a 0.7), il modello diventa piu' "rigoroso":

- **Precision sale** (i pochi alert sono piu' affidabili).
- **Recall scende** (perdi positivi che erano fra 0.5 e 0.7).

Visualizzato dalla **curva precision-recall** (PR curve). Esempio tipico su Diabetes-130:

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.30 | 0.18 | 0.78 | 0.29 |
| 0.50 | 0.25 | 0.55 | 0.34 |
| 0.70 | 0.40 | 0.25 | 0.31 |

Punto di lavoro consigliato: dove F1 e' massima, oppure dove la precision raggiunge la soglia operativa minima del programma di follow-up (es. "non posso accettare meno del 25% di precision per non sprecare risorse").

## 6. F1 e F-beta

F1 e' la media armonica di precision e recall:

$$
F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

E' utile come metrica singola che bilancia precision e recall.

**F-beta** generalizza con un parametro $\beta$ che pesa la recall $\beta^2$ volte la precision:

$$
F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}
$$

| $\beta$ | Significato |
|---|---|
| 0.5 | Pesa la precision il doppio della recall (es. anti-spam). |
| 1.0 | Bilanciato (F1). |
| 2.0 | Pesa la recall il doppio della precision. **Default in sanita'**. |

Per readmission usiamo F2 perche' il costo del FN >> costo del FP.

## 7. AUC-ROC vs AUC-PR

### 7.1 ROC curve

Plot di **TPR vs FPR** al variare della soglia.

$$
\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad
\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}
$$

L'area sotto la curva (**AUC-ROC**) e' la probabilita' che il modello assegni uno score piu' alto a un positivo casuale rispetto a un negativo casuale.

- AUC-ROC = 1.0 -> classificatore perfetto.
- AUC-ROC = 0.5 -> random.

### 7.2 PR curve

Plot di **Precision vs Recall** al variare della soglia. **AUC-PR** = area sotto la curva = average precision.

### 7.3 Quale usare?

| Caso | Metrica preferita |
|---|---|
| Dataset bilanciato (50/50) | AUC-ROC |
| **Dataset sbilanciato (es. 11/89)** | **AUC-PR** |
| Confronto fra modelli con stesso base rate | Entrambe |
| Pubblicazione (leaderboard standard) | AUC-ROC (per tradizione) |

Razionale: su classi sbilanciate, AUC-ROC e' **gonfiata** dai numerosi veri negativi (TN). Il denominatore di FPR e' grande, FPR resta basso anche con molti FP. AUC-PR invece guarda direttamente alla classe positiva ed e' piu' sensibile.

Su Diabetes-130:

- AUC-ROC tipico ~0.64 (sembra "decente").
- AUC-PR tipico ~0.20 (con base rate 0.11). Significa: "+80% sopra il random".

Stessa performance, ma **AUC-PR e' onesta** sullo sforzo richiesto.

## 8. Curve diagnostiche da plottare sempre

1. **Precision-Recall curve** sul test set, con marker della soglia di lavoro.
2. **ROC curve** + AUC nel titolo.
3. **Confusion matrix** alla soglia di lavoro: comunica al clinico quanti FP/FN si producono in numeri assoluti.
4. **Distribuzione delle probabilita' predette**: se il modello "schiaccia" tutto vicino a 0 (collasso verso la maggioritaria), la curva ha la moda fra 0.05 e 0.10. Se e' calibrato, copre [0, 1] in modo piu' uniforme.

## 9. Stratificazione delle metriche per sottogruppo

Per il fairness audit, non basta calcolare la metrica globale: serve la **stessa metrica per ogni sottogruppo**. Esempio Diabetes-130:

```python
df_test["y_pred"] = y_pred
recall_by_race = (
    df_test.groupby("race")
    .apply(lambda g: (g["y_pred"][g["y_true"]==1] == 1).mean())
)
```

Una recall = 0.65 globale puo' nascondere recall = 0.80 per Caucasian e 0.45 per AfricanAmerican: il modello "manca" molti piu' positivi nel secondo gruppo. E' il fenomeno di **disparate mistreatment** (Zafar et al. 2017), centrale all'audit di equita'.

Implementato in `fairness.fairness_report` via Fairlearn `MetricFrame`.

## 10. Riferimenti

- **Saito, T. & Rehmsmeier, M.** (2015). *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*. PLoS ONE 10(3).
- **Powers, D.** (2011). *Evaluation: from Precision, Recall and F-measure to ROC, Informedness, Markedness and Correlation*. Journal of Machine Learning Technologies, 2(1).
- **scikit-learn user guide**: [Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html).
- **Zafar, M. et al.** (2017). *Fairness Beyond Disparate Treatment & Disparate Impact*. WWW '17.
