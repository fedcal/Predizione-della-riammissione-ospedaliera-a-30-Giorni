---
layout: default
title: Modelli classificazione binaria
parent: Teoria
nav_order: 4
math: mathjax
description: >-
  Modelli di classificazione binaria su classi sbilanciate: Logistic
  Regression vs Random Forest, class_weight balanced, SMOTE,
  undersampling e calibrazione probabilistica.
---

# Modelli di classificazione binaria
{: .no_toc }

## Indice
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 1. Cosa significa "classificazione binaria sbilanciata"

In `Diabetes-130`, la classe positiva (`readmitted = '<30'`) e' al **~11%**. Un classificatore che predice **sempre 0** raggiunge l'89% di accuracy senza imparare nulla. Tre conseguenze:

1. **L'accuracy e' inutile** come metrica primaria.
2. **Il modello, lasciato a default, tende al collasso** verso la classe maggioritaria. Va spinto a "vedere" la minoritaria.
3. **AUC-ROC perde sensibilita'**: con base rate 11% una AUC-ROC alta puo' nascondere precision-recall scadenti.

Le strategie principali per affrontare lo sbilanciamento:

| Strategia | Pro | Contro |
|---|---|---|
| `class_weight='balanced'` | Una linea, no rischio di overfitting su sintetici | Funziona solo se il modello supporta i pesi |
| Oversampling (SMOTE) | Genera dati sintetici per la minoritaria | Rischio overfit; richiede `imblearn` |
| Undersampling | Bilanciato e veloce | Butta via dati informativi della maggioritaria |
| Ottimizzazione soglia | Conserva il fit, sposta solo il decision boundary | Serve calibrazione |

**Scelta progettuale**: `class_weight='balanced'` come default + ottimizzazione della soglia su matrice costi. SMOTE come opzione esplorabile.

## 2. Logistic Regression come baseline

### 2.1 Modello

$$
P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b), \quad \sigma(z) = \frac{1}{1+e^{-z}}
$$

Stima dei pesi via Maximum Likelihood Estimation (MLE), con loss log-loss + penalita' L2:

$$
\mathcal{L}(\mathbf{w}) = -\sum_{i} \big[ y_i \log p_i + (1-y_i) \log (1-p_i) \big] + \frac{1}{2C} \|\mathbf{w}\|_2^2
$$

dove $C$ e' l'inverso della forza di regolarizzazione (`C` piccolo = forte regolarizzazione).

### 2.2 Perche' baseline

- **Interpretabile**: i coefficienti $\beta_j$ delle feature standardizzate sono direttamente leggibili. Un $\beta_j = 0.3$ significa "ogni deviazione standard in piu' nella feature $j$ aumenta il log-odds del 30%, cioe' moltiplica il rischio per $e^{0.3} \approx 1.35$".
- **Robusto su tabular**: con regolarizzazione L2 sopravvive bene a colonne correlate (multicollinearita'), comuni post-OHE.
- **Veloce**: training su 80k righe + 200 feature richiede ~3 secondi.
- **Calibrato di default**: la probabilita' predetta e' direttamente la probabilita' stimata (non serve `CalibratedClassifierCV`).

### 2.3 Iperparametri

```python
LogisticRegression(
    C=1.0,                  # regolarizzazione moderata
    penalty="l2",           # standard, robusto
    solver="lbfgs",         # default sklearn, supporta L2
    class_weight="balanced", # gestione sbilanciamento
    max_iter=1000,          # 1000 e' di solito sufficiente
)
```

`class_weight='balanced'` calcola automaticamente:

$$
w_c = \frac{n_{\text{tot}}}{n_{\text{classi}} \cdot n_c}
$$

cioe' pesi inversamente proporzionali alla frequenza. Su Diabetes-130: $w_0 \approx 0.56$, $w_1 \approx 4.5$.

## 3. RandomForest come ensemble

### 3.1 Modello

Bagging di `n_estimators` decision tree decorrelati (Breiman 2001):

1. Per ogni albero: estrai un campione bootstrap dal training set.
2. Ad ogni split: campiona casualmente `max_features` colonne.
3. Predici con voto a maggioranza (classificazione) o media (regressione).

### 3.2 Perche' RF e non solo LogReg

- **Cattura interazioni** non lineari fra feature senza tuning fine. Su Diabetes-130 ci sono interazioni evidenti (es. `age * num_medications`, `prior_inpatient * A1Cresult`).
- **Robusto al rumore** sulle feature (irrelevant features non degradano il modello).
- **Non richiede scaling**.

Su Diabetes-130 RF tipicamente migliora l'AUC-PR di **+2-3 percentage points** rispetto a LogReg, ma:

### 3.3 Trade-off

- **Interpretabilita' degradata**: `feature_importances_` impurity-based ha bias verso variabili continue ad alta cardinalita'. Per analisi serie servono `permutation_importance` (sklearn) o SHAP.
- **Latenza inferenza** ~50 ms vs <1 ms di LogReg (su 400 alberi).
- **Probabilita' meno calibrate**: `predict_proba()` di RF non e' una vera probabilita' bayesiana; e' una frequenza fra alberi. Per applicazioni dove la probabilita' va comunicata (es. "rischio 35%") usare `CalibratedClassifierCV(rf, method='isotonic')`.

### 3.4 Iperparametri

```python
RandomForestClassifier(
    n_estimators=400,        # convergenza al plateau
    max_depth=None,          # alberi pieni: bias del singolo + decorrelazione del bagging
    max_features="sqrt",     # classico Breiman 2001
    min_samples_leaf=1,      # default; tunare a 5-10 se overfit
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)
```

## 4. XGBoost (opzionale)

Gradient boosting sequenziale (Chen & Guestrin 2016). Su Diabetes-130 il guadagno tipico su RF e' marginale (~1-2 pt AUC-PR), e il tuning richiede piu' attenzione (learning rate, depth, regolarizzazione). Lasciato come **dipendenza opzionale**:

```bash
pip install -e ".[advanced]"
readmit-train --include-xgb
```

Equivalente a `class_weight='balanced'` per XGB:

```python
XGBClassifier(scale_pos_weight=8.0, ...)  # ~ n_neg / n_pos
```

## 5. Class weighting vs SMOTE: confronto pratico

### 5.1 `class_weight='balanced'`

- Scala i contributi alla loss in base al peso di classe.
- Non genera dati sintetici: stima lo stesso modello su dati originali.
- **Default raccomandato**: ha pochi rischi.

### 5.2 SMOTE (Synthetic Minority Over-sampling Technique)

- Genera campioni sintetici per la classe minoritaria interpolando fra vicini in spazio feature.
- Implementato in [`imbalanced-learn`](https://imbalanced-learn.org/) (`pip install imbalanced-learn`).
- **Quando funziona**: dataset piccolo (centinaia di positivi), spazio feature continuo e ben strutturato.
- **Quando fallisce**: dataset grande con feature categoriche dense (caso Diabetes-130). Genera ibridi clinicamente impossibili (paziente con `gender='Female'` interpolato a 0.7 con `gender='Male'`).

Su Diabetes-130 SMOTE non ha tipicamente vantaggi su `class_weight='balanced'`, ed e' piu' complesso da gestire (va inserito in `Pipeline` con `imblearn.pipeline.Pipeline`, non `sklearn.pipeline.Pipeline`). Tenuto come **estensione esplorabile**.

## 6. Scoring di tuning: AUC-PR

Per il tuning con `GridSearchCV` non si puo' usare `accuracy` (per i motivi gia' detti). Le opzioni:

| Scoring | Quando usarlo |
|---|---|
| `average_precision` (AUC-PR) | **Default consigliato** su classi sbilanciate |
| `roc_auc` | Se la calibrazione del rank e' piu' importante della precision |
| `f1` | Se serve un singolo cutoff binario; richiede di scegliere la soglia |
| `recall` | Se il problema e' "non perdere" positivi a tutti i costi |

`average_precision` corrisponde all'area sotto la curva precision-recall. Su Diabetes-130 con base rate 0.11, una `average_precision = 0.20` e' gia' un miglioramento di ~80% rispetto al random.

Implementato in `tuning.DEFAULT_SCORING = "average_precision"`.

## 7. Cross-validation group-aware

Punto cruciale. La cross-validation **deve** essere group-aware: se il fold di validazione contiene ricoveri di pazienti che sono in train, la stima della performance e' over-ottimistica e il tuning sceglie iperparametri sbagliati.

```python
from sklearn.model_selection import StratifiedGroupKFold

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
search.fit(X, y, groups=patient_nbr)  # GROUPS e' essenziale!
```

`StratifiedGroupKFold` (sklearn>=1.0):
- Bilancia la classe positiva fra fold (`stratified`).
- Mantiene `groups` disgiunti (`group`).

Vedi `splits.make_group_cv`.

## 8. Selezione del miglior modello

A parita' di performance, in contesto clinico si preferisce il modello **piu' interpretabile**. Criteri composti:

1. **AUC-PR su CV** (group-aware): la metrica primaria.
2. **Stabilita'** (std fra fold).
3. **Interpretabilita'**: LR > RF > XGB (ordine inverso di flessibilita').
4. **Latenza inferenza** + **dimensione modello** (per produzione).

Sul nostro dataset, con `class_weight='balanced'` e tuning standard:

- LogReg: AUC-PR CV ~0.20, training 3s, latenza <1ms, modello 50KB.
- RF: AUC-PR CV ~0.22, training 60s, latenza ~50ms, modello 30MB.

Il delta di +2 pt non giustifica la perdita di ~600x sulla latenza. **In produzione: LogReg.**

## 9. Riferimenti

- **He, H. & Garcia, E. A.** (2009). *Learning from Imbalanced Data*. IEEE TKDE 21(9). Survey delle strategie.
- **King, G. & Zeng, L.** (2001). *Logistic Regression in Rare Events Data*. Political Analysis 9(2). Discussione delle correzioni di intercept per dati sbilanciati.
- **Breiman, L.** (2001). *Random Forests*. Machine Learning 45(1).
- **Chen, T. & Guestrin, C.** (2016). *XGBoost: A Scalable Tree Boosting System*. KDD.
- **Chawla, N. V. et al.** (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. JAIR 16.
