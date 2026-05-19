---
sidebar_position: 2
title: "Prerequisiti di Machine Learning"
description: "Concetti base di ML necessari per affrontare il progetto: classificazione binaria, split train/test, classi sbilanciate, metriche, pipeline scikit-learn. Esempi e ripasso veloce."
---

# Prerequisiti di Machine Learning

:::info A chi serve questa pagina
Se "logistic regression" e "ColumnTransformer" ti dicono qualcosa, puoi saltare oltre. Se invece sei al primo progetto ML serio, prenditi 20 minuti per ripassare i concetti qui sotto — torneranno **tutti** nei capitoli successivi.
:::

## 1. Classificazione binaria

**Definizione informale**: assegnare a ogni osservazione una delle due possibili etichette (`0` o `1`), partendo da un insieme di feature.

Esempio:

| Età | Num. farmaci | Ricoveri pregressi | → | `readmitted_30d` |
|---|---|---|---|---|
| 65 | 12 | 3 | → | **1** (riammesso) |
| 42 | 4  | 0 | → | **0** (non riammesso) |

Il modello impara una funzione $f(\mathbf{x}) \to [0, 1]$ che produce una **probabilità**. Per decidere la classe applichi una **soglia**:

$$
\hat{y} = \begin{cases} 1 & \text{se } f(\mathbf{x}) \geq \tau \\ 0 & \text{altrimenti} \end{cases}
$$

Per default $\tau = 0.5$, ma vedremo che **questa è quasi sempre la scelta sbagliata** in contesto clinico.

## 2. Train / Validation / Test

| Split | A cosa serve | Quando si usa |
|---|---|---|
| **Train** | Stima i parametri del modello (es. i coefficienti della LogReg). | Durante il training. |
| **Validation** | Sceglie iperparametri e soglia. | Durante il tuning. |
| **Test** | Stima onesta della performance "in produzione". | **Una sola volta**, alla fine. |

:::warning Data leakage: il peccato originale
Se calcoli la media di una feature usando **tutti** i dati (train + test) e poi imputi i missing, hai contaminato il test con informazione che in produzione non avresti. Risultato: la performance riportata è **gonfiata** e in produzione collasserà. Tutto il preprocessing va fittato **solo sul train** (sklearn `Pipeline` lo garantisce automaticamente).
:::

### Stratified split

Quando una classe è rara (qui ~11%), un random split può finire per mandare quasi tutti i positivi in train e lasciarne pochissimi in test. Uno **stratified split** preserva la proporzione delle classi in entrambi i fold.

### Group-aware split (concetto chiave del progetto)

Se lo stesso paziente compare più volte (qui circa il 30% del dataset), uno split casuale può mettere ricoveri **dello stesso paziente** in train e in test. Il modello "vede" pattern di quel paziente in training e li ritrova in test → overfitting nascosto.

Soluzione: `GroupShuffleSplit` o `GroupKFold` di scikit-learn, raggruppando per `patient_nbr`. Tutti i ricoveri di un paziente finiscono nello stesso fold.

## 3. Classi sbilanciate

**Sbilanciata = una classe è molto più frequente dell'altra.** Qui: 89% negativi vs 11% positivi.

Conseguenze:

- L'**accuracy** mente: un modello che predice sempre `0` ha l'89% di accuracy ma è inutile.
- La **soglia 0.5** spesso non produce alcun positivo: la probabilità media è bassa per costruzione.
- Servono **metriche e tecniche dedicate** (vedi capitolo dedicato).

### Tre strategie classiche

1. **Class weighting**: dai più peso agli errori sulla classe rara durante il training (`class_weight="balanced"` in sklearn).
2. **Resampling**: oversampling della minoranza (SMOTE) o undersampling della maggioranza.
3. **Threshold tuning**: lascia il modello "naturale" ma sposta la soglia in fase di decisione.

Nel progetto usiamo principalmente **(1) + (3)**, perché sono trasparenti e non alterano la distribuzione originale.

## 4. Metriche per classi sbilanciate

### Confusion matrix

|                | Predetto = 0 | Predetto = 1 |
|---             |---           |---           |
| **Reale = 0**  | TN (true negative)  | FP (false positive)  |
| **Reale = 1**  | FN (false negative) | TP (true positive)   |

Dalle quattro celle si derivano tutte le metriche utili:

| Metrica | Formula | Significato clinico |
|---|---|---|
| **Recall** (sensibilità) | $\frac{TP}{TP+FN}$ | Frazione di pazienti riammessi **che sono stati intercettati**. |
| **Precision** | $\frac{TP}{TP+FP}$ | Frazione di **allarmi corretti**. |
| **F1** | media armonica di precision e recall | Compromesso. |
| **F-β** ($\beta > 1$) | enfatizza la recall | Utile quando perdere un paziente costa più di un falso allarme. |

### AUC-ROC vs AUC-PR

- **AUC-ROC**: area sotto la curva ROC. Stima la capacità di **ordinare** i pazienti per rischio. Su classi molto sbilanciate può dare numeri ottimistici.
- **AUC-PR** (precision-recall): area sotto la curva precision-recall. **Molto più informativa** quando i positivi sono rari, perché la baseline non è 0.5 ma la prevalenza (qui 0.11).

:::tip Quale uso?
Nel progetto la metrica **primaria** è **AUC-PR**, perché coerente con lo sbilanciamento. ROC la riportiamo come riferimento.
:::

## 5. Pipeline scikit-learn

Un `Pipeline` collega più step in sequenza:

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
])

pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(class_weight="balanced")),
])

pipe.fit(X_train, y_train)            # tutto il preprocessing fittato sul train
y_proba = pipe.predict_proba(X_test)  # in inferenza, stesse trasformazioni
```

Vantaggi:

- Niente data leakage.
- Serializzabile con `joblib` → riusabile in inferenza senza ricostruire nulla.
- Sostituire un modello richiede una sola riga (`("clf", RandomForestClassifier())`).

## 6. Ensemble: il minimo indispensabile

**Random Forest**: tante decision tree addestrate su bootstrap del dataset; ogni split valuta solo un sottoinsieme casuale di feature. Si vota a maggioranza.

- Pro: gestisce non linearità, robusta a outlier, importanza feature out-of-the-box.
- Contro: meno interpretabile di una LogReg, può overfittare se troppo profonda.

**Gradient Boosting (XGBoost / LightGBM)**: alberi costruiti in sequenza, ognuno corregge gli errori del precedente.

- Pro: spesso lo state-of-the-art su tabular.
- Contro: tuning più delicato, dipendenza esterna.

Nel progetto: **Random Forest** come ensemble di riferimento (no dipendenze extra), XGBoost opzionale.

## Ripasso in 5 punti

1. Classificazione binaria = probabilità + soglia.
2. Stratified split + group-aware split (per `patient_nbr`) per evitare leakage.
3. Sbilanciamento → no accuracy, sì AUC-PR + recall + F-β.
4. Pipeline sklearn = preprocessing + modello in un solo oggetto, senza leakage.
5. LogReg (interpretabile) **vs** Random Forest (espressivo). Confronto, non scelta a priori.

## Prossimo passo

[**Prerequisiti clinici**](./03-prerequisiti-clinici.md) per imparare il vocabolario sanitario di base (ICD-9, HbA1c, HRRP) che ti servirà per non perderti nei dati.
