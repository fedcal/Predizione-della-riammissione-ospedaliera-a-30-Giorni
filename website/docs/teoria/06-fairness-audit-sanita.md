---
sidebar_position: 6
title: "Fairness audit in sanità"
description: Capitolo: Fairness audit in sanità. Progetto Hospital Readmission 30d.
---

# Fairness audit in sanità

## Indice

---

## 1. Perche' un audit di equita' in un modello clinico

Un modello che predice peggio per certi sottogruppi demografici puo' **amplificare disuguaglianze gia' esistenti**. Esempio paradigmatico: Obermeyer et al. (2019) hanno mostrato che un algoritmo usato negli ospedali statunitensi per allocare risorse extra ai pazienti piu' "complessi" sottostimava sistematicamente il bisogno dei pazienti afroamericani — non perche' usasse esplicitamente la razza, ma perche' il proxy usato (spesa sanitaria storica) era a sua volta condizionato da disparita' di accesso.

Su `Diabetes-130`, le variabili sensibili obbligatorie da auditare sono:

- **`race`** (Caucasian, AfricanAmerican, Hispanic, Asian, Other, Missing)
- **`age`** (fasce decennali da [0-10) a [90-100))

Opzionalmente: `gender`.

## 2. Tre definizioni di fairness (e perche' sono incompatibili)

Tutte le definizioni provano a rispondere alla stessa domanda — *"il modello tratta i sottogruppi in modo equo?"* — ma operativamente sono diverse.

### 2.1 Demographic Parity (Statistical Parity)

$$
P(\hat{y}=1 \mid A=a) = P(\hat{y}=1 \mid A=b) \quad \forall a, b
$$

Tradotto: **il tasso di alert deve essere uguale fra gruppi**.

- **Quando ha senso**: quando vogliamo equa distribuzione del **trattamento**, indipendentemente dal "merito" (es. accesso a una risorsa con capacita' fissa).
- **Quando fallisce**: ignora il base rate. Se Caucasian ha base rate 9% e AfricanAmerican ha base rate 13%, imporre Demographic Parity significa segnalare *piu'* alert nei Caucasian (per uniformare i tassi), penalizzando AfricanAmerican.

In Fairlearn:

```python
from fairlearn.metrics import demographic_parity_difference
gap = demographic_parity_difference(y_true, y_pred, sensitive_features=race)
```

### 2.2 Equalized Odds

$$
P(\hat{y}=1 \mid y=1, A=a) = P(\hat{y}=1 \mid y=1, A=b) \quad \text{(TPR uguale)}
$$
$$
P(\hat{y}=1 \mid y=0, A=a) = P(\hat{y}=1 \mid y=0, A=b) \quad \text{(FPR uguale)}
$$

Tradotto: **dato lo stesso esito reale, la probabilita' di essere classificati e' uguale fra gruppi**.

- **Quando ha senso**: quando vogliamo che il modello sia **ugualmente accurato** per tutti i gruppi. In sanita' e' tipicamente la definizione piu' rilevante: tutti i pazienti realmente a rischio devono avere uguale probabilita' di essere intercettati.
- **Quando fallisce**: se i base rate differiscono, in genere serve un classificatore con soglie diverse per gruppo (post-processing) per soddisfarla.

In Fairlearn:

```python
from fairlearn.metrics import equalized_odds_difference
gap = equalized_odds_difference(y_true, y_pred, sensitive_features=race)
```

### 2.3 Predictive Parity (Calibration)

$$
P(y=1 \mid \hat{y}=1, A=a) = P(y=1 \mid \hat{y}=1, A=b)
$$

Tradotto: **dato che il modello segnala "a rischio", la probabilita' che sia veramente a rischio e' uguale fra gruppi (precision uguale)**.

- **Quando ha senso**: quando il messaggio del modello viene comunicato a un decisore umano. Se il clinico sente "alto rischio", la probabilita' che sia veramente alto rischio deve essere la stessa indipendentemente dalla razza.
- **Quando fallisce**: e' (in genere) **incompatibile** con Equalized Odds, vedi 2.4.

### 2.4 Impossibility Theorem (Chouldechova 2017, Kleinberg et al. 2017)

> Se i **base rate** $P(y=1 \mid A=a)$ differiscono fra gruppi e il modello non e' perfetto, **non si possono soddisfare contemporaneamente** Equalized Odds, Predictive Parity e Calibration.

Su Diabetes-130 i base rate differiscono fra `race`. Conseguenza pratica: **dobbiamo scegliere quale criterio rispettare**.

## 3. Quale criterio scegliere in contesto clinico

Per il programma di follow-up post-dimissione la domanda chiave e':

> *Vogliamo che ogni paziente realmente a rischio abbia la stessa probabilita' di essere intercettato, indipendentemente dalla razza?*

Risposta: **si**, e questo e' **Equalized Odds**.

L'alternativa (Demographic Parity) significherebbe assegnare gli stessi tassi di alert ai gruppi indipendentemente dal rischio reale: in pratica, sotto-segnalare i gruppi con base rate piu' alto (= disparita' clinica).

Predictive Parity e' valida ma piu' "orientata al messaggio": il clinico sente "alert" e si fida ugualmente. Importante ma secondaria rispetto a non perdere casi reali.

**Posizione del progetto**: priorita' a **Equalized Odds**, monitor di Demographic Parity e Predictive Parity per completezza.

## 4. Cosa misura concretamente Fairlearn

```python
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate

mf = MetricFrame(
    metrics={
        "selection_rate": selection_rate,
        "tpr": true_positive_rate,
        "fpr": false_positive_rate,
    },
    y_true=y_test, y_pred=y_pred, sensitive_features=race_test,
)
mf.by_group  # tabella metrica x gruppo
```

Output tipico su Diabetes-130 (alla soglia 0.5):

| race | selection_rate | tpr | fpr | precision |
|---|---|---|---|---|
| Caucasian | 0.18 | 0.55 | 0.12 | 0.30 |
| AfricanAmerican | 0.20 | 0.62 | 0.13 | 0.32 |
| Hispanic | 0.12 | 0.42 | 0.08 | 0.28 |
| Asian | 0.10 | 0.38 | 0.07 | 0.25 |

Diagnostica:

- **selection_rate** varia fra 0.10 e 0.20 -> Demographic Parity gap = 0.10. Sopra la soglia tipica di "attenzione" (>0.05).
- **tpr** varia fra 0.38 e 0.62 -> 24 punti di gap. Equalized Odds significativamente violata.
- **precision** varia fra 0.25 e 0.32 -> Predictive Parity gap = 0.07.

## 5. Mitigation: cosa fare se il gap e' grande

Tre famiglie di approcci (Fairlearn supporta tutte e tre):

### 5.1 Pre-processing

Modifica i dati prima del training. Esempio: rimuovere correlazioni della feature target con l'attributo protetto.

- **Pro**: agnostico al modello.
- **Contro**: degrada le performance assolute.

### 5.2 In-processing

Ottimizza un obiettivo che includa un termine di fairness.

```python
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
mitigator = ExponentiatedGradient(LogisticRegression(), constraints=EqualizedOdds())
```

- **Pro**: trade-off accuracy/fairness piu' efficiente.
- **Contro**: solo per modelli con interfaccia compatibile.

### 5.3 Post-processing

Calibra soglie diverse per gruppo dopo il training.

```python
from fairlearn.postprocessing import ThresholdOptimizer
postproc = ThresholdOptimizer(estimator=trained_model, constraints="equalized_odds")
```

- **Pro**: non richiede ri-training.
- **Contro**: usa esplicitamente l'attributo protetto al momento della decisione (`disparate treatment` in alcuni framework legali).

## 6. Comunicazione dei risultati

Un audit di fairness tecnico va tradotto in un linguaggio operativo per la direzione clinica:

> "Il modello a soglia 0.5 produce tassi di alert simili fra gruppi razziali (gap 10 pp), ma la **sensibilita' clinica** (recall) sui pazienti AfricanAmerican e' del 62% rispetto al 38% degli Asian. Cio' significa che, su 100 pazienti Asian realmente riammessi, ne intercettiamo solo 38. Per uniformare la sensibilita' clinica fra gruppi, una soglia di lavoro **post-processing** abbasserebbe il cutoff per il gruppo Asian, aumentando alert e potenzialmente FP."

E' la traduzione che il decisore (medico, governance, comitato etico) puo' usare.

## 7. Limiti del fairness audit basato sui dati storici

- Le **etichette** stesse possono essere biased: se in ospedali con piu' pazienti AfricanAmerican il personale e' inferiore, il `readmitted` osservato non e' quello clinicamente "vero" ma quello *misurato*.
- L'attributo protetto **`race`** non e' biologicamente significativo: e' un costrutto sociale. La sua eventuale predittivita' codifica disparita' strutturali (accesso ai servizi, dieta, stress cronico) piu' che differenze biologiche.
- Le metriche di fairness **non sono comprehensive**: catturano solo aspetti specifici. Servono comunque processi umani di review (algorithmic auditing).

## 8. Discussione etica: includere `race` come feature?

**Opzione A — escludere `race` dal modello**:

- Argomento: principio di non-discriminazione diretta.
- Rischio: se `race` e' correlata con altre feature predittive (es. `medical_specialty`, `payer_code`), il modello "ricostruisce" implicitamente la razza tramite proxy. Il bias resta, mascherato.

**Opzione B — includere `race`**:

- Argomento: ignorare la razza significa fingere che le disparita' strutturali non esistano. Includerla puo' permettere al modello di **correggere** per quelle disparita' (in interazione con altre feature).
- Rischio: se il modello viene comunicato a clinici, la presenza di `race` nei coefficienti puo' generare profezie che si autoavverano ("aspetto piu' problemi su questo paziente").

Non c'e' una risposta unica. **Posizione del progetto**: includere `race` nel modello, ma esplicitamente auditare le disparita' risultanti e documentarle come parte della consegna.

## 9. Riferimenti

- **Hardt, M., Price, E., Srebro, N.** (2016). *Equality of Opportunity in Supervised Learning*. NeurIPS 29.
- **Chouldechova, A.** (2017). *Fair prediction with disparate impact: A study of bias in recidivism prediction instruments*. Big Data, 5(2).
- **Kleinberg, J., Mullainathan, S., Raghavan, M.** (2017). *Inherent Trade-Offs in the Fair Determination of Risk Scores*. ITCS '17.
- **Obermeyer, Z. et al.** (2019). *Dissecting racial bias in an algorithm used to manage the health of populations*. Science, 366(6464).
- **Mehrabi, N. et al.** (2021). *A Survey on Bias and Fairness in Machine Learning*. ACM Computing Surveys.
- **Fairlearn user guide**: https://fairlearn.org/main/user_guide/