---
sidebar_position: 7
title: "Interpretabilità & limiti"
description: Capitolo: Interpretabilità & limiti. Progetto Hospital Readmission 30d.
---

# Interpretabilità e limiti del modello

## Indice

---

## 1. Perche' l'interpretabilita' e' obbligatoria, non opzionale

In sanita' il modello produce una **decisione che cambia il trattamento**. Un alert di "alto rischio" attiva un follow-up. Senza spiegazione, il clinico non puo':

- **Validare** la decisione (a volte il modello vede pattern che il clinico riconosce subito come falsi positivi).
- **Adattare l'intervento** alle ragioni specifiche del rischio (es. focus sulla riconciliazione farmacologica vs sull'aderenza terapeutica).
- **Comunicare** al paziente il motivo del follow-up.
- **Documentare** la decisione nel fascicolo clinico (richiesto da regulation EU AI Act per "sistemi ad alto rischio").

Tre livelli di interpretabilita':

1. **Globale**: cosa pesa di piu' nel modello complessivo?
2. **Locale**: perche' questo paziente e' classificato a rischio?
3. **Controfattuale**: cosa cambierebbe se questa feature fosse diversa?

## 2. Coefficienti della Logistic Regression (interpretabilita' globale)

La LogReg con feature standardizzate produce coefficienti $\beta_j$ direttamente leggibili:

$$
\hat{p} = \sigma(\beta_0 + \sum_j \beta_j x_j)
$$

dove ogni $x_j$ ha media 0 e std 1.

**Interpretazione**: $\beta_j = 0.40$ significa che, **a parita' delle altre feature**, una std in piu' nella feature $j$ aumenta il **log-odds** di 0.40, ovvero moltiplica il rischio per $e^{0.40} \approx 1.49$ (49% in piu').

Esempio (output `interpretability.logistic_regression_coefficients`):

| Feature | Coef | Interpretation |
|---|---|---|
| `prior_healthcare_use` | +0.45 | Pazienti con piu' visite/ricoveri pregressi: rischio molto piu' alto |
| `n_med_changed` | +0.30 | Modifica della terapia diabetica: paziente fragile |
| `time_in_hospital` | +0.18 | Permanenza prolungata: gravita' baseline maggiore |
| `A1C_measured` | -0.12 | HbA1c misurata: paziente meglio gestito (ipotesi Strack 2014 confermata) |
| `age=[20-30)` | -0.40 | Pazienti giovani: rischio molto piu' basso |
| `discharge_to_home` | -0.25 | Dimissione a domicilio normale: prognosi migliore |

**Caveat**: con OneHotEncoder ogni categoria diventa un coefficiente separato. La feature `race=AfricanAmerican` ha un suo $\beta$, distinto da quello di `race=Caucasian`. Se un coefficiente ha segno problematico, e' una bandiera rossa per il fairness audit.

## 3. Feature importance di RandomForest

Il `feature_importances_` di RF e' **impurity-based** (Mean Decrease Impurity, Breiman 2001):

$$
\text{importance}_j = \sum_{t \in \text{tree}} \sum_{n \in \text{node}_j(t)} \Delta\text{Gini}(n)
$$

Cioe' la somma su tutti gli split che usano la feature $j$ del miglioramento di Gini.

**Pro**: veloce (O(numero di nodi)), nativo.

**Contro nota**: bias verso variabili continue ad alta cardinalita'. Esempio classico: l'ID paziente, se per errore non rimosso, otterrebbe importance altissima perche' permette infiniti split.

Soluzione piu' rigorosa: **permutation importance** (sklearn `inspection.permutation_importance`):

$$
\text{importance}_j = \mathbb{E}\big[\,\text{score}(\hat{y}, y) - \text{score}(\hat{y}_{\pi_j}, y)\,\big]
$$

dove $\hat{y}_{\pi_j}$ e' la predizione dopo aver permutato la feature $j$ (rompendone il legame con il target). Importance = quanto degrada la metrica permutando. Indipendente dal tipo di modello, no bias di cardinalita'.

## 4. SHAP (interpretabilita' locale)

SHAP (Lundberg & Lee 2017) attribuisce a ogni feature di una **specifica predizione** un contributo additivo:

$$
\hat{f}(\mathbf{x}) = \phi_0 + \sum_j \phi_j(\mathbf{x})
$$

dove $\phi_0$ e' il valore atteso del modello e $\phi_j(\mathbf{x})$ e' lo "Shapley value" della feature $j$ per quel paziente.

**Output**:
- "Per questo paziente, il rischio predetto e' 0.42. La feature `prior_healthcare_use=12` contribuisce +0.18, `age=[70-80)` contribuisce +0.06, `A1C_measured=No` contribuisce +0.04, le altre feature contribuiscono complessivamente -0.05."

In `interpretability.shap_summary`. Dipendenza opzionale (`pip install -e ".[advanced]"`).

**Limiti pratici**:

- Costoso: O(N x trees x features) su RF. Per dataset grandi va campionato (max 200-500 righe).
- I valori SHAP **non sono probabilita'**: sono contributi additivi al log-odds (per LR) o all'output del modello.

## 5. Limiti del modello su questo dataset

### 5.1 Limiti dei dati

| Limite | Conseguenza |
|---|---|
| Periodo 1999-2008 | Pratiche cliniche obsolete (es. introduzione DPP-4 inhibitors post-2006). Modello da retrainare ogni 3-5 anni. |
| Codifica ICD-9 | Dal 2015 negli USA si usa ICD-10. La pipeline non e' direttamente trasferibile. |
| Manca *socioeconomic status* | Predittore forte e mancante: il modello ha un "soffitto" di performance. |
| Manca *medication adherence* post-dimissione | Predittore causale fondamentale: non disponibile a tempo di predizione comunque. |
| Manca *outpatient visit follow-up* | Confound: pazienti con miglior follow-up territoriale hanno meno readmission. |

### 5.2 Limiti del modello

- **AUC-ROC tipico 0.60-0.67** in letteratura (Strack 2014, Rajkomar 2018, Pollard 2018). Non e' un fallimento del modello, e' un **plateau strutturale** del problema.
- **Calibrazione imperfetta**: la LogReg e' meglio calibrata di RF, ma e' tarata sul training set. Eventi acuti post-dimissione (non inclusi nel modello) producono **underconfidence** sistematica.
- **No causalita'**: il modello e' associativo, non causale. L'inclusione di `prior_inpatient` non significa "ridurre i ricoveri pregressi riduce la readmission futura".

### 5.3 Implicazioni per il deployment

- **Periodicita' di retraining**: ogni 6-12 mesi su nuovi dati.
- **Drift detection**: monitor su distribuzione delle feature in input (KS-test mensile).
- **Ombra (shadow mode)** prima del go-live: il modello gira ma le sue predizioni non vengono usate; si confrontano con le decisioni dei clinici per costruire il dataset di follow-up.
- **A/B test** dell'intervento attivato dal modello (follow-up vs no follow-up sui pazienti predetti a rischio).
- **Logging delle predizioni** in audit trail per ricostruire eventuali decisioni controverse.

## 6. Limiti dell'analisi degli errori

L'analisi dei **falsi negativi** (pazienti riammessi che il modello non ha intercettato) e' importante ma ha trappole:

- I FN possono essere **eterogenei**: alcuni hanno feature simili ai TP (errori "casuali" al confine della soglia), altri sono fenotipi clinici che il modello non "vede" (es. pazienti con condizioni socio-economiche difficili, non rappresentate nel dataset).
- Aggregare i FN in "ha le stesse caratteristiche" puo' generare **interventi sbagliati**. Un FN classico su Diabetes-130: paziente Asian, eta' [40-50), basso `prior_healthcare_use`. Il modello sottoconta quel fenotipo perche' nel dataset i pazienti Asian sono ~640 su 80.000.

Conclusione operativa: l'analisi degli errori va integrata con **case review clinico** sui pazienti specifici, non solo aggregata statisticamente.

## 7. Automation bias

Il rischio piu' insidioso del deployment: il clinico, dopo qualche mese, **smette di pensare** e si fida ciecamente dell'output del modello. Ogni paziente che il modello segnala riceve follow-up; ogni paziente non segnalato non lo riceve. La capacita' di **override** del clinico e' formalmente prevista ma nella pratica viene usata raramente.

Mitigazioni:

1. **Mostrare incertezza**: probabilita' invece di label binaria, intervallo di confidenza se calibrato.
2. **Mostrare le feature contributing** (top 3-5 SHAP/coefficienti). Costringe il clinico a guardare la spiegazione, non solo il numero.
3. **Audit periodico** del rate di override (clinical override rate): se sotto soglia, attivare formazione.
4. **Comunicare i limiti**: AUC-ROC 0.65 = **accuratezza moderata**. Non e' un oracolo.

## 8. Riferimenti

- **Lundberg, S. M. & Lee, S.-I.** (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS 30.
- **Breiman, L.** (2001). *Random Forests*. Machine Learning 45(1) — Mean Decrease Impurity.
- **Rajkomar, A. et al.** (2018). *Scalable and accurate deep learning with electronic health records*. NPJ Digital Medicine 1, 18.
- **Pollard, T. J. et al.** (2018). *The eICU Collaborative Research Database*. Scientific Data 5.
- **Goddard, K., Roudsari, A., Wyatt, J. C.** (2012). *Automation bias: a systematic review of frequency, effect mediators, and mitigators*. JAMIA 19(1).
- **EU AI Act** (Regolamento (UE) 2024/1689), allegato III: i sistemi di IA per il triage medico sono "ad alto rischio".