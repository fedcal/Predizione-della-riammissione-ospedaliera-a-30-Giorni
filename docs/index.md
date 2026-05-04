---
layout: home
title: Home
nav_order: 1
description: >-
  Pipeline ML production-friendly per predire la riammissione ospedaliera a 30
  giorni su pazienti diabetici (UCI 296). Classificazione binaria sbilanciata,
  ICD-9 grouping, group-aware split, fairness audit con Fairlearn.
permalink: /
---

<div class="hero-banner" markdown="0">
  <h1>Hospital Readmission 30d &mdash; ML &amp; Fairness Audit</h1>
  <p>
    Dal CSV grezzo a <code>predict_readmission_risk()</code>: cleaning, ICD&#8209;9
    macro&#8209;grouping, feature engineering clinico, group&#8209;aware split,
    LogReg vs RandomForest, ottimizzazione soglia su matrice costi e audit di
    equit&agrave; con Fairlearn su <code>race</code> e <code>age</code>.
  </p>
</div>

## In sintesi

Progetto del percorso **Machine Learning Engineer** di
[DataMasters](https://datamasters.it/)/Skiller dedicato alla
**classificazione binaria** del rischio di riammissione ospedaliera entro 30
giorni su pazienti diabetici (dataset *Diabetes 130-US Hospitals*, UCI 296).
Tre scelte metodologiche guidano l'intero progetto:

- **Binarizzazione `<30` vs resto** della target multiclasse, allineata al
  tutorial Fairlearn ufficiale e all'ottica HRRP (programma USA che penalizza
  ospedali con riammissioni a 30 giorni superiori alla media).
- **Group-aware split** sul `patient_nbr`: lo stesso paziente non finisce mai
  contemporaneamente in train e test, evitando leakage subdolo dovuto agli
  encounter multipli (~30% del dataset).
- **Fairness audit obbligatorio** su `race` e `age` con
  [Fairlearn](https://fairlearn.org/), con discussione esplicita dei trade-off
  fra demographic parity, equalized odds e predictive parity in contesto
  clinico.

<div class="kpi-grid" markdown="0">
  <div class="kpi-card">
    <div class="kpi-label">AUC-ROC (atteso)</div>
    <div class="kpi-value">0.62 – 0.66</div>
    <div>letteratura su UCI 296</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">AUC-PR (atteso)</div>
    <div class="kpi-value">0.18 – 0.25</div>
    <div>prevalenza positivi ~11%</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Attributi protetti</div>
    <div class="kpi-value">race, age</div>
    <div>fairness audit Fairlearn</div>
  </div>
</div>

{: .note }
> Le metriche predittive sono **placeholder onesti** allineati alla
> letteratura (Strack et al. 2014, tutorial Fairlearn). Il dataset UCI 296
> non è incluso nel repository per motivi di policy: va scaricato manualmente
> e i numeri reali variano in funzione del seed e delle scelte di modellazione.

## Repository GitHub

- **Nome del repository**: `hospital-readmission-30d`
- **URL**: [github.com/fedcal/hospital-readmission-30d](https://github.com/fedcal/hospital-readmission-30d)
- **Documentazione (questo sito)**: pubblicata via **GitHub Pages** dalla
  cartella [`/docs`](https://github.com/fedcal/hospital-readmission-30d/tree/main/docs).

{: .note }
> La documentazione è servita da Jekyll con il tema **Just the Docs** a
> partire dai file Markdown in `docs/`. Ogni push su `main` aggiorna
> automaticamente il sito (impostazione di GitHub Pages: *Deploy from a
> branch* → `main` / `/docs`).

## Quick start

```bash
git clone https://github.com/fedcal/hospital-readmission-30d.git
cd hospital-readmission-30d
python3 -m venv venv && source venv/bin/activate
pip install -e ".[notebooks]"

# Scarica diabetic_data.csv + IDS_mapping.csv da UCI 296
# (https://archive.ics.uci.edu/dataset/296) ed estraili in data/raw/

readmit-train              # full tuning ~5-15 min
readmit-train --quick      # smoke test ~1 min
```

Inferenza programmatica:

```python
from readmit_pipeline.inference import predict_readmission_risk

paziente = {
    "race": "Caucasian",
    "gender": "Female",
    "age": "[60-70)",
    "time_in_hospital": 4,
    "num_medications": 15,
    "A1Cresult": ">7",
    "diag_1": "428",
    "insulin": "Steady",
}
proba = predict_readmission_risk(paziente)
print(f"Rischio readmission 30d: {proba:.1%}")
```

Oppure da CLI:

```bash
echo '{"race":"Caucasian","age":"[60-70)","num_medications":15,"insulin":"Steady"}' > paziente.json
readmit-predict --input paziente.json --threshold 0.3
```

{: .tip }
> Una soglia inferiore a `0.5` è ragionevole quando la matrice dei costi
> penalizza i falsi negativi (paziente ad alto rischio non intercettato)
> molto più dei falsi positivi (paziente che riceve un follow-up
> non strettamente necessario). La scelta della soglia è documentata in
> *Teoria → 05 Metriche classi sbilanciate*.

## Mappa della documentazione

### [Teoria](teoria/)

Sette capitoli che costruiscono progressivamente il razionale clinico,
metodologico ed etico del progetto:

- [01 Problem framing clinico](teoria/01_problem_framing_clinico/) — readmission a 30 giorni, HRRP, matrice costi asimmetrica.
- [02 EDA clinica](teoria/02_eda_clinica/) — distribuzione target, demografia, missing non standard, encounter multipli.
- [03 Preprocessing dati clinici](teoria/03_preprocessing_dati_clinici/) — ICD-9 grouping, IDS mapping, encoding farmaci, scaling.
- [04 Modelli classificazione binaria](teoria/04_modelli_classificazione_binaria/) — LogReg vs Random Forest su classi sbilanciate.
- [05 Metriche classi sbilanciate](teoria/05_metriche_classi_sbilanciate/) — AUC-PR, recall, F-beta, soglia ottima.
- [06 Fairness audit in sanità](teoria/06_fairness_audit_in_sanita/) — demographic parity, equalized odds, predictive parity.
- [07 Interpretabilità & limiti](teoria/07_interpretabilita_e_limiti/) — coefficienti, errori, automation bias, limiti del dataset.

### [Scelte tecniche](scelte_tecniche/)

Decisioni architetturali e di modellazione documentate con trade-off espliciti:

- [Architettura](scelte_tecniche/architettura/) — moduli `src/readmit_pipeline/`, flusso dati, CLI.
- [Scelte di modellazione](scelte_tecniche/scelte_modello/) — LogReg vs Ensemble, gestione sbilanciamento, soglia, fairness.

## Stack tecnologico

| Layer | Tecnologie |
|:--|:--|
| Linguaggio | Python 3.10+ |
| ML | scikit-learn, (opzionale) xgboost |
| Fairness | Fairlearn |
| Data | pandas, numpy, scipy |
| Plotting | matplotlib, seaborn |
| Notebook | jupyter, jupytext, nbformat |
| Persistenza | joblib |
| Documentazione | Jekyll + Just the Docs |

## Autore

Progetto realizzato da **Federico Calò** come parte del percorso
*Machine Learning Engineer* di [DataMasters](https://datamasters.it/)/Skiller.

Per altri progetti, articoli e contatti:
[**federicocalo.dev**](https://federicocalo.dev){: .btn .btn-purple }
