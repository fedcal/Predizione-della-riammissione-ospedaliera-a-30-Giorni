---
sidebar_position: 1
title: Introduzione
description: |
  Pipeline ML production-friendly per predire la riammissione ospedaliera a 30 giorni su pazienti diabetici (UCI 296). Classificazione binaria sbilanciata, ICD-9 grouping, group-aware split, fairness audit con Fairlearn.
slug: /intro
---

# Hospital Readmission 30d — ML & Fairness Audit

Dal CSV grezzo a `predict_readmission_risk()`: cleaning, ICD-9 macro-grouping, feature engineering clinico, group-aware split, LogReg vs Random Forest, ottimizzazione soglia su matrice costi e audit di equità con Fairlearn su `race` e `age`.

:::tip In una riga
*Classificazione binaria del rischio di readmission a 30 giorni con fairness audit obbligatorio su race/age.*
:::

## In sintesi

Progetto del percorso **Machine Learning Engineer** di [DataMasters](https://datamasters.it/)/Skiller dedicato alla **classificazione binaria** del rischio di riammissione ospedaliera entro 30 giorni su pazienti diabetici (dataset *Diabetes 130-US Hospitals*, UCI 296). Tre scelte metodologiche guidano l'intero progetto:

- **Binarizzazione `&lt;30` vs resto** della target multiclasse, allineata al tutorial Fairlearn ufficiale e all'ottica HRRP (programma USA che penalizza ospedali con riammissioni a 30 giorni superiori alla media).
- **Group-aware split** sul `patient_nbr`: lo stesso paziente non finisce mai contemporaneamente in train e test, evitando leakage subdolo dovuto agli encounter multipli (~30% del dataset).
- **Fairness audit obbligatorio** su `race` e `age` con [Fairlearn](https://fairlearn.org/), con discussione esplicita dei trade-off fra demographic parity, equalized odds e predictive parity in contesto clinico.

## Repository GitHub

| Item | Link |
|---|---|
| Repo | [`fedcal/Predizione-della-riammissione-ospedaliera-a-30-Giorni`](https://github.com/fedcal/Predizione-della-riammissione-ospedaliera-a-30-Giorni) |
| Documentazione | [https://fedcal.github.io/Predizione-della-riammissione-ospedaliera-a-30-Giorni/](https://fedcal.github.io/Predizione-della-riammissione-ospedaliera-a-30-Giorni/) |
| Licenza | MIT |
| Stack docs | Docusaurus 3 + TypeScript + KaTeX |

:::note
Le metriche predittive sono **placeholder onesti** allineati alla letteratura (Strack et al. 2014, tutorial Fairlearn). Il dataset UCI 296 non è incluso nel repository per motivi di policy: va scaricato manualmente e i numeri reali variano in funzione del seed e delle scelte di modellazione.
:::

## Quick start

```bash
git clone https://github.com/fedcal/Predizione-della-riammissione-ospedaliera-a-30-Giorni.git
cd Predizione-della-riammissione-ospedaliera-a-30-Giorni
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

:::tip
Una soglia inferiore a `0.5` è ragionevole quando la matrice dei costi penalizza i falsi negativi (paziente ad alto rischio non intercettato) molto più dei falsi positivi (paziente che riceve un follow-up non strettamente necessario). La scelta della soglia è documentata in *Teoria → 05 Metriche su classi sbilanciate*.
:::

## Mappa della documentazione

:::tip Sei nuovo al progetto?
Inizia da [**Per iniziare → Cosa imparerai**](./per-iniziare/01-cosa-imparerai.md). Sono 4 capitoli che ti danno onboarding completo (obiettivi, prerequisiti ML, prerequisiti clinici, percorso di studio).
:::

### [Per iniziare](/docs/category/per-iniziare) — onboarding didattico

Per chi affronta il progetto da zero. Spiega obiettivi, prerequisiti e percorso consigliato:

1. [Cosa imparerai](./per-iniziare/01-cosa-imparerai.md) — patto formativo, obiettivi misurabili.
2. [Prerequisiti di Machine Learning](./per-iniziare/02-prerequisiti-ml.md) — classificazione binaria, split, metriche, pipeline.
3. [Prerequisiti clinici](./per-iniziare/03-prerequisiti-clinici.md) — readmission, ICD-9, HbA1c, HRRP, variabili sensibili.
4. [Percorso di studio](./per-iniziare/04-percorso-studio.md) — ordine consigliato, percorso veloce vs completo.

### [Teoria](/docs/category/teoria) — il "perché"

Sette capitoli che costruiscono progressivamente il razionale clinico, metodologico ed etico del progetto:

1. [Problem framing clinico](./teoria/01-problem-framing-clinico.md) — readmission a 30 giorni, HRRP, matrice costi asimmetrica.
2. [EDA clinica](./teoria/02-eda-clinica.md) — distribuzione target, demografia, missing non standard, encounter multipli.
3. [Preprocessing dati clinici](./teoria/03-preprocessing-dati-clinici.md) — ICD-9 grouping, IDS mapping, encoding farmaci, scaling.
4. [Modelli classificazione binaria](./teoria/04-modelli-classificazione-binaria.md) — LogReg vs Random Forest su classi sbilanciate.
5. [Metriche su classi sbilanciate](./teoria/05-metriche-classi-sbilanciate.md) — AUC-PR, recall, F-beta, soglia ottima.
6. [Fairness audit in sanità](./teoria/06-fairness-audit-sanita.md) — demographic parity, equalized odds, predictive parity.
7. [Interpretabilità & limiti](./teoria/07-interpretabilita-limiti.md) — coefficienti, errori, automation bias, limiti del dataset.

### [Scelte tecniche](/docs/category/scelte-tecniche) — il "perché di quel codice"

Decisioni architetturali e di modellazione documentate con trade-off espliciti:

- [Architettura](./scelte-tecniche/architettura.md) — moduli `src/readmit_pipeline/`, flusso dati, CLI.
- [Scelte di modellazione](./scelte-tecniche/scelte-modello.md) — LogReg vs Ensemble, gestione sbilanciamento, soglia, fairness.

### [Laboratorio](/docs/category/laboratorio) — hands-on passo passo

Guida pratica per chi vuole **fare** il progetto, non solo leggerlo:

1. [Setup ambiente](./laboratorio/01-setup-ambiente.md) — clone, venv, dataset, smoke test.
2. [Prima esplorazione (EDA)](./laboratorio/02-prima-esplorazione.md) — apri il dataset e impara a guardarlo.
3. [Primo modello](./laboratorio/03-primo-modello.md) — esegui il training e leggi gli output.
4. [Leggi le metriche](./laboratorio/04-leggi-le-metriche.md) — interpretazione di AUC-PR, recall, F-β, soglia.
5. [Fairness audit pratico](./laboratorio/05-fairness-audit-pratico.md) — Fairlearn, MetricFrame, mitigazione.
6. [Esercizi proposti](./laboratorio/06-esercizi-proposti.md) — 5 esercizi graduati per consolidare.

### [Appendici](/docs/category/appendici) — riferimenti rapidi

- [Glossario](./appendici/glossario.md) — termini ML, clinici e di engineering.
- [FAQ](./appendici/faq.md) — risposte alle domande ricorrenti su setup, scelte e fairness.

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
| Documentazione | Docusaurus 3 + TypeScript + KaTeX |
| CI/CD | GitHub Actions + GitHub Pages |

## Autore

Progetto realizzato da **Federico Calò** come parte del percorso *Machine Learning Engineer* di [DataMasters](https://datamasters.it/)/Skiller.

Per altri progetti, articoli e contatti: [**federicocalo.dev**](https://federicocalo.dev).
