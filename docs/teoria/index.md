---
layout: default
title: Teoria
nav_order: 2
has_children: true
permalink: /teoria/
description: >-
  Fondamenti teorici del progetto Hospital Readmission 30d: problem framing
  clinico, EDA, preprocessing dati clinici, modelli per classificazione
  binaria sbilanciata, metriche AUC-PR, fairness audit e interpretabilità.
---

# Teoria

I sette capitoli di questa sezione costruiscono progressivamente le basi
necessarie per leggere il progetto **Hospital Readmission 30d** e per
ragionare in modo critico sui risultati, sia da un punto di vista
**predittivo** che **etico-clinico**.

## Percorso consigliato di lettura

| Capitolo | Titolo | Concetti chiave |
|:--|:--|:--|
| 1 | [Problem framing clinico](01_problem_framing_clinico/) | HRRP, binarizzazione `<30`, matrice costi asimmetrica |
| 2 | [EDA clinica](02_eda_clinica/) | distribuzione target, demografia, missing `?`, encounter multipli |
| 3 | [Preprocessing dati clinici](03_preprocessing_dati_clinici/) | ICD-9 macro-grouping, IDS mapping, encoding farmaci, scaling |
| 4 | [Modelli classificazione binaria](04_modelli_classificazione_binaria/) | LogReg `class_weight=balanced`, Random Forest, gestione sbilanciamento |
| 5 | [Metriche classi sbilanciate](05_metriche_classi_sbilanciate/) | AUC-ROC vs AUC-PR, recall, F-beta, ottimizzazione soglia |
| 6 | [Fairness audit in sanità](06_fairness_audit_in_sanita/) | Fairlearn `MetricFrame`, demographic / equalized / predictive parity |
| 7 | [Interpretabilità & limiti](07_interpretabilita_e_limiti/) | coefficienti LR, feature importance, automation bias, limiti dataset |

{: .note }
> Ogni capitolo è autocontenuto: leggi nell'ordine se vuoi una progressione
> didattica coerente, oppure salta direttamente al capitolo che ti serve.
> I capitoli 1, 6 e 7 sono i più importanti per la **discussione critica**
> richiesta dalla traccia del Project Work.

## Riferimenti trasversali

- Strack, B. et al. (2014). *Impact of HbA1c Measurement on Hospital
  Readmission Rates: Analysis of 70,000 Clinical Database Patient Records*.
  BioMed Research International, 2014:781670.
- Fairlearn team (Microsoft) — documentazione e tutorial sul Diabetes
  130-Hospitals Dataset come caso studio: <https://fairlearn.org>.
- Chouldechova, A. (2017). *Fair prediction with disparate impact: A study
  of bias in recidivism prediction instruments*. Big Data, 5(2), 153–163.
- Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019).
  *Dissecting racial bias in an algorithm used to manage the health of
  populations*. Science, 366(6464), 447–453.
- UCI ML Repository, dataset id 296: <https://archive.ics.uci.edu/dataset/296>.
