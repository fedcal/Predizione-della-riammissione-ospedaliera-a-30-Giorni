---
sidebar_position: 4
title: "Percorso di studio consigliato"
description: "Ordine consigliato per affrontare il progetto Hospital Readmission 30d: mappa della documentazione, ordine dei capitoli e dei notebook, tempi indicativi per ogni step."
---

# Percorso di studio consigliato

:::tip Come usare questa pagina
Sotto trovi due percorsi: **veloce** (3–4 ore per capire le idee chiave) e **completo** (15–20 ore per fare il progetto end-to-end). Scegli in base a quanto tempo hai, ma non saltare la fase di EDA: è dove si annidano le scoperte.
:::

## Mappa della documentazione

```
docs/
├── intro.md                       ← inizia qui
│
├── per-iniziare/                  ← onboarding didattico (questa sezione)
│   ├── 01 Cosa imparerai
│   ├── 02 Prerequisiti ML
│   ├── 03 Prerequisiti clinici
│   └── 04 Percorso di studio       ← sei qui
│
├── teoria/                        ← fondamenti: il "perché"
│   ├── 01 Problem framing clinico
│   ├── 02 EDA clinica
│   ├── 03 Preprocessing dati clinici
│   ├── 04 Modelli classificazione binaria
│   ├── 05 Metriche su classi sbilanciate
│   ├── 06 Fairness audit in sanità
│   └── 07 Interpretabilità & limiti
│
├── scelte-tecniche/               ← decisioni di progetto, trade-off
│   ├── Architettura
│   └── Scelte di modellazione
│
├── laboratorio/                   ← hands-on, passo passo
│   ├── 01 Setup ambiente
│   ├── 02 Prima esplorazione
│   ├── 03 Primo modello
│   ├── 04 Leggi le metriche
│   ├── 05 Fairness audit pratico
│   └── 06 Esercizi proposti
│
└── appendici/                     ← riferimenti rapidi
    ├── Glossario
    └── FAQ
```

E nella **repository di codice**:

```
notebooks/                          ← 5 notebook didattici
├── 01_eda_demographics_target.ipynb
├── 02_preprocessing_icd9_grouping.ipynb
├── 03_models_logreg_vs_ensemble.ipynb
├── 04_fairness_audit.ipynb
└── 05_interpretability_and_errors.ipynb

src/readmit_pipeline/               ← pipeline package
scripts/                            ← entrypoint training/inferenza
data/                               ← (vuoto: dataset va scaricato)
reports/                            ← output di run (figure, metriche)
tests/                              ← test pytest
```

## Percorso veloce (3–4 ore)

Per chi vuole **capire le idee chiave** senza ancora scrivere codice.

1. **Leggi** [Intro](/docs/intro) (10 min)
2. **Leggi** [Cosa imparerai](./01-cosa-imparerai.md) → [Prerequisiti ML](./02-prerequisiti-ml.md) → [Prerequisiti clinici](./03-prerequisiti-clinici.md) (45 min)
3. **Leggi** [Teoria → 01 Problem framing](../teoria/01-problem-framing-clinico.md) (30 min)
4. **Leggi** [Teoria → 05 Metriche su classi sbilanciate](../teoria/05-metriche-classi-sbilanciate.md) (30 min)
5. **Leggi** [Teoria → 06 Fairness audit in sanità](../teoria/06-fairness-audit-sanita.md) (45 min)
6. **Sfoglia** un notebook (`01_eda_demographics_target.ipynb`) per vedere come si traduce in codice (30 min)

A questo punto sai **di cosa parla il progetto** e **perché** è impostato così. Sufficiente per discuterne in aula.

## Percorso completo (15–20 ore)

Per chi vuole **rifare il progetto end-to-end**.

### Settimana 1 — Comprensione (3–4 h)

- Tutto il percorso veloce.
- In più: [Teoria → 02 EDA clinica](../teoria/02-eda-clinica.md) e [Teoria → 03 Preprocessing](../teoria/03-preprocessing-dati-clinici.md).

**Output**: scrivi mezza pagina che risponde a *"qual è il problema, qual è la metrica, perché è difficile"*.

### Settimana 2 — Setup e dati (2–3 h)

- [Laboratorio → 01 Setup ambiente](../laboratorio/01-setup-ambiente.md): clone, `venv`, dataset.
- [Laboratorio → 02 Prima esplorazione](../laboratorio/02-prima-esplorazione.md): apri il primo notebook, esegui le celle, **modificane** almeno una.

**Output**: notebook 01 eseguito e con almeno una nuova cella di analisi tua.

### Settimana 3 — Primo modello (4–5 h)

- [Teoria → 04 Modelli classificazione binaria](../teoria/04-modelli-classificazione-binaria.md).
- [Laboratorio → 03 Primo modello](../laboratorio/03-primo-modello.md): esegui `readmit-train --quick` poi il training completo. Capisci ogni file output.
- [Laboratorio → 04 Leggi le metriche](../laboratorio/04-leggi-le-metriche.md): interpreta `holdout_metrics.json` e `cv_summary.csv`.

**Output**: tabella con AUC-PR, recall, precision dei due modelli a soglia 0.5 e a soglia ottima.

### Settimana 4 — Fairness audit (3–4 h)

- [Teoria → 06 Fairness audit in sanità](../teoria/06-fairness-audit-sanita.md) (rilettura).
- [Laboratorio → 05 Fairness audit pratico](../laboratorio/05-fairness-audit-pratico.md).

**Output**: 1 pagina di discussione su *quale gruppo penalizza il modello e perché*, con almeno **una proposta di mitigazione**.

### Settimana 5 — Interpretabilità e report (3–4 h)

- [Teoria → 07 Interpretabilità & limiti](../teoria/07-interpretabilita-limiti.md).
- Analisi errori sul notebook `05_interpretability_and_errors.ipynb`.
- Redazione **report finale** 4–5 pagine (vedi struttura nel `README.md` del repo).

**Output finale**: report + PR su GitHub con almeno un esercizio della sezione [Esercizi proposti](../laboratorio/06-esercizi-proposti.md).

## Ordine dei capitoli di teoria — perché in quest'ordine

Non è un ordine arbitrario. È il flusso logico naturale di un progetto ML clinico:

| # | Capitolo | Domanda a cui risponde |
|---|---|---|
| 1 | Problem framing | *Cosa stiamo davvero predicendo? E perché?* |
| 2 | EDA | *Che forma hanno i dati? Cosa scopro guardandoli?* |
| 3 | Preprocessing | *Come trasformo "dati clinici grezzi" in feature pulite?* |
| 4 | Modelli | *Quali modelli ha senso provare? Cosa li distingue?* |
| 5 | Metriche | *Come misuro se il modello funziona davvero?* |
| 6 | Fairness | *Funziona ugualmente bene per tutti i gruppi di pazienti?* |
| 7 | Interpretabilità & limiti | *Cosa posso dire al clinico? Cosa NON posso dire?* |

Saltare a "modelli" prima di aver letto "problem framing" è la ricetta classica per costruire una pipeline tecnicamente corretta ma **rispondere alla domanda sbagliata**.

## Quando consultare le sezioni laterali

- **`Scelte tecniche/`** → quando vuoi sapere *perché un certo trade-off è stato risolto in un certo modo* (es. *"perché non XGBoost di default?"*, *"perché class_weight invece di SMOTE?"*). È la sezione "ragioni dietro al codice".
- **`Appendici/Glossario`** → ogni volta che incontri un termine sconosciuto.
- **`Appendici/FAQ`** → prima di chiedere aiuto, controlla se la domanda c'è già.

## Strumenti consigliati

- **Editor**: VS Code con estensioni Python + Jupyter.
- **Ambiente**: Python 3.10+ in `venv`. Non installare le dipendenze a livello di sistema.
- **Notebook**: JupyterLab (`jupyter lab notebooks/`).
- **Versioning**: `git`. Crea un branch per ogni esercizio (`exercise/03-feature-engineering`).

## Pronto?

Vai a [**Laboratorio → 01 Setup ambiente**](../laboratorio/01-setup-ambiente.md) per partire con la pratica, oppure inizia con [**Teoria → 01 Problem framing**](../teoria/01-problem-framing-clinico.md) se preferisci leggere prima.
