---
sidebar_position: 1
title: "Setup ambiente"
description: "Guida pratica per configurare l'ambiente di sviluppo: clone repository, virtualenv Python, installazione dipendenze, download dataset UCI 296."
---

# Setup ambiente

:::tip Tempo stimato
~20 minuti su una connessione decente, di cui ~5 per scaricare il dataset.
:::

## Prerequisiti di sistema

- **Python 3.10 o superiore** (`python3 --version` per verificare).
- **Git** (`git --version`).
- ~500 MB liberi su disco (dataset + dipendenze).
- Sistema operativo: Linux, macOS, o Windows con WSL2 (consigliato).

## 1. Clona la repository

```bash
git clone https://github.com/fedcal/Predizione-della-riammissione-ospedaliera-a-30-Giorni.git
cd Predizione-della-riammissione-ospedaliera-a-30-Giorni
```

Ti ritroverai con questa struttura:

```
.
├── README.md
├── data/                # vuota: dataset da scaricare
├── notebooks/           # 5 notebook didattici
├── scripts/             # script di build/run
├── src/readmit_pipeline/ # package Python installabile
├── tests/               # test pytest
├── pyproject.toml       # configurazione del package
├── requirements.txt
└── website/             # documentazione Docusaurus
```

## 2. Crea un virtual environment

```bash
python3 -m venv venv
source venv/bin/activate     # Linux/macOS
# venv\Scripts\activate      # Windows PowerShell
```

Dopo l'attivazione, il prompt dovrebbe iniziare con `(venv)`.

:::warning Mai installare a livello di sistema
Le dipendenze del progetto possono entrare in conflitto con altri progetti Python. Usa **sempre** un virtualenv (o conda, o uv, o poetry — l'importante è isolare).
:::

## 3. Installa il package in modalità editable

```bash
pip install --upgrade pip
pip install -e ".[notebooks]"
```

Cosa fa questo comando?

- `pip install -e .` installa il package `readmit_pipeline` in modalità **editable**: le modifiche al codice in `src/` sono immediatamente attive senza reinstallare.
- `[notebooks]` è un **extra** che include `jupyter`, `jupytext`, `nbformat`, `matplotlib`, `seaborn` — utili per lavorare con i notebook didattici.

Verifica che l'installazione sia andata a buon fine:

```bash
readmit-train --help
readmit-predict --help
```

Dovresti vedere l'help dei due entry-point CLI.

## 4. Scarica il dataset

Il dataset **Diabetes 130-US Hospitals for Years 1999–2008** **non è incluso** nella repo per motivi di policy/dimensione. Va scaricato manualmente.

### Opzione A — UCI ML Repository (ufficiale)

1. Vai a [archive.ics.uci.edu/dataset/296](https://archive.ics.uci.edu/dataset/296).
2. Clicca su "Download" e scarica lo ZIP.
3. Estrai i file e copia in `data/raw/`:
   - `diabetic_data.csv` (~20 MB)
   - `IDS_mapping.csv` (~2 KB)

### Opzione B — Kaggle

1. [kaggle.com/datasets/brandao/diabetes](https://www.kaggle.com/datasets/brandao/diabetes)
2. Scarica `diabetic_data.csv` e `IDS_mapping.csv`.
3. Copiali in `data/raw/`.

Struttura attesa:

```
data/
└── raw/
    ├── diabetic_data.csv
    └── IDS_mapping.csv
```

### Verifica

```bash
head -1 data/raw/diabetic_data.csv | tr ',' '\n' | head -20
wc -l data/raw/diabetic_data.csv
```

Dovresti vedere ~50 colonne nell'header e **101.767 righe** (di cui 1 è l'header → 101.766 record).

## 5. Smoke test

Esegui un training "veloce" per verificare che tutto funzioni:

```bash
readmit-train --quick
```

Dovrebbe completarsi in **~1 minuto** e produrre in `reports/`:

- `models/best_model.joblib`
- `cv_summary.csv`
- `holdout_metrics.json`
- `fairness_summary.csv`
- `figures/*.png`

Se vedi questi file, **il setup è completo** ✅.

## 6. Avvia JupyterLab

```bash
jupyter lab notebooks/
```

Si aprirà nel browser la lista dei 5 notebook didattici. Apri `01_eda_demographics_target.ipynb` per partire — vedi il prossimo capitolo.

## Troubleshooting comune

### `pip install -e .` fallisce con errori di compilazione

Probabilmente manca un compilatore C per build di `numpy`/`scipy`. Su Ubuntu:

```bash
sudo apt install build-essential python3-dev
```

Su macOS:

```bash
xcode-select --install
```

Su Windows: installa Microsoft C++ Build Tools, oppure usa WSL2.

### `readmit-train` non viene trovato

Probabilmente non hai attivato il venv. Riattivalo:

```bash
source venv/bin/activate
```

### Errore: "data/raw/diabetic_data.csv not found"

Controlla che il file sia esattamente in `data/raw/` (non in `data/` o in una sottocartella diversa).

### Il training è lentissimo

Esegui `readmit-train --quick` per la prima volta, è normale che il training completo richieda 5–15 minuti.

## Setup riassuntivo

```bash
git clone https://github.com/fedcal/Predizione-della-riammissione-ospedaliera-a-30-Giorni.git
cd Predizione-della-riammissione-ospedaliera-a-30-Giorni
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -e ".[notebooks]"

# scarica diabetic_data.csv e IDS_mapping.csv da UCI 296
# copia in data/raw/

readmit-train --quick
jupyter lab notebooks/
```

## Prossimo passo

[**Prima esplorazione**](./02-prima-esplorazione.md): apri il primo notebook e capisci cosa stai guardando.
