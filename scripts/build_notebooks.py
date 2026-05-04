"""Genera i 5 notebook didattici in `notebooks/`.

Lo script e' la SORGENTE DI VERITA' dei notebook: per modificarli si
edita questo file e si rilancia (`python scripts/build_notebooks.py`).
Vantaggi:
    - Sorgente in formato testuale -> diff Git leggibili.
    - Riproducibilita' (chiunque rigenera notebook identici).
    - Niente metadati casuali (kernel locale, output cache) committati.

I notebook vengono SCRITTI ma NON ESEGUITI dallo script: l'esecuzione
end-to-end e' delegata a `scripts/run_full.sh` o all'utente in JupyterLab.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


def write_notebook(name: str, cells: list[nbf.NotebookNode]) -> None:
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    out = NB_DIR / name
    nbf.write(nb, out)
    print(f"[OK] {out.relative_to(ROOT)}  ({len(cells)} celle)")


SETUP_CELL = (
    "import sys\n"
    "sys.path.insert(0, '../src')\n"
    "\n"
    "import warnings\n"
    "warnings.filterwarnings('ignore')\n"
    "\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "\n"
    "sns.set_theme(style='whitegrid')\n"
    "plt.rcParams['figure.dpi'] = 110\n"
)


# ============================================================================
# Notebook 01 - EDA: demografia e target
# ============================================================================
nb01 = [
    md(
        "# 01 - EDA: demografia, target e missing values\n\n"
        "## Obiettivi didattici\n\n"
        "1. Comprendere la **struttura del dataset Diabetes 130-US Hospitals** "
        "(101.766 ricoveri, 50 feature, 1999-2008).\n"
        "2. Esplorare la **distribuzione del target** `readmitted` "
        "(`<30`, `>30`, `NO`) e motivare la binarizzazione.\n"
        "3. Analizzare la **demografia** (race, age, gender) e verificare "
        "se il tasso di readmission a 30 giorni differisce per sottogruppo.\n"
        "4. Mappare i **valori mancanti** (codificati come `?`) e identificare "
        "le colonne da rimuovere (`weight`, `examide`, `citoglipton`).\n"
        "5. Conteggiare i **pazienti con encounter multipli** "
        "(stesso `patient_nbr` con piu' ricoveri).\n\n"
        "## Riferimenti\n\n"
        "- Strack, B. et al. (2014). *Impact of HbA1c Measurement on Hospital "
        "Readmission Rates*. BioMed Research International.\n"
        "- Fairlearn user guide: case study Diabetes 130-Hospitals.\n"
    ),
    code(SETUP_CELL + "\nfrom readmit_pipeline.data import load_raw, basic_clean, missing_summary\n"),
    md(
        "## Caricamento del dataset grezzo\n\n"
        "Il file `diabetic_data.csv` va scaricato manualmente dal "
        "**UCI ML Repository (id 296)** ed estratto in `data/raw/`.\n"
        "`load_raw()` converte automaticamente il marker `?` in NaN.\n"
    ),
    code(
        "df = load_raw()\n"
        "print(f'Shape: {df.shape}')\n"
        "print(f'Pazienti unici: {df.patient_nbr.nunique():,}')\n"
        "df.head(3)"
    ),
    md(
        "## Distribuzione del target originale (3 classi)\n\n"
        "Il target `readmitted` ha tre valori:\n"
        "- `<30`: riammesso entro 30 giorni (classe d'interesse).\n"
        "- `>30`: riammesso oltre 30 giorni.\n"
        "- `NO`: non riammesso.\n\n"
        "La classe `<30` e' fortemente sbilanciata (~11%): nessun ML \"magico\" "
        "ci puo' salvare da accuracy fuorviante.\n"
    ),
    code(
        "ax = df['readmitted'].value_counts().plot(kind='bar')\n"
        "ax.set_title('Distribuzione readmission (3 classi)')\n"
        "ax.set_ylabel('count')\n"
        "plt.show()\n"
        "print(df['readmitted'].value_counts(normalize=True).mul(100).round(2))\n"
    ),
    md(
        "## Binarizzazione e cleaning\n\n"
        "Convenzione Fairlearn: `readmitted == '<30'` -> 1, altrimenti 0.\n"
        "Rimuoviamo anche colonne strutturalmente inutili (zero variance, "
        "~97% missing) e ricoveri terminati con decesso.\n"
    ),
    code(
        "df_clean = basic_clean(df)\n"
        "print(f'Shape post-clean: {df_clean.shape}')\n"
        "rate = df_clean['readmitted_30d'].mean()\n"
        "print(f'Tasso readmission <30g: {rate*100:.2f}%')\n"
    ),
    md(
        "## Demografia: race, age, gender\n\n"
        "Verifichiamo se il tasso di readmission differisce per sottogruppo. "
        "Differenze significative motivano l'audit di fairness in fase "
        "di valutazione.\n"
    ),
    code(
        "for col in ['race', 'age', 'gender']:\n"
        "    if col not in df_clean.columns:\n"
        "        continue\n"
        "    rates = df_clean.groupby(col)['readmitted_30d'].agg(['mean', 'count'])\n"
        "    rates['mean'] = rates['mean'].mul(100).round(2)\n"
        "    rates.columns = ['readmit_rate_%', 'n']\n"
        "    print(f'\\n--- {col} ---')\n"
        "    print(rates)\n"
    ),
    md(
        "## Missing values\n\n"
        "Le colonne con piu' missing meritano una decisione esplicita "
        "(rimuovere, imputare, raggruppare). `missing_summary` produce la tabella.\n"
    ),
    code(
        "missing = missing_summary(df)\n"
        "missing.head(15)\n"
    ),
    md(
        "## Pazienti con encounter multipli\n\n"
        "Lo stesso `patient_nbr` puo' comparire piu' volte. Questo e' un "
        "motivo cruciale per usare un **group-aware split**: lasciare ricoveri "
        "dello stesso paziente sia in train sia in test introdurrebbe leakage.\n"
    ),
    code(
        "n_encounters_per_patient = df.groupby('patient_nbr').size()\n"
        "print(f'Pazienti unici: {df.patient_nbr.nunique():,}')\n"
        "print(f'Pazienti con >=2 ricoveri: {(n_encounters_per_patient > 1).sum():,}')\n"
        "print(f'Max encounter di un singolo paziente: {n_encounters_per_patient.max()}')\n"
    ),
]


# ============================================================================
# Notebook 02 - Preprocessing & ICD-9 grouping
# ============================================================================
nb02 = [
    md(
        "# 02 - Preprocessing e raggruppamento ICD-9\n\n"
        "## Obiettivi didattici\n\n"
        "1. Implementare il mapping **ICD-9 -> 9 macro-categorie** secondo "
        "Strack et al. (2014, Tabella 2).\n"
        "2. Costruire le **feature derivate cliniche**: complessita' "
        "farmacologica, intensita' di utilizzo sanitario pregresso, comorbidita'.\n"
        "3. Configurare il `ColumnTransformer` (imputer mediana per numeriche, "
        "OneHotEncoder con `min_frequency` per categoriche).\n"
        "4. Verificare la struttura della pipeline serializzabile.\n"
    ),
    code(SETUP_CELL +
        "\nfrom readmit_pipeline.data import load_raw, basic_clean\n"
        "from readmit_pipeline.icd9 import map_icd9_to_category, ALL_CATEGORIES\n"
        "from readmit_pipeline.features import ReadmissionFeatureEngineer\n"
        "from readmit_pipeline.preprocessing import build_preprocessor, infer_column_groups\n"
    ),
    md("## Mapping ICD-9 -> macro-categorie (Strack 2014)\n"),
    code(
        "examples = ['250.83', '428.0', '486', '530', 'V58.67', 'E885.9', '999', '710']\n"
        "for c in examples:\n"
        "    print(f'  {c:>8s} -> {map_icd9_to_category(c)}')\n"
        "print()\n"
        "print('Categorie disponibili:', ALL_CATEGORIES)\n"
    ),
    md("## Feature engineering\n"),
    code(
        "df = basic_clean(load_raw())\n"
        "fe = ReadmissionFeatureEngineer()\n"
        "df_fe = fe.fit_transform(df.drop(columns=['readmitted', 'readmitted_30d']))\n"
        "added = [c for c in df_fe.columns if c not in df.columns]\n"
        "print('Feature derivate aggiunte:')\n"
        "for c in added:\n"
        "    print(f'  - {c}')\n"
    ),
    md("## Distribuzione delle macro-categorie diagnostiche\n"),
    code(
        "if 'diag_1_cat' in df_fe.columns:\n"
        "    df_fe['diag_1_cat'].value_counts().plot(kind='bar')\n"
        "    plt.title('diag_1: macro-categoria principale')\n"
        "    plt.xticks(rotation=45, ha='right')\n"
        "    plt.tight_layout(); plt.show()\n"
    ),
    md(
        "## ColumnTransformer\n\n"
        "Combina:\n"
        "- imputer mediana + StandardScaler sulle feature numeriche.\n"
        "- imputer 'Unknown' + OneHotEncoder (`min_frequency=10`) sulle categoriche.\n"
    ),
    code(
        "groups = infer_column_groups(df_fe)\n"
        "print({k: len(v) for k, v in groups.items()})\n"
        "preprocessor = build_preprocessor(groups['numeric'], groups['categorical'])\n"
        "preprocessor\n"
    ),
]


# ============================================================================
# Notebook 03 - Modeling: LogReg vs Ensemble
# ============================================================================
nb03 = [
    md(
        "# 03 - Modeling: LogReg vs RandomForest (group-aware CV)\n\n"
        "## Obiettivi didattici\n\n"
        "1. Costruire la pipeline `feature_engineer -> preprocessor -> model`.\n"
        "2. Implementare lo split train/test **group-aware** su `patient_nbr`.\n"
        "3. Eseguire **StratifiedGroupKFold** con scoring `average_precision` "
        "(AUC-PR), adatto a classi sbilanciate.\n"
        "4. Confrontare un baseline interpretabile (LogReg) e un ensemble "
        "(RandomForest), entrambi con `class_weight='balanced'`.\n"
    ),
    code(SETUP_CELL +
        "\nfrom readmit_pipeline.pipeline import prepare_data, build_candidate_pipelines\n"
        "from readmit_pipeline.features import ReadmissionFeatureEngineer\n"
        "from readmit_pipeline.tuning import tune_all_models, summarize_tuning\n"
        "from readmit_pipeline.config import PipelineConfig\n"
    ),
    md("## Split group-aware + costruzione pipeline\n"),
    code(
        "config = PipelineConfig(cv_folds=3, random_state=42)\n"
        "X_train, X_test, y_train, y_test, g_train, g_test, df_clean = prepare_data(config)\n"
        "fe = ReadmissionFeatureEngineer()\n"
        "X_train_fe = fe.fit_transform(X_train)\n"
        "pipelines = build_candidate_pipelines(X_train_fe)\n"
        "list(pipelines.keys())\n"
    ),
    md("## Tuning (GridSearchCV + StratifiedGroupKFold)\n"),
    code(
        "results = tune_all_models(\n"
        "    pipelines=pipelines,\n"
        "    X=X_train, y=y_train, groups=g_train,\n"
        "    config=config,\n"
        ")\n"
        "summarize_tuning(results)\n"
    ),
]


# ============================================================================
# Notebook 04 - Fairness Audit
# ============================================================================
nb04 = [
    md(
        "# 04 - Fairness Audit (Fairlearn)\n\n"
        "## Obiettivi didattici\n\n"
        "1. Calcolare **metriche disaggregate per sottogruppo** (selection rate, "
        "TPR, FPR, precision) su `race` e `age`.\n"
        "2. Misurare il **demographic parity gap** e l'**equalized odds gap**.\n"
        "3. Discutere il trade-off tra definizioni di fairness incompatibili "
        "(Chouldechova 2017).\n"
        "4. Collegare i risultati a possibili azioni cliniche/governance.\n\n"
        "## Riferimenti\n\n"
        "- Hardt, Price, Srebro (2016). *Equality of Opportunity in Supervised "
        "Learning*. NeurIPS 29.\n"
        "- Chouldechova (2017). *Fair prediction with disparate impact*.\n"
        "- Obermeyer et al. (2019). *Dissecting racial bias in an algorithm used "
        "to manage the health of populations*. Science.\n"
    ),
    code(SETUP_CELL +
        "\nimport joblib\n"
        "from readmit_pipeline.config import MODELS_DIR\n"
        "from readmit_pipeline.fairness import (\n"
        "    fairness_report, aggregate_fairness_metrics,\n"
        ")\n"
        "from readmit_pipeline.pipeline import prepare_data\n"
        "from readmit_pipeline.config import PipelineConfig\n"
    ),
    md(
        "## Setup\n\n"
        "Carichiamo il modello allenato e ricostruiamo lo split holdout test "
        "(deterministico: stesso seed = stesso split).\n"
    ),
    code(
        "model = joblib.load(MODELS_DIR / 'best_model.joblib')\n"
        "config = PipelineConfig(random_state=42)\n"
        "_, X_test, _, y_test, _, _, df_clean = prepare_data(config)\n"
        "y_score = model.predict_proba(X_test)[:, 1]\n"
        "y_pred = (y_score >= 0.5).astype(int)\n"
        "print(f'Test size: {len(X_test)}, positivi reali: {y_test.sum()}')\n"
    ),
    md("## Fairness report disaggregato per sottogruppo\n"),
    code(
        "sf = df_clean.loc[X_test.index, ['race', 'age']]\n"
        "report_race = fairness_report(y_test.values, y_pred, sf['race'])\n"
        "report_race\n"
    ),
    code(
        "report_age = fairness_report(y_test.values, y_pred, sf['age'])\n"
        "report_age\n"
    ),
    md("## Riepilogo gap (demographic parity, equalized odds)\n"),
    code(
        "summary = aggregate_fairness_metrics(\n"
        "    y_true=y_test.values, y_pred=y_pred,\n"
        "    sensitive_features_dict={'race': sf['race'], 'age': sf['age']},\n"
        ")\n"
        "summary\n"
    ),
    md(
        "## Discussione\n\n"
        "- Un **demographic_parity_gap > 0.10** indica che il tasso di alert "
        "differisce sensibilmente fra gruppi: e' un problema se il follow-up "
        "post-dimissione ha capacita' limitata e va distribuito equamente.\n"
        "- Un **equalized_odds_gap > 0.10** indica che il modello \"manca\" "
        "piu' readmission per un gruppo che per un altro: implica disuguaglianza "
        "nell'efficacia clinica per sottogruppo.\n"
        "- I due criteri **non sono simultaneamente raggiungibili** se i base "
        "rate (prevalenza della classe positiva) differiscono fra gruppi "
        "(impossibility theorem, Chouldechova 2017).\n"
        "- Scelta consigliata in contesto di follow-up: **equalized odds** "
        "(piu' rilevante clinicamente: tutti i gruppi devono avere stessa "
        "probabilita' di essere intercettati se a rischio).\n"
    ),
]


# ============================================================================
# Notebook 05 - Interpretability & Error Analysis
# ============================================================================
nb05 = [
    md(
        "# 05 - Interpretabilita' e analisi errori\n\n"
        "## Obiettivi didattici\n\n"
        "1. Ispezionare i **coefficienti** della LogReg (modello interpretabile "
        "per costruzione) o le `feature_importances_` di RandomForest.\n"
        "2. Studiare i **falsi negativi**: pazienti riammessi entro 30 giorni "
        "non intercettati dal modello.\n"
        "3. Discutere i **limiti** del dataset (1999-2008, ICD-9, mancanza di "
        "dati socio-economici) e del modello.\n"
        "4. (Opzionale) generare uno **SHAP summary** se la dipendenza extra "
        "e' installata.\n"
    ),
    code(SETUP_CELL +
        "\nimport joblib\n"
        "from readmit_pipeline.config import MODELS_DIR, PipelineConfig\n"
        "from readmit_pipeline.pipeline import prepare_data\n"
        "from readmit_pipeline.interpretability import (\n"
        "    feature_importance_or_coef, shap_summary,\n"
        ")\n"
    ),
    code(
        "model = joblib.load(MODELS_DIR / 'best_model.joblib')\n"
        "config = PipelineConfig(random_state=42)\n"
        "_, X_test, _, y_test, _, _, _ = prepare_data(config)\n"
        "# Naviga in TransformedTargetRegressor / Pipeline annidate.\n"
        "inner = model\n"
        "while hasattr(inner, 'named_steps') and 'feature_engineer' in inner.named_steps:\n"
        "    # estrai pipeline modello (preprocessor + model) saltando il feature_engineer\n"
        "    inner = type(inner)(steps=[(n, s) for n, s in inner.steps if n != 'feature_engineer'])\n"
        "    break\n"
        "top = feature_importance_or_coef(inner, top_n=20)\n"
        "top\n"
    ),
    md("## Analisi falsi negativi\n"),
    code(
        "y_score = model.predict_proba(X_test)[:, 1]\n"
        "y_pred = (y_score >= 0.5).astype(int)\n"
        "fn_mask = (y_test.values == 1) & (y_pred == 0)\n"
        "print(f'Falsi negativi: {fn_mask.sum()} su {y_test.sum()} positivi reali ({fn_mask.sum()/y_test.sum()*100:.1f}%)')\n"
        "X_test.loc[fn_mask].describe(include='all').T.head(15)\n"
    ),
    md(
        "## Limiti e implicazioni\n\n"
        "- **Temporali**: dati 1999-2008; le pratiche cliniche sono cambiate.\n"
        "- **Coding**: ICD-9 non e' piu' lo standard (ICD-10 dal 2015 negli USA).\n"
        "- **Mancanza di feature**: nessuna informazione su condizioni socio-"
        "economiche, supporto familiare, aderenza terapeutica post-dimissione.\n"
        "- **Performance moderate**: AUC-ROC tipico in letteratura ~0.60-0.67. "
        "Il problema e' intrinsecamente difficile.\n"
        "- **Etica**: includere `race` come feature predittiva e' un dilemma. "
        "La razza e' un costrutto sociale, non biologico; la sua inclusione "
        "puo' codificare disparita' strutturali piu' che fornire informazione.\n"
    ),
]


def main() -> None:
    write_notebook("01_eda_demographics_target.ipynb", nb01)
    write_notebook("02_preprocessing_icd9_grouping.ipynb", nb02)
    write_notebook("03_models_logreg_vs_ensemble.ipynb", nb03)
    write_notebook("04_fairness_audit.ipynb", nb04)
    write_notebook("05_interpretability_and_errors.ipynb", nb05)


if __name__ == "__main__":
    main()
