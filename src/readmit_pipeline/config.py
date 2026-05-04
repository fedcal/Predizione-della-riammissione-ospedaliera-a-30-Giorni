"""Configurazione globale della pipeline Hospital Readmission 30d.

Centralizza path, costanti e iperparametri di default. Frozen dataclass
per garantire immutabilita' delle configurazioni di esperimento (no
mutazioni accidentali, no side effects).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --- Paths ---
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
EXTERNAL_DIR: Path = DATA_DIR / "external"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
MODELS_DIR: Path = REPORTS_DIR / "models"

# --- Dataset ---
# UCI 296 — file ufficiali nel ZIP scaricabile da
# https://archive.ics.uci.edu/dataset/296
DATASET_FILENAME: str = "diabetic_data.csv"
IDS_MAPPING_FILENAME: str = "IDS_mapping.csv"

# Marker dei missing nel dataset originale (Strack 2014).
MISSING_MARKER: str = "?"

# Target originale (3 classi) e binarizzato (Fairlearn convention).
TARGET_COLUMN_RAW: str = "readmitted"
TARGET_COLUMN: str = "readmitted_30d"
POSITIVE_LABEL: str = "<30"

# Identificatori dei record. patient_nbr e' l'ID paziente (un paziente
# puo' comparire piu' volte): chiave per il group-aware split.
ID_COLUMNS: tuple[str, ...] = ("encounter_id", "patient_nbr")
GROUP_COLUMN: str = "patient_nbr"

# Attributi protetti per l'audit di equita'.
SENSITIVE_FEATURES: tuple[str, ...] = ("race", "age")
SENSITIVE_FEATURES_OPTIONAL: tuple[str, ...] = ("gender",)

# Colonne strutturalmente da rimuovere:
# - weight: ~97% missing.
# - examide, citoglipton: zero variance (un solo valore).
# - payer_code: ~40% missing + non e' un fattore clinico.
COLUMNS_TO_DROP: tuple[str, ...] = (
    "weight",
    "examide",
    "citoglipton",
    "payer_code",
)

# discharge_disposition_id che indicano decesso/hospice (Strack 2014).
# Pazienti con questi codici NON possono essere riammessi.
EXPIRED_DISCHARGE_IDS: tuple[int, ...] = (11, 13, 14, 19, 20, 21)

# 23 farmaci antidiabetici (Strack 2014, Tabella 1).
DIABETES_MEDS: tuple[str, ...] = (
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide",
    "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone",
    "tolazamide", "insulin",
    "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
    # examide e citoglipton vengono rimosse a monte (zero variance).
)

# --- Riproducibilita' ---
RANDOM_SEED: int = 42
TEST_SIZE: float = 0.2
CV_FOLDS: int = 5

# --- Soglia decisionale di default ---
# Modello calibrato deve scegliere una soglia coerente con la matrice
# dei costi (FN >> FP). Default = 0.5; l'ottimizzazione e' in `threshold.py`.
DEFAULT_THRESHOLD: float = 0.5

# Costo relativo: un FN (paziente ad alto rischio non intercettato che
# si riammette) costa COST_FN_OVER_FP volte un FP (follow-up inutile).
# Range tipico in letteratura: 5-20. Default conservativo = 5.
COST_FN_OVER_FP: float = 5.0


@dataclass(frozen=True)
class PipelineConfig:
    """Iperparametri e flag della pipeline.

    Frozen=True per evitare mutazioni accidentali dopo l'inizializzazione.
    Ogni esperimento crea un proprio config immutabile.
    """
    random_state: int = RANDOM_SEED
    test_size: float = TEST_SIZE
    cv_folds: int = CV_FOLDS
    use_smote: bool = False
    class_weight_balanced: bool = True
    threshold: float = DEFAULT_THRESHOLD
    cost_fn_over_fp: float = COST_FN_OVER_FP
    n_jobs: int = -1
    verbose: int = 1


DEFAULT_CONFIG: PipelineConfig = PipelineConfig()


# --- Iperparametri per il tuning ---
# Volutamente piccoli per consentire l'esecuzione in tempi didattici
# (~5-15 min) su laptop senza GPU. Ampliarli per esperimenti veri.
LOGREG_PARAM_GRID: dict[str, list] = {
    "model__C": [0.01, 0.1, 1.0, 10.0],
    "model__penalty": ["l2"],
    "model__solver": ["lbfgs"],
}

RF_PARAM_GRID: dict[str, list] = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [None, 12, 20],
    "model__min_samples_leaf": [1, 5],
    "model__max_features": ["sqrt", 0.5],
}


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DIR",
    "PROCESSED_DIR",
    "EXTERNAL_DIR",
    "REPORTS_DIR",
    "FIGURES_DIR",
    "MODELS_DIR",
    "DATASET_FILENAME",
    "IDS_MAPPING_FILENAME",
    "MISSING_MARKER",
    "TARGET_COLUMN_RAW",
    "TARGET_COLUMN",
    "POSITIVE_LABEL",
    "ID_COLUMNS",
    "GROUP_COLUMN",
    "SENSITIVE_FEATURES",
    "SENSITIVE_FEATURES_OPTIONAL",
    "COLUMNS_TO_DROP",
    "EXPIRED_DISCHARGE_IDS",
    "DIABETES_MEDS",
    "RANDOM_SEED",
    "TEST_SIZE",
    "CV_FOLDS",
    "DEFAULT_THRESHOLD",
    "COST_FN_OVER_FP",
    "PipelineConfig",
    "DEFAULT_CONFIG",
    "LOGREG_PARAM_GRID",
    "RF_PARAM_GRID",
]
