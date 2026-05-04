"""Caricamento e preparazione iniziale del dataset Diabetes 130-US Hospitals.

Single responsibility: I/O del file CSV grezzo, conversione del marker
"?" in NaN, binarizzazione del target, rimozione delle colonne zero-variance
e dei pazienti deceduti durante il ricovero.

Il preprocessing statistico (encoding, imputazione condizionata) vive
in `preprocessing.py` ed e' applicato dentro la pipeline sklearn.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    COLUMNS_TO_DROP,
    DATASET_FILENAME,
    EXPIRED_DISCHARGE_IDS,
    MISSING_MARKER,
    POSITIVE_LABEL,
    RAW_DIR,
    TARGET_COLUMN,
    TARGET_COLUMN_RAW,
)

logger = logging.getLogger(__name__)


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Carica il file `diabetic_data.csv` da `data/raw/` (o path custom).

    Il file UCI 296 usa `?` come marker dei missing: lo convertiamo in
    NaN per sfruttare le pipeline standard di pandas/sklearn.

    Args:
        path: path opzionale al CSV. Default: `RAW_DIR / DATASET_FILENAME`.

    Returns:
        DataFrame grezzo con NaN al posto dei marker `?`.

    Raises:
        FileNotFoundError: se il file non esiste. Il dataset va scaricato
            manualmente dall'UCI ML Repository (id 296) per motivi di
            licenza/dimensione.
    """
    if path is None:
        path = RAW_DIR / DATASET_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset non trovato: {path}.\n"
            "Scaricalo dal UCI ML Repository (id 296) ed estrai\n"
            f"`diabetic_data.csv` e `IDS_mapping.csv` in {path.parent}/."
        )
    df = pd.read_csv(path, na_values=[MISSING_MARKER])
    logger.info("Dataset caricato: %d righe x %d colonne", df.shape[0], df.shape[1])
    return df


def binarize_target(df: pd.DataFrame) -> pd.DataFrame:
    """Binarizza la colonna `readmitted` secondo Fairlearn/Strack 2014.

    Mapping:
        '<30'  -> 1 (positivo, riammesso entro 30 giorni)
        '>30'  -> 0
        'NO'   -> 0

    Aggiunge la colonna `readmitted_30d` e restituisce un nuovo DataFrame
    (immutabilita': l'input non viene modificato).
    """
    if TARGET_COLUMN_RAW not in df.columns:
        raise ValueError(
            f"Colonna target '{TARGET_COLUMN_RAW}' assente. "
            f"Colonne presenti: {list(df.columns)[:10]}..."
        )
    out = df.copy()
    out[TARGET_COLUMN] = (out[TARGET_COLUMN_RAW] == POSITIVE_LABEL).astype(int)
    rate = float(out[TARGET_COLUMN].mean())
    logger.info(
        "Target binarizzato: %d positivi (%.2f%%) su %d.",
        int(out[TARGET_COLUMN].sum()), rate * 100, len(out),
    )
    return out


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rimuove colonne strutturalmente inutili (~97% missing o zero variance).

    Vedi `config.COLUMNS_TO_DROP` per la lista completa con motivazione.
    """
    to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    if not to_drop:
        return df.copy()
    logger.info("Drop colonne strutturali: %s", to_drop)
    return df.drop(columns=to_drop)


def drop_expired_patients(df: pd.DataFrame) -> pd.DataFrame:
    """Rimuove i ricoveri terminati con decesso/hospice del paziente.

    Razionale: questi pazienti non possono essere riammessi entro 30
    giorni. Lasciarli nel training distorce la stima del modello (sono
    `readmitted=0` per ragioni non predicibili dai dati clinici di
    ammissione).

    Codici ID rimossi (da `IDS_mapping.csv`):
        11 (Expired), 13 (Hospice/home), 14 (Hospice/medical facility),
        19 (Expired at home), 20 (Expired in medical facility),
        21 (Expired place unknown).
    """
    if "discharge_disposition_id" not in df.columns:
        return df.copy()
    n_before = len(df)
    mask = ~df["discharge_disposition_id"].isin(EXPIRED_DISCHARGE_IDS)
    out = df.loc[mask].copy()
    logger.info(
        "Rimossi %d record con discharge_disposition_id expired/hospice (%.2f%% del totale).",
        n_before - len(out), 100 * (n_before - len(out)) / max(n_before, 1),
    )
    return out


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline minima di pulizia pre-split.

    Combina (in ordine, immutabile):
        1. Binarizzazione del target.
        2. Drop colonne zero-variance / quasi-tutte-missing.
        3. Drop pazienti deceduti.

    Tutto cio' che dipende da statistiche del training (imputazione,
    encoding) e' lasciato fuori: vive nella pipeline sklearn.
    """
    df = binarize_target(df)
    df = drop_unused_columns(df)
    df = drop_expired_patients(df)
    return df


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Tabella riassuntiva dei NaN per colonna (utile in EDA)."""
    n = len(df)
    counts = df.isna().sum()
    counts = counts[counts > 0].sort_values(ascending=False)
    pct = (counts / n * 100).round(2)
    return pd.DataFrame({"n_missing": counts, "pct_missing": pct})


def select_feature_target_groups(
    df: pd.DataFrame,
    drop_id_cols: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Suddivide il DataFrame in (X, y, groups) per il group-aware split.

    Args:
        df: DataFrame post `basic_clean` (target binarizzato presente).
        drop_id_cols: se True, rimuove `encounter_id` da X (sempre).
            `patient_nbr` viene rimosso da X ma usato come `groups`.

    Returns:
        (X, y, groups) dove:
            - X: feature, senza target ne' ID.
            - y: target binario `readmitted_30d`.
            - groups: serie di `patient_nbr` (per GroupShuffleSplit).
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target '{TARGET_COLUMN}' assente. Chiama `binarize_target` prima."
        )

    groups = df["patient_nbr"].copy() if "patient_nbr" in df.columns else pd.Series(
        np.arange(len(df)), index=df.index, name="patient_nbr"
    )
    y = df[TARGET_COLUMN].astype(int)

    cols_to_drop = [TARGET_COLUMN, TARGET_COLUMN_RAW]
    if drop_id_cols:
        cols_to_drop += ["encounter_id", "patient_nbr"]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    X = df.drop(columns=cols_to_drop)

    return X, y, groups


__all__ = [
    "load_raw",
    "binarize_target",
    "drop_unused_columns",
    "drop_expired_patients",
    "basic_clean",
    "missing_summary",
    "select_feature_target_groups",
]
