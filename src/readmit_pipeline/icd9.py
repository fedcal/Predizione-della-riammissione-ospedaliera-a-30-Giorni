"""Mapping codici ICD-9 -> macro-categorie cliniche.

Implementa la classificazione descritta da:

    Strack, B. et al. (2014). Impact of HbA1c Measurement on Hospital
    Readmission Rates: Analysis of 70,000 Clinical Database Patient
    Records. BioMed Research International, 2014:781670, Table 2.

I codici ICD-9-CM hanno tre forme:
    - puramente numeriche (es. "250.83" = diabete con complicazioni)
    - prefissate "V" (codici di stato, es. "V58.67")
    - prefissate "E" (cause esterne di trauma, es. "E885.9")

Le macro-categorie scelte (9) sono quelle del paper:
    Circulatory, Respiratory, Digestive, Diabetes, Injury, Musculoskeletal,
    Genitourinary, Neoplasms, Other.
"""
from __future__ import annotations

import logging
from typing import Final

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Macro-categorie come stringhe canoniche.
CATEGORY_CIRCULATORY: Final[str] = "Circulatory"
CATEGORY_RESPIRATORY: Final[str] = "Respiratory"
CATEGORY_DIGESTIVE: Final[str] = "Digestive"
CATEGORY_DIABETES: Final[str] = "Diabetes"
CATEGORY_INJURY: Final[str] = "Injury"
CATEGORY_MUSCULOSKELETAL: Final[str] = "Musculoskeletal"
CATEGORY_GENITOURINARY: Final[str] = "Genitourinary"
CATEGORY_NEOPLASMS: Final[str] = "Neoplasms"
CATEGORY_OTHER: Final[str] = "Other"
CATEGORY_MISSING: Final[str] = "Missing"

ALL_CATEGORIES: Final[tuple[str, ...]] = (
    CATEGORY_CIRCULATORY,
    CATEGORY_RESPIRATORY,
    CATEGORY_DIGESTIVE,
    CATEGORY_DIABETES,
    CATEGORY_INJURY,
    CATEGORY_MUSCULOSKELETAL,
    CATEGORY_GENITOURINARY,
    CATEGORY_NEOPLASMS,
    CATEGORY_OTHER,
    CATEGORY_MISSING,
)


def _to_float(code: str) -> float | None:
    """Converte un codice ICD-9 in float. Restituisce None per V/E o invalidi.

    I codici V/E (status/external causes) non sono coperti dalle range
    numeriche di Strack 2014: vanno automaticamente in 'Other'.
    """
    if code is None:
        return None
    s = str(code).strip()
    if not s or s.upper().startswith(("V", "E")):
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def map_icd9_to_category(code: object) -> str:
    """Mappa un singolo codice ICD-9 alla sua macro-categoria.

    Le range sono quelle di Strack 2014, Table 2:

        390-459, 785        -> Circulatory
        460-519, 786        -> Respiratory
        520-579, 787        -> Digestive
        250.xx              -> Diabetes
        800-999             -> Injury
        710-739             -> Musculoskeletal
        580-629, 788        -> Genitourinary
        140-239             -> Neoplasms
        tutto il resto      -> Other

    Args:
        code: codice ICD-9 come stringa o numero (puo' essere NaN).

    Returns:
        Macro-categoria (stringa); 'Missing' se NaN, 'Other' se non
        classificabile in una categoria principale.

    Examples:
        >>> map_icd9_to_category("250.83")
        'Diabetes'
        >>> map_icd9_to_category("428.0")
        'Circulatory'
        >>> map_icd9_to_category("V58.67")
        'Other'
        >>> map_icd9_to_category(None)
        'Missing'
    """
    # NaN / None / stringa vuota.
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return CATEGORY_MISSING
    s = str(code).strip()
    if not s or s.lower() == "nan":
        return CATEGORY_MISSING

    # Diabete: matching dedicato sulla famiglia 250.xx (anche come
    # stringa, prima della conversione float, perche' "250" puo' essere
    # 250.0 in float).
    if s.startswith("250"):
        return CATEGORY_DIABETES

    num = _to_float(s)
    if num is None:
        # Codici V/E o invalidi -> Other.
        return CATEGORY_OTHER

    # Circolatorie.
    if 390 <= num < 460 or int(num) == 785:
        return CATEGORY_CIRCULATORY
    # Respiratorie.
    if 460 <= num < 520 or int(num) == 786:
        return CATEGORY_RESPIRATORY
    # Digestive.
    if 520 <= num < 580 or int(num) == 787:
        return CATEGORY_DIGESTIVE
    # Lesioni (injury / poisoning).
    if 800 <= num < 1000:
        return CATEGORY_INJURY
    # Muscoloscheletriche.
    if 710 <= num < 740:
        return CATEGORY_MUSCULOSKELETAL
    # Genitourinarie.
    if 580 <= num < 630 or int(num) == 788:
        return CATEGORY_GENITOURINARY
    # Neoplasie.
    if 140 <= num < 240:
        return CATEGORY_NEOPLASMS

    return CATEGORY_OTHER


def add_diagnosis_category_columns(
    df: pd.DataFrame,
    diag_cols: tuple[str, ...] = ("diag_1", "diag_2", "diag_3"),
    suffix: str = "_cat",
) -> pd.DataFrame:
    """Aggiunge colonne `<diag_n>_cat` con le macro-categorie cliniche.

    Restituisce un nuovo DataFrame (immutabile, l'input non viene
    modificato). Le colonne originali vengono mantenute.
    """
    out = df.copy()
    for col in diag_cols:
        if col not in out.columns:
            continue
        out[col + suffix] = out[col].apply(map_icd9_to_category)
    return out


__all__ = [
    "ALL_CATEGORIES",
    "CATEGORY_CIRCULATORY",
    "CATEGORY_RESPIRATORY",
    "CATEGORY_DIGESTIVE",
    "CATEGORY_DIABETES",
    "CATEGORY_INJURY",
    "CATEGORY_MUSCULOSKELETAL",
    "CATEGORY_GENITOURINARY",
    "CATEGORY_NEOPLASMS",
    "CATEGORY_OTHER",
    "CATEGORY_MISSING",
    "map_icd9_to_category",
    "add_diagnosis_category_columns",
]
