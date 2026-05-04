"""Smoke test sul mapping ICD-9 -> macro-categorie cliniche (Strack 2014, Tabella 2)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from readmit_pipeline.icd9 import (
    CATEGORY_CIRCULATORY,
    CATEGORY_DIABETES,
    CATEGORY_DIGESTIVE,
    CATEGORY_GENITOURINARY,
    CATEGORY_INJURY,
    CATEGORY_MISSING,
    CATEGORY_MUSCULOSKELETAL,
    CATEGORY_NEOPLASMS,
    CATEGORY_OTHER,
    CATEGORY_RESPIRATORY,
    add_diagnosis_category_columns,
    map_icd9_to_category,
)


def test_diabetes_codes() -> None:
    """Codici 250.xx vanno tutti in Diabetes."""
    assert map_icd9_to_category("250") == CATEGORY_DIABETES
    assert map_icd9_to_category("250.83") == CATEGORY_DIABETES
    assert map_icd9_to_category("250.6") == CATEGORY_DIABETES


def test_circulatory_range_390_459_and_785() -> None:
    assert map_icd9_to_category("390") == CATEGORY_CIRCULATORY
    assert map_icd9_to_category("428.0") == CATEGORY_CIRCULATORY
    assert map_icd9_to_category("459.9") == CATEGORY_CIRCULATORY
    assert map_icd9_to_category("785") == CATEGORY_CIRCULATORY
    assert map_icd9_to_category("459") == CATEGORY_CIRCULATORY


def test_respiratory_range_460_519_and_786() -> None:
    assert map_icd9_to_category("460") == CATEGORY_RESPIRATORY
    assert map_icd9_to_category("486") == CATEGORY_RESPIRATORY
    assert map_icd9_to_category("519.9") == CATEGORY_RESPIRATORY
    assert map_icd9_to_category("786") == CATEGORY_RESPIRATORY


def test_digestive_genitourinary_neoplasms_injury_musculoskeletal() -> None:
    assert map_icd9_to_category("520") == CATEGORY_DIGESTIVE
    assert map_icd9_to_category("787") == CATEGORY_DIGESTIVE
    assert map_icd9_to_category("580") == CATEGORY_GENITOURINARY
    assert map_icd9_to_category("788") == CATEGORY_GENITOURINARY
    assert map_icd9_to_category("140") == CATEGORY_NEOPLASMS
    assert map_icd9_to_category("239") == CATEGORY_NEOPLASMS
    assert map_icd9_to_category("800") == CATEGORY_INJURY
    assert map_icd9_to_category("999") == CATEGORY_INJURY
    assert map_icd9_to_category("710") == CATEGORY_MUSCULOSKELETAL
    assert map_icd9_to_category("739") == CATEGORY_MUSCULOSKELETAL


def test_v_and_e_codes_go_to_other() -> None:
    assert map_icd9_to_category("V58.67") == CATEGORY_OTHER
    assert map_icd9_to_category("E885.9") == CATEGORY_OTHER


def test_missing_handling() -> None:
    assert map_icd9_to_category(None) == CATEGORY_MISSING
    assert map_icd9_to_category(float("nan")) == CATEGORY_MISSING
    assert map_icd9_to_category("") == CATEGORY_MISSING


def test_add_diagnosis_category_columns_does_not_mutate() -> None:
    df = pd.DataFrame({
        "diag_1": ["250.83", "428.0", None],
        "diag_2": ["V58.67", "786", "850"],
        "diag_3": ["140", "999", "250"],
    })
    out = add_diagnosis_category_columns(df)
    # Immutability.
    assert "diag_1_cat" not in df.columns
    # Nuove colonne presenti.
    assert "diag_1_cat" in out.columns
    assert "diag_2_cat" in out.columns
    assert "diag_3_cat" in out.columns
    # Valori coerenti.
    assert out.loc[0, "diag_1_cat"] == CATEGORY_DIABETES
    assert out.loc[1, "diag_1_cat"] == CATEGORY_CIRCULATORY
    assert out.loc[2, "diag_1_cat"] == CATEGORY_MISSING
    assert out.loc[2, "diag_3_cat"] == CATEGORY_DIABETES
