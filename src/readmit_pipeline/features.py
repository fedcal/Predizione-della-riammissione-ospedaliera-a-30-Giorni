"""Feature engineering domain-specific per Hospital Readmission 30d.

Tre famiglie di feature derivate, ispirate a Strack 2014 e alla logica
clinica del dominio:

1. **Complessita' farmacologica**: numero di farmaci antidiabetici
   prescritti durante il ricovero, e numero di modifiche di dosaggio
   (Up/Down). Cattura la "fragilita' farmacologica" del paziente.

2. **Intensita' di utilizzo sanitario pregresso**: somma pesata di visite
   ambulatoriali, accessi al pronto soccorso e ricoveri precedenti.
   E' uno dei predittori piu' forti in letteratura (Kansagara 2011).

3. **Comorbidita'**: presenza di specifiche macro-categorie diagnostiche
   (es. circolatorie, respiratorie) tra le diagnosi secondarie/terziarie.

Tutte le trasformazioni sono dentro un `BaseEstimator/TransformerMixin`
per essere componibili nella `Pipeline` sklearn (no leakage, serializzabili
con joblib insieme al modello).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import DIABETES_MEDS
from .icd9 import (
    ALL_CATEGORIES,
    CATEGORY_CIRCULATORY,
    CATEGORY_DIABETES,
    CATEGORY_RESPIRATORY,
    map_icd9_to_category,
)


class ReadmissionFeatureEngineer(BaseEstimator, TransformerMixin):
    """Aggiunge feature derivate al DataFrame.

    Caratteristiche create (se le colonne sorgenti sono presenti):

    - **n_med_prescribed**: numero di farmaci antidiabetici con valore
      != 'No' (cioe' attualmente prescritti).
    - **n_med_changed**: numero di farmaci antidiabetici con valore in
      {'Up', 'Down'} (modifica di dosaggio durante il ricovero).
    - **prior_healthcare_use**: visite ambulatoriali + 2*PS + 3*ricoveri
      nell'anno precedente (pesi crescenti per intensita' clinica).
    - **diag_1_cat / diag_2_cat / diag_3_cat**: macro-categoria ICD-9
      delle tre diagnosi (vedi `icd9.py`).
    - **has_circulatory_comorb / has_respiratory_comorb / has_diabetes_diag**:
      flag binarie sulle diagnosi secondarie/terziarie.
    - **A1C_measured**: 1 se A1Cresult e' diverso da 'None' (test eseguito).
      Strack 2014 mostra correlazione con tassi di readmission ridotti.
    - **change_flag / diabetesMed_flag**: binarizzazione delle colonne
      `change` (Yes/No) e `diabetesMed` (Yes/No).
    """

    def __init__(self, drop_originals: bool = False) -> None:
        # Se True, dopo il calcolo delle feature derivate rimuove le 23
        # colonne dei singoli farmaci (gia' aggregate in n_med_*).
        # Default False: i modelli non lineari possono comunque imparare
        # signal aggiuntivo dalle colonne grezze.
        self.drop_originals = drop_originals

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> "ReadmissionFeatureEngineer":
        # Salva i nomi colonna in input. sklearn>=1.0 usa
        # `feature_names_in_` per propagare i nomi attraverso la pipeline.
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
            self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            if hasattr(self, "feature_names_in_"):
                X = pd.DataFrame(X, columns=self.feature_names_in_)
            else:
                raise TypeError(
                    "ReadmissionFeatureEngineer richiede un pd.DataFrame in input."
                )
        out = X.copy()

        # --- Complessita' farmacologica ---
        med_cols = [c for c in DIABETES_MEDS if c in out.columns]
        if med_cols:
            sub = out[med_cols].astype(str)
            out["n_med_prescribed"] = (sub != "No").sum(axis=1).astype(int)
            out["n_med_changed"] = sub.isin(["Up", "Down"]).sum(axis=1).astype(int)

        # --- Intensita' di utilizzo sanitario pregresso ---
        if {"number_outpatient", "number_emergency", "number_inpatient"}.issubset(
            out.columns
        ):
            out["prior_healthcare_use"] = (
                out["number_outpatient"].fillna(0)
                + 2 * out["number_emergency"].fillna(0)
                + 3 * out["number_inpatient"].fillna(0)
            ).astype(float)

        # --- Macro-categorie diagnostiche ICD-9 ---
        for diag_col in ("diag_1", "diag_2", "diag_3"):
            if diag_col in out.columns:
                out[diag_col + "_cat"] = out[diag_col].apply(map_icd9_to_category)

        # --- Flag di comorbidita' (su diag_2 e diag_3) ---
        secondary_cats = pd.DataFrame(index=out.index)
        for col in ("diag_2_cat", "diag_3_cat"):
            if col in out.columns:
                secondary_cats[col] = out[col]
        if not secondary_cats.empty:
            out["has_circulatory_comorb"] = (
                (secondary_cats == CATEGORY_CIRCULATORY).any(axis=1).astype(int)
            )
            out["has_respiratory_comorb"] = (
                (secondary_cats == CATEGORY_RESPIRATORY).any(axis=1).astype(int)
            )
            out["has_diabetes_diag"] = (
                (secondary_cats == CATEGORY_DIABETES).any(axis=1).astype(int)
            )

        # --- A1C measured (Strack 2014 hypothesis) ---
        if "A1Cresult" in out.columns:
            out["A1C_measured"] = (
                (out["A1Cresult"].astype(str) != "None")
                & (out["A1Cresult"].notna())
            ).astype(int)
        if "max_glu_serum" in out.columns:
            out["glu_measured"] = (
                (out["max_glu_serum"].astype(str) != "None")
                & (out["max_glu_serum"].notna())
            ).astype(int)

        # --- Flag binarie su change / diabetesMed ---
        if "change" in out.columns:
            out["change_flag"] = (out["change"].astype(str) == "Ch").astype(int)
        if "diabetesMed" in out.columns:
            out["diabetesMed_flag"] = (out["diabetesMed"].astype(str) == "Yes").astype(int)

        if self.drop_originals:
            originals = [c for c in DIABETES_MEDS if c in out.columns]
            out = out.drop(columns=originals)

        return out

    def get_feature_names_out(
        self,
        input_features: Iterable[str] | None = None,
    ) -> np.ndarray:
        if input_features is None:
            return np.array([], dtype=object)
        return np.array(list(input_features), dtype=object)


__all__ = ["ReadmissionFeatureEngineer", "ALL_CATEGORIES"]
