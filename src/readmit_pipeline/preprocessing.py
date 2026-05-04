"""Costruzione del preprocessor sklearn (ColumnTransformer).

Combina tre branch parallele:

- **Numeriche**: imputazione mediana + scaling (StandardScaler).
- **Categoriche binarie/basse cardinalita'**: imputazione 'Unknown' +
  OneHotEncoder con `handle_unknown='ignore'`.
- **Drop**: ID, target, e ogni colonna eccessivamente missing (>=95%).

Tutto in un `ColumnTransformer` per due ragioni didattiche fondamentali:

1. **No data leakage**: le statistiche (mediana, frequenze categoria)
   sono calcolate solo sul training set; in test/inferenza vengono
   riapplicate.
2. **Riproducibilita'**: serializzando il `ColumnTransformer` con
   joblib, l'inferenza e' bit-identica al training.
"""
from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


# Colonne che richiedono ID-mapping: vengono trattate come categoriche
# nominali a bassa cardinalita' anche se int (i numeri sono codici, non
# quantita').
ID_MAPPED_CATEGORICAL: tuple[str, ...] = (
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
)


def _build_numeric_branch(scale: bool = True) -> Pipeline:
    """Branch numerica: imputer mediana + scaling opzionale.

    Lo scaling e' richiesto da modelli sensibili alla scala (LogReg, SVM).
    Per modelli tree-based (RandomForest) viene comunque applicato (cost
    trascurabile, ma garantisce uniformita').
    """
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps=steps)


def _build_categorical_branch() -> Pipeline:
    """Branch categorica: imputer 'Unknown' + OneHotEncoder."""
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                # Riduce dimensionalita' aggregando categorie con
                # <min_frequency occorrenze in 'infrequent_sklearn'.
                min_frequency=10,
            ),
        ),
    ])


def infer_column_groups(X: pd.DataFrame) -> dict[str, list[str]]:
    """Classifica le colonne di X in numeric / categorical.

    Strategia:
        - Le colonne `ID_MAPPED_CATEGORICAL` vanno in 'categorical' anche
          se sono int (sono codici).
        - Le colonne `object` (stringa) vanno in 'categorical'.
        - Le restanti colonne numeriche vanno in 'numeric'.

    Le colonne con dtype `bool` vengono trattate come numeriche (sklearn
    le converte in 0/1 implicitamente).
    """
    categorical: list[str] = []
    numeric: list[str] = []

    for col in X.columns:
        if col in ID_MAPPED_CATEGORICAL:
            categorical.append(col)
            continue
        dtype = X[col].dtype
        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
            categorical.append(col)
        else:
            numeric.append(col)
    return {"numeric": numeric, "categorical": categorical}


def build_preprocessor(
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Costruisce il `ColumnTransformer` completo.

    Args:
        numeric_cols: nomi colonne continue/discrete (interi/float).
        categorical_cols: nomi colonne categoriche (object o ID-mapped).
        scale_numeric: applica StandardScaler alle numeriche. Default True.

    Returns:
        `ColumnTransformer` parametrizzato. Va `fit` su X_train; in
        test/inferenza ricicla le statistiche apprese.
    """
    transformers: list[tuple] = []
    if numeric_cols:
        transformers.append(
            ("num", _build_numeric_branch(scale=scale_numeric), list(numeric_cols))
        )
    if categorical_cols:
        transformers.append(
            ("cat", _build_categorical_branch(), list(categorical_cols))
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    logger.info(
        "Preprocessor pronto: %d numeriche, %d categoriche.",
        len(numeric_cols), len(categorical_cols),
    )
    return preprocessor


__all__ = [
    "ID_MAPPED_CATEGORICAL",
    "infer_column_groups",
    "build_preprocessor",
]
