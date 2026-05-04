"""Interpretabilita' del modello: coefficienti LR, feature importance, SHAP opzionale.

Il modello scelto come baseline (LogisticRegression) e' interpretabile
per costruzione: i coefficienti standardizzati sono direttamente leggibili
come "log-odds dell'esito al variare di una std della feature".

Per i modelli ensemble (RandomForest) usiamo `feature_importances_`
(impurity-based, veloce ma con bias verso variabili a alta cardinalita').
Per analisi piu' rigorose: `permutation_importance` (incluso in sklearn).

SHAP e' una dipendenza opzionale (`pip install -e ".[advanced]"`) per
spiegazioni locali (per singolo paziente). Vedi `_safe_import_shap`.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _final_estimator(pipeline: Any) -> Any:
    """Estrae l'estimator finale (modello) dalla Pipeline."""
    if hasattr(pipeline, "named_steps"):
        return pipeline.named_steps.get("model", pipeline)
    return pipeline


def _feature_names_after_preprocessing(pipeline: Any) -> list[str]:
    """Recupera i nomi delle feature post-preprocessing (post-OHE).

    Funziona con `Pipeline` contenente `preprocessor` (ColumnTransformer).
    Restituisce lista vuota se non ricavabile.
    """
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        return list(preprocessor.get_feature_names_out())
    except (AttributeError, KeyError):
        return []


def logistic_regression_coefficients(
    pipeline: Any,
    top_n: int = 20,
) -> pd.DataFrame:
    """Top N coefficienti (per |valore|) della LogisticRegression."""
    model = _final_estimator(pipeline)
    if not hasattr(model, "coef_"):
        raise AttributeError(
            f"{type(model).__name__} non espone `coef_`: usa "
            "`feature_importances_` o `permutation_importance`."
        )
    coefs = np.ravel(model.coef_)
    names = _feature_names_after_preprocessing(pipeline)
    if len(names) != len(coefs):
        names = [f"f_{i}" for i in range(len(coefs))]
    df = pd.DataFrame({"feature": names, "coef": coefs})
    df["abs_coef"] = df["coef"].abs()
    return (
        df.sort_values("abs_coef", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def tree_feature_importance(
    pipeline: Any,
    top_n: int = 20,
) -> pd.DataFrame:
    """Top N feature_importances_ per modelli tree-based."""
    model = _final_estimator(pipeline)
    if not hasattr(model, "feature_importances_"):
        raise AttributeError(
            f"{type(model).__name__} non espone `feature_importances_`."
        )
    importances = model.feature_importances_
    names = _feature_names_after_preprocessing(pipeline)
    if len(names) != len(importances):
        names = [f"f_{i}" for i in range(len(importances))]
    df = pd.DataFrame({"feature": names, "importance": importances})
    return (
        df.sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def feature_importance_or_coef(
    pipeline: Any,
    top_n: int = 20,
) -> pd.DataFrame:
    """Wrapper polimorfo: LR -> coef, RF/XGB -> feature_importances_.

    Il DataFrame restituito ha sempre due colonne:
        - feature
        - score (coefficiente in valore assoluto, o feature_importance)
    """
    model = _final_estimator(pipeline)
    if hasattr(model, "coef_"):
        df = logistic_regression_coefficients(pipeline, top_n=top_n)
        df = df.rename(columns={"abs_coef": "score"})[["feature", "score", "coef"]]
        return df
    if hasattr(model, "feature_importances_"):
        df = tree_feature_importance(pipeline, top_n=top_n)
        df = df.rename(columns={"importance": "score"})
        return df
    raise AttributeError(
        f"Il modello {type(model).__name__} non espone ne' coef_ ne' "
        "feature_importances_. Usa permutation_importance."
    )


def _safe_import_shap() -> Any | None:
    try:
        import shap  # noqa: WPS433
        return shap
    except ImportError:
        return None


def shap_summary(
    pipeline: Any,
    X_sample: pd.DataFrame,
    max_samples: int = 200,
) -> Any | None:
    """Calcola SHAP values su un campione (se shap e' disponibile).

    Limita a `max_samples` per non saturare la memoria: 200 e' un buon
    compromesso fra rappresentativita' e velocita' (TreeExplainer su RF
    e' O(N x trees), pesante su 80k righe).

    Returns:
        Oggetto SHAP `Explanation` (o None se shap non e' installato).
    """
    shap = _safe_import_shap()
    if shap is None:
        logger.warning("shap non installato: SHAP summary saltato.")
        return None

    n = min(max_samples, len(X_sample))
    X_sub = X_sample.sample(n=n, random_state=0)

    # Trasforma manualmente con il preprocessor, poi spiega il modello finale.
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]
        X_t = preprocessor.transform(X_sub)
        explainer = shap.Explainer(model, X_t)
        return explainer(X_t)
    except Exception as exc:  # noqa: BLE001
        logger.warning("SHAP summary fallito: %s", exc)
        return None


__all__ = [
    "logistic_regression_coefficients",
    "tree_feature_importance",
    "feature_importance_or_coef",
    "shap_summary",
]
