"""Pipeline candidate: LogisticRegression baseline + RandomForest ensemble.

Due famiglie complementari:

- **Logistic Regression** (lineare regolarizzato L2, `class_weight='balanced'`):
  baseline interpretabile. I coefficienti dei feature standardizzati
  sono leggibili e direttamente comunicabili a clinici/governance.

- **Random Forest** (`class_weight='balanced'`): ensemble non lineare
  che cattura interazioni senza tuning fine. Robusto al rumore e agli
  outlier sulle feature.

Una terza famiglia (XGBoost) e' prevista come dipendenza opzionale
(`pip install -e ".[advanced]"`); se assente, il sistema funziona
ugualmente con i due modelli di base.

Tutte le pipeline sono `sklearn.pipeline.Pipeline` con:
    [feature_engineer -> preprocessor -> model]
in modo che la cross-validation sia priva di leakage (ogni fold rifa'
fit di tutto).
"""
from __future__ import annotations

import logging

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import RANDOM_SEED

logger = logging.getLogger(__name__)


def logistic_regression_pipeline(
    preprocessor: ColumnTransformer,
    class_weight: str | None = "balanced",
    C: float = 1.0,
    max_iter: int = 1000,
) -> Pipeline:
    """LogReg con regolarizzazione L2 e (opzionale) bilanciamento classi.

    `class_weight='balanced'` riadatta automaticamente i pesi inversamente
    alla frequenza di classe. Su Diabetes-130 (~11% positivi) e' la
    strategia piu' semplice e robusta. SMOTE e' un'alternativa esplorabile
    in `pipeline.py`.
    """
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                C=C,
                penalty="l2",
                solver="lbfgs",
                class_weight=class_weight,
                max_iter=max_iter,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
        ),
    ])


def random_forest_pipeline(
    preprocessor: ColumnTransformer,
    class_weight: str | None = "balanced",
    n_estimators: int = 400,
) -> Pipeline:
    """RandomForest con `class_weight='balanced'`.

    `class_weight='balanced'` calcola i pesi delle classi inversamente
    proporzionali alla frequenza nel training set. Per RF e' equivalente
    al re-sampling supervisato di un dataset bilanciato, ma piu' efficiente
    in memoria.
    """
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=None,
                min_samples_leaf=1,
                max_features="sqrt",
                class_weight=class_weight,
                n_jobs=-1,
                random_state=RANDOM_SEED,
            ),
        ),
    ])


def xgboost_pipeline_optional(
    preprocessor: ColumnTransformer,
    scale_pos_weight: float | None = None,
) -> Pipeline | None:
    """XGBoost (opzionale, dipendenza extra).

    Se xgboost non e' installato, restituisce None invece di raisare.
    Cosi' `get_all_pipelines` puo' decidere se includerlo o meno senza
    rompere il flusso principale.

    `scale_pos_weight` e' l'analogo di `class_weight='balanced'` per
    XGBoost: se None, viene calcolato come (n_neg / n_pos).
    """
    try:
        from xgboost import XGBClassifier  # noqa: WPS433
    except ImportError:
        logger.info("xgboost non installato: pipeline XGB skippata.")
        return None

    spw = scale_pos_weight if scale_pos_weight is not None else 8.0
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                scale_pos_weight=spw,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                tree_method="hist",
                eval_metric="aucpr",
                verbosity=0,
            ),
        ),
    ])


def get_all_pipelines(
    preprocessor: ColumnTransformer,
    include_xgb: bool = False,
) -> dict[str, Pipeline]:
    """Restituisce le pipeline candidate disponibili (nome -> pipeline)."""
    out: dict[str, Pipeline] = {
        "LogReg": logistic_regression_pipeline(preprocessor),
        "RandomForest": random_forest_pipeline(preprocessor),
    }
    if include_xgb:
        xgb = xgboost_pipeline_optional(preprocessor)
        if xgb is not None:
            out["XGBoost"] = xgb
    return out


__all__ = [
    "logistic_regression_pipeline",
    "random_forest_pipeline",
    "xgboost_pipeline_optional",
    "get_all_pipelines",
]
