"""Hyperparameter tuning con cross-validation **group-aware**.

Per le grid piccole (LogReg, RF) usiamo `GridSearchCV`. Su Diabetes-130
con ~80k righe il tempo di tuning e' dominato dal fit, non dal numero
di combinazioni: GridSearch e' adeguato.

**Punto chiave**: la `cv` passata a GridSearchCV e' uno
`StratifiedGroupKFold` (`splits.make_group_cv`), e `groups` viene passato
a `fit()`. Cosi' lo stesso `patient_nbr` non appare mai in train+val di
uno stesso fold.

**Scoring**: `average_precision` (AUC-PR). Su classi sbilanciate (~11%
positivi) AUC-PR e' piu' informativa di AUC-ROC: misura specificamente
quanto bene il modello discrimina la classe minoritaria.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from .config import (
    DEFAULT_CONFIG,
    LOGREG_PARAM_GRID,
    PipelineConfig,
    RF_PARAM_GRID,
)
from .splits import make_group_cv

logger = logging.getLogger(__name__)


# Scoring di default: average precision (AUC-PR), robusta su sbilanciamento.
DEFAULT_SCORING: str = "average_precision"


@dataclass
class TuningResult:
    """Risultato del tuning di un singolo modello."""
    model_name: str
    best_estimator: Pipeline
    best_params: dict[str, Any]
    best_score: float                     # average_precision (CV, group-aware)
    cv_results: dict[str, Any]
    duration_seconds: float


def tune_grid(
    name: str,
    pipeline: Pipeline,
    param_grid: dict[str, list],
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    config: PipelineConfig = DEFAULT_CONFIG,
    scoring: str = DEFAULT_SCORING,
) -> TuningResult:
    """Tuning esauriente via GridSearchCV con StratifiedGroupKFold."""
    cv = make_group_cv(config)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=config.n_jobs,
        verbose=config.verbose,
        refit=True,
        return_train_score=False,
    )
    t0 = time.perf_counter()
    # `groups` e' essenziale: senza, StratifiedGroupKFold lancia errore.
    search.fit(X, y, groups=groups)
    duration = time.perf_counter() - t0
    logger.info(
        "[%s] tuning completato in %.1fs. Best %s = %.4f",
        name, duration, scoring, search.best_score_,
    )
    return TuningResult(
        model_name=name,
        best_estimator=search.best_estimator_,
        best_params=dict(search.best_params_),
        best_score=float(search.best_score_),
        cv_results=search.cv_results_,
        duration_seconds=duration,
    )


def tune_all_models(
    pipelines: dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    config: PipelineConfig = DEFAULT_CONFIG,
    scoring: str = DEFAULT_SCORING,
) -> dict[str, TuningResult]:
    """Esegue il tuning su tutte le pipeline registrate.

    Mappatura nome-modello -> param_grid:
        - LogReg       -> LOGREG_PARAM_GRID
        - RandomForest -> RF_PARAM_GRID
        - XGBoost      -> grid minimal (single point) se presente; il
                          tuning approfondito di XGB e' lasciato come
                          estensione (`pip install -e ".[advanced]"`).
    """
    results: dict[str, TuningResult] = {}
    for name, pipeline in pipelines.items():
        if name == "LogReg":
            grid = LOGREG_PARAM_GRID
        elif name == "RandomForest":
            grid = RF_PARAM_GRID
        elif name == "XGBoost":
            grid = {"model__n_estimators": [400]}  # placeholder: niente tuning di default
        else:
            raise ValueError(f"Tuning non definito per modello '{name}'.")
        results[name] = tune_grid(
            name=name,
            pipeline=pipeline,
            param_grid=grid,
            X=X,
            y=y,
            groups=groups,
            config=config,
            scoring=scoring,
        )
    return results


def summarize_tuning(results: dict[str, TuningResult]) -> pd.DataFrame:
    """Tabella riepilogo tuning (ordinata per score decrescente)."""
    rows = [
        {
            "model": r.model_name,
            "score_cv": r.best_score,
            "duration_s": round(r.duration_seconds, 2),
            "best_params": r.best_params,
        }
        for r in results.values()
    ]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("score_cv", ascending=False).reset_index(drop=True)
    return df


__all__ = [
    "DEFAULT_SCORING",
    "TuningResult",
    "tune_grid",
    "tune_all_models",
    "summarize_tuning",
]
