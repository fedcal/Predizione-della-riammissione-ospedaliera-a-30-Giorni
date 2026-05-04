"""Orchestratore end-to-end della pipeline Hospital Readmission 30d.

Esegue, nell'ordine:
    1. Caricamento + binarizzazione + cleaning del dataset.
    2. Group-aware train/test split su `patient_nbr`.
    3. Costruzione delle pipeline candidate (LogReg + RandomForest).
    4. Tuning con StratifiedGroupKFold + scoring AUC-PR.
    5. Valutazione su holdout e ottimizzazione della soglia.
    6. Selezione del miglior modello (AUC-PR CV).
    7. Audit di equita' su race e age.
    8. Persistenza modelli + report (joblib + CSV + JSON).

Eseguibile come modulo:

    readmit-train              # full tuning, ~5-15 min
    readmit-train --quick      # smoke-test, ~30s
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from .config import (
    DEFAULT_CONFIG,
    LOGREG_PARAM_GRID,
    MODELS_DIR,
    PipelineConfig,
    REPORTS_DIR,
    RF_PARAM_GRID,
    SENSITIVE_FEATURES,
)
from .data import basic_clean, load_raw, select_feature_target_groups
from .evaluation import evaluate_on_holdout, plot_precision_recall_curve, plot_roc_curve
from .fairness import aggregate_fairness_metrics, fairness_report
from .features import ReadmissionFeatureEngineer
from .models import get_all_pipelines
from .preprocessing import build_preprocessor, infer_column_groups
from .splits import assert_no_group_leakage, group_aware_train_test_split
from .threshold import find_optimal_threshold
from .tuning import TuningResult, summarize_tuning, tune_all_models

logger = logging.getLogger(__name__)


def _attach_feature_engineering(
    pipelines: dict[str, Pipeline],
) -> dict[str, Pipeline]:
    """Antepone `ReadmissionFeatureEngineer` a ogni pipeline esistente.

    Il feature engineer crea n_med_prescribed, prior_healthcare_use,
    diag_*_cat ecc.; il preprocessor successivo le inferisce automaticamente
    grazie a `infer_column_groups`. Inserire l'engineer come step della
    pipeline (e non a mano sul DataFrame) e' essenziale per la cross
    validation: deve girare sui fold di train senza vedere il validation.
    """
    out: dict[str, Pipeline] = {}
    for name, pipe in pipelines.items():
        steps = [("feature_engineer", ReadmissionFeatureEngineer())] + list(pipe.steps)
        out[name] = Pipeline(steps=steps)
    return out


def prepare_data(
    config: PipelineConfig = DEFAULT_CONFIG,
) -> tuple[
    pd.DataFrame, pd.DataFrame,
    pd.Series, pd.Series,
    pd.Series, pd.Series,
    pd.DataFrame,
]:
    """Carica e pulisce il dataset, restituisce uno split group-aware.

    Returns:
        (X_train, X_test, y_train, y_test, g_train, g_test, df_clean)
    """
    df = load_raw()
    df = basic_clean(df)

    X, y, groups = select_feature_target_groups(df, drop_id_cols=True)

    X_train, X_test, y_train, y_test, g_train, g_test = group_aware_train_test_split(
        X, y, groups, config=config,
    )
    assert_no_group_leakage(g_train, g_test)
    return X_train, X_test, y_train, y_test, g_train, g_test, df


def build_candidate_pipelines(
    X_train_after_fe: pd.DataFrame,
    include_xgb: bool = False,
) -> dict[str, Pipeline]:
    """Costruisce tutte le pipeline candidate (LogReg + RandomForest [+ XGB]).

    Args:
        X_train_after_fe: DataFrame X_train DOPO `ReadmissionFeatureEngineer`.
            Usato solo per inferire i gruppi di colonne (numeric/categorical).
    """
    groups = infer_column_groups(X_train_after_fe)
    preprocessor = build_preprocessor(
        numeric_cols=groups["numeric"],
        categorical_cols=groups["categorical"],
    )
    base_pipelines = get_all_pipelines(preprocessor, include_xgb=include_xgb)
    return _attach_feature_engineering(base_pipelines)


def select_best_model(
    tuning_results: dict[str, TuningResult],
) -> tuple[str, TuningResult]:
    """Seleziona il modello con AUC-PR CV piu' alto."""
    best_name = max(tuning_results, key=lambda k: tuning_results[k].best_score)
    return best_name, tuning_results[best_name]


def save_artifacts(
    best_name: str,
    best_estimator: Pipeline,
    holdout_metrics_by_model: dict,
    cv_summary: pd.DataFrame,
    fairness_summary: pd.DataFrame,
    fairness_full: pd.DataFrame,
    optimal_threshold: float,
    models_dir: Path = MODELS_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, Path]:
    """Persiste modello + report metriche + report fairness."""
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_named_path = models_dir / f"{best_name.lower()}_best.joblib"
    model_default_path = models_dir / "best_model.joblib"
    joblib.dump(best_estimator, model_named_path)
    joblib.dump(best_estimator, model_default_path)

    cv_path = reports_dir / "cv_summary.csv"
    cv_summary.to_csv(cv_path, index=False)

    metrics_path = reports_dir / "holdout_metrics.json"
    payload = {
        "models": holdout_metrics_by_model,
        "best_model": best_name,
        "optimal_threshold": optimal_threshold,
    }
    metrics_path.write_text(json.dumps(payload, indent=2))

    fair_summary_path = reports_dir / "fairness_summary.csv"
    fairness_summary.to_csv(fair_summary_path, index=False)
    fair_full_path = reports_dir / "fairness_report.csv"
    fairness_full.to_csv(fair_full_path, index=False)

    return {
        "model_named": model_named_path,
        "model_default": model_default_path,
        "cv_summary": cv_path,
        "holdout_metrics": metrics_path,
        "fairness_summary": fair_summary_path,
        "fairness_full": fair_full_path,
    }


def run_full_pipeline(
    config: PipelineConfig = DEFAULT_CONFIG,
    quick: bool = False,
    include_xgb: bool = False,
) -> dict:
    """Esegue l'intera pipeline e restituisce un dizionario di risultati."""
    logger.info("=" * 70)
    logger.info("Avvio pipeline Hospital Readmission 30d (quick=%s)", quick)
    logger.info("=" * 70)

    # 1-2. Data preparation + group-aware split
    X_train, X_test, y_train, y_test, g_train, _g_test, df_clean = prepare_data(config)

    # 3. Pipeline candidate (con feature engineer in testa)
    fe = ReadmissionFeatureEngineer()
    X_train_fe = fe.fit_transform(X_train)
    pipelines = build_candidate_pipelines(X_train_fe, include_xgb=include_xgb)

    # 4. Tuning
    if quick:
        # Grid ridotte per smoke-test (eseguibile in <1 min su laptop).
        from . import config as cfg
        cfg.LOGREG_PARAM_GRID["model__C"] = [1.0]
        cfg.RF_PARAM_GRID["model__n_estimators"] = [100]
        cfg.RF_PARAM_GRID["model__max_depth"] = [12]
        cfg.RF_PARAM_GRID["model__min_samples_leaf"] = [1]
        cfg.RF_PARAM_GRID["model__max_features"] = ["sqrt"]

    tuning_results = tune_all_models(
        pipelines=pipelines, X=X_train, y=y_train, groups=g_train, config=config,
    )
    cv_summary = summarize_tuning(tuning_results)
    logger.info("\nRiepilogo tuning (CV AUC-PR):\n%s", cv_summary.to_string(index=False))

    # 5. Holdout evaluation con soglia di default 0.5
    holdout_metrics: dict[str, dict] = {}
    for name, result in tuning_results.items():
        m = evaluate_on_holdout(result.best_estimator, X_test, y_test, threshold=config.threshold)
        holdout_metrics[name] = m.as_dict()

    # 6. Selezione miglior modello
    best_name, best_result = select_best_model(tuning_results)
    logger.info("\n>>> Miglior modello: %s (AUC-PR CV = %.4f)\n",
                best_name, best_result.best_score)

    # Ottimizzazione soglia su holdout test (per costo asimmetrico).
    y_score = best_result.best_estimator.predict_proba(X_test)[:, 1]
    threshold_analysis = find_optimal_threshold(
        y_true=y_test.to_numpy(),
        y_score=y_score,
        cost_fn_over_fp=config.cost_fn_over_fp,
    )

    # Re-evaluation alla soglia ottima.
    m_opt = evaluate_on_holdout(
        best_result.best_estimator, X_test, y_test,
        threshold=threshold_analysis.optimal_threshold,
    )
    holdout_metrics[f"{best_name}_optimal_threshold"] = m_opt.as_dict()

    # 7. Fairness audit sul miglior modello (alla soglia ottima).
    y_pred_opt = (y_score >= threshold_analysis.optimal_threshold).astype(int)
    sensitive_holdout = df_clean.loc[X_test.index, list(SENSITIVE_FEATURES)]

    fairness_full = pd.DataFrame()
    fairness_summary = pd.DataFrame()
    try:
        rows = []
        for attr in SENSITIVE_FEATURES:
            if attr not in sensitive_holdout.columns:
                continue
            rep = fairness_report(
                y_true=y_test.values,
                y_pred=y_pred_opt,
                sensitive_features=sensitive_holdout[attr],
            )
            rep.insert(0, "attribute", attr)
            rows.append(rep)
        if rows:
            fairness_full = pd.concat(rows, ignore_index=True)
        sensitive_dict = {
            attr: sensitive_holdout[attr]
            for attr in SENSITIVE_FEATURES
            if attr in sensitive_holdout.columns
        }
        fairness_summary = aggregate_fairness_metrics(
            y_true=y_test.values,
            y_pred=y_pred_opt,
            sensitive_features_dict=sensitive_dict,
        )
        logger.info("\nFairness summary:\n%s", fairness_summary.to_string(index=False))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Fairness audit fallito: %s", exc)

    # 8. Persist
    artifacts = save_artifacts(
        best_name=best_name,
        best_estimator=best_result.best_estimator,
        holdout_metrics_by_model=holdout_metrics,
        cv_summary=cv_summary,
        fairness_summary=fairness_summary,
        fairness_full=fairness_full,
        optimal_threshold=threshold_analysis.optimal_threshold,
    )

    # Plot diagnostici (best-effort).
    try:
        from .config import FIGURES_DIR
        plot_roc_curve(
            y_test.values, y_score,
            title=f"{best_name}: ROC (test)",
            save_path=FIGURES_DIR / f"{best_name.lower()}_roc.png",
        )
        plot_precision_recall_curve(
            y_test.values, y_score,
            title=f"{best_name}: PR curve (test)",
            save_path=FIGURES_DIR / f"{best_name.lower()}_pr.png",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Plot diagnostici saltati: %s", exc)

    return {
        "best_model_name": best_name,
        "best_estimator": best_result.best_estimator,
        "cv_summary": cv_summary,
        "holdout_metrics": holdout_metrics,
        "threshold_analysis": threshold_analysis,
        "fairness_summary": fairness_summary,
        "fairness_full": fairness_full,
        "artifacts": artifacts,
        "tuning_results": tuning_results,
    }


# --- CLI ---

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hospital Readmission 30d — pipeline end-to-end.")
    parser.add_argument(
        "--quick", action="store_true",
        help="Tuning ridotto per smoke-test (<1 min).",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Directory di output (default: ./reports).",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=DEFAULT_CONFIG.cv_folds,
        help=f"Numero di fold per la CV. Default: {DEFAULT_CONFIG.cv_folds}.",
    )
    parser.add_argument(
        "--include-xgb", action="store_true",
        help="Includi XGBoost (richiede pip install -e .[advanced]).",
    )
    return parser


def main_train() -> int:
    """Entry point CLI per `readmit-train`."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    args = _build_arg_parser().parse_args()
    config = PipelineConfig(
        random_state=DEFAULT_CONFIG.random_state,
        test_size=DEFAULT_CONFIG.test_size,
        cv_folds=args.cv_folds,
        use_smote=DEFAULT_CONFIG.use_smote,
        class_weight_balanced=DEFAULT_CONFIG.class_weight_balanced,
        threshold=DEFAULT_CONFIG.threshold,
        cost_fn_over_fp=DEFAULT_CONFIG.cost_fn_over_fp,
        n_jobs=DEFAULT_CONFIG.n_jobs,
        verbose=DEFAULT_CONFIG.verbose,
    )
    run_full_pipeline(config=config, quick=args.quick, include_xgb=args.include_xgb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main_train())


__all__ = [
    "prepare_data",
    "build_candidate_pipelines",
    "select_best_model",
    "save_artifacts",
    "run_full_pipeline",
    "main_train",
]
