"""Fairness audit con Fairlearn.

Calcola, per ogni attributo protetto (race, age, gender), le metriche
predittive disaggregate per sottogruppo + tre criteri di equita' classici:

- **Demographic Parity**: la probabilita' di essere classificati come
  positivi (selection rate) deve essere uguale fra gruppi.
- **Equalized Odds**: dato il vero stato (positivo/negativo), la
  probabilita' di essere classificati positivi deve essere uguale fra
  gruppi (TPR e FPR uguali).
- **Predictive Parity**: dato che il modello predice positivo, la
  probabilita' che sia veramente positivo (precision) deve essere uguale
  fra gruppi.

Riferimenti:
    - Hardt, Price, Srebro (2016). Equality of Opportunity in Supervised
      Learning. NeurIPS 29.
    - Chouldechova (2017). Fair prediction with disparate impact.
    - Fairlearn user guide: https://fairlearn.org

API:
    fairness_report(y_true, y_pred, sensitive_features) -> pd.DataFrame
    selection_rate_difference(...), equalized_odds_difference(...)
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _safe_import_fairlearn() -> Any:
    """Import lazy con messaggio chiaro se la dipendenza manca."""
    try:
        import fairlearn.metrics as flm  # noqa: WPS433
        return flm
    except ImportError as exc:
        raise ImportError(
            "fairlearn non installato. Esegui: pip install fairlearn>=0.10"
        ) from exc


def fairness_report(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    sensitive_features: pd.Series | pd.DataFrame,
    y_score: np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    """Calcola le metriche disaggregate per sottogruppo via Fairlearn.

    Args:
        y_true: ground truth binario (0/1).
        y_pred: predizioni binarie alla soglia di lavoro.
        sensitive_features: serie/dataframe con i valori dell'attributo
            protetto (es. `race`, oppure `df[['race', 'age']]`).
        y_score: probabilita' della classe positiva (opzionale; abilita
            AUC-ROC e AUC-PR per sottogruppo).

    Returns:
        DataFrame con una riga per (attributo, sottogruppo) e colonne:
            - count, base_rate, selection_rate
            - recall (TPR), specificity (TNR)
            - false_positive_rate (FPR), false_negative_rate (FNR)
            - precision (PPV)
    """
    flm = _safe_import_fairlearn()

    metrics = {
        "selection_rate": flm.selection_rate,
        "recall": flm.true_positive_rate,
        "false_positive_rate": flm.false_positive_rate,
        "false_negative_rate": flm.false_negative_rate,
    }

    mf = flm.MetricFrame(
        metrics=metrics,
        y_true=np.asarray(y_true),
        y_pred=np.asarray(y_pred),
        sensitive_features=sensitive_features,
    )
    by_group = mf.by_group.copy()

    # Aggiunge counts per gruppo (Fairlearn non li espone di default).
    if isinstance(sensitive_features, pd.DataFrame):
        # multi-attributo: count per tupla.
        sf_df = sensitive_features.copy()
        sf_df["__y_true"] = np.asarray(y_true)
        cols = list(sensitive_features.columns)
        counts = sf_df.groupby(cols).size().rename("count")
        base_rate = sf_df.groupby(cols)["__y_true"].mean().rename("base_rate")
        by_group = by_group.join(counts).join(base_rate)
    else:
        sf_series = pd.Series(np.asarray(sensitive_features), name="__sf")
        sf_df = pd.DataFrame({
            "__sf": sf_series,
            "__y_true": np.asarray(y_true),
        })
        counts = sf_df.groupby("__sf").size().rename("count")
        base_rate = sf_df.groupby("__sf")["__y_true"].mean().rename("base_rate")
        by_group = by_group.join(counts).join(base_rate)

    # Precision per gruppo (calcolata a mano: precision e' definita solo
    # quando esistono predizioni positive nel gruppo).
    by_group["precision"] = _precision_by_group(
        y_true, y_pred, sensitive_features
    )
    return by_group.reset_index()


def _precision_by_group(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    sensitive_features: pd.Series | pd.DataFrame,
) -> pd.Series:
    """Calcola precision per sottogruppo (gestendo gruppi senza positivi predetti)."""
    df = pd.DataFrame({
        "y_true": np.asarray(y_true),
        "y_pred": np.asarray(y_pred),
    })
    if isinstance(sensitive_features, pd.DataFrame):
        for c in sensitive_features.columns:
            df[c] = np.asarray(sensitive_features[c])
        keys = list(sensitive_features.columns)
    else:
        df["sf"] = np.asarray(sensitive_features)
        keys = ["sf"]

    def _grp_precision(g: pd.DataFrame) -> float:
        n_pos_pred = int((g["y_pred"] == 1).sum())
        if n_pos_pred == 0:
            return float("nan")
        n_tp = int(((g["y_pred"] == 1) & (g["y_true"] == 1)).sum())
        return n_tp / n_pos_pred

    return df.groupby(keys).apply(_grp_precision).rename("precision")


def selection_rate_difference(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    sensitive_features: pd.Series,
) -> float:
    """Demographic Parity gap: max - min selection rate fra gruppi.

    Valori vicini a 0 = parity. Valori > 0.10 = disparita' rilevante.
    """
    flm = _safe_import_fairlearn()
    return float(flm.demographic_parity_difference(
        y_true=np.asarray(y_true),
        y_pred=np.asarray(y_pred),
        sensitive_features=sensitive_features,
    ))


def equalized_odds_difference(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    sensitive_features: pd.Series,
) -> float:
    """Equalized Odds gap: max gap fra TPR e FPR fra gruppi.

    Definizione Fairlearn: max(TPR_diff, FPR_diff).
    """
    flm = _safe_import_fairlearn()
    return float(flm.equalized_odds_difference(
        y_true=np.asarray(y_true),
        y_pred=np.asarray(y_pred),
        sensitive_features=sensitive_features,
    ))


def aggregate_fairness_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    sensitive_features_dict: dict[str, pd.Series],
) -> pd.DataFrame:
    """Tabella aggregata: per ogni attributo, gap selection rate ed equalized odds.

    Args:
        sensitive_features_dict: mapping nome_attributo -> serie con i
            valori (es. {'race': df['race'], 'age': df['age']}).
    """
    rows: list[dict[str, Any]] = []
    for name, sf in sensitive_features_dict.items():
        try:
            dp_gap = selection_rate_difference(y_true, y_pred, sf)
            eo_gap = equalized_odds_difference(y_true, y_pred, sf)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Fairness su '%s' fallita: %s", name, exc)
            dp_gap, eo_gap = float("nan"), float("nan")
        rows.append({
            "attribute": name,
            "demographic_parity_gap": dp_gap,
            "equalized_odds_gap": eo_gap,
            "n_groups": int(pd.Series(sf).nunique(dropna=False)),
        })
    return pd.DataFrame(rows)


__all__ = [
    "fairness_report",
    "selection_rate_difference",
    "equalized_odds_difference",
    "aggregate_fairness_metrics",
]
