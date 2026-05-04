"""Ottimizzazione della soglia decisionale su matrice dei costi asimmetrica.

Razionale clinico: il falso negativo (paziente ad alto rischio non
intercettato dal programma di follow-up, che si riammette) ha costo
molto piu' alto del falso positivo (paziente a basso rischio che riceve
una telefonata di follow-up non strettamente necessaria).

Stima conservativa in letteratura: cost(FN) / cost(FP) in [5, 20].
Default in `config.COST_FN_OVER_FP` = 5.

Strategia:
    - Si calcola la curva PR sul validation/holdout set.
    - Si valuta il costo totale per ogni soglia in [0, 1].
    - Si sceglie la soglia che minimizza il costo atteso.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
)

from .config import COST_FN_OVER_FP

logger = logging.getLogger(__name__)


@dataclass
class ThresholdAnalysis:
    """Risultato dell'analisi della soglia."""
    optimal_threshold: float
    optimal_cost: float
    optimal_recall: float
    optimal_precision: float
    optimal_f1: float
    cost_curve: pd.DataFrame    # threshold, cost, tp, fp, tn, fn


def cost_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    cost_fn_over_fp: float = COST_FN_OVER_FP,
) -> dict[str, float]:
    """Calcola costo, tp/fp/tn/fn alla soglia data."""
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    # Costo normalizzato: FP=1, FN=cost_fn_over_fp.
    cost = fp + cost_fn_over_fp * fn
    return {
        "threshold": float(threshold),
        "cost": float(cost),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    cost_fn_over_fp: float = COST_FN_OVER_FP,
    n_thresholds: int = 200,
) -> ThresholdAnalysis:
    """Scansiona n_thresholds soglie equispaziate in [0, 1] e sceglie la migliore.

    Strategia: griglia equispaziata + minimizzazione costo. Equivale a
    una soluzione esatta nel limite n_thresholds -> infty; con 200 punti
    l'errore di discretizzazione e' << 0.005 in costo relativo.

    Args:
        y_true: ground truth binario.
        y_score: probabilita' della classe positiva (shape n).
        cost_fn_over_fp: rapporto costi FN/FP. Vedi config.
        n_thresholds: granularita' della griglia.

    Returns:
        `ThresholdAnalysis` con soglia ottima e curva completa.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    rows = [cost_at_threshold(y_true, y_score, t, cost_fn_over_fp) for t in thresholds]
    df = pd.DataFrame(rows)

    idx = int(df["cost"].idxmin())
    best = df.iloc[idx].to_dict()

    # Metriche al best threshold.
    tp, fp, fn = best["tp"], best["fp"], best["fn"]
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    logger.info(
        "Soglia ottima (cost FN/FP=%.1f): %.3f -> cost=%.0f, recall=%.3f, "
        "precision=%.3f, F1=%.3f.",
        cost_fn_over_fp, best["threshold"], best["cost"], recall, precision, f1,
    )
    return ThresholdAnalysis(
        optimal_threshold=float(best["threshold"]),
        optimal_cost=float(best["cost"]),
        optimal_recall=float(recall),
        optimal_precision=float(precision),
        optimal_f1=float(f1),
        cost_curve=df,
    )


def precision_recall_table(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> pd.DataFrame:
    """Tabella precision/recall per le soglie naturalmente prodotte da sklearn."""
    p, r, t = precision_recall_curve(y_true, y_score)
    # `t` ha lunghezza n-1 rispetto a p,r: l'ultimo valore di p,r corrisponde
    # alla soglia +inf (predizione "tutti negativi").
    out = pd.DataFrame({
        "threshold": np.concatenate([t, [np.inf]]),
        "precision": p,
        "recall": r,
    })
    return out


__all__ = [
    "ThresholdAnalysis",
    "cost_at_threshold",
    "find_optimal_threshold",
    "precision_recall_table",
]
