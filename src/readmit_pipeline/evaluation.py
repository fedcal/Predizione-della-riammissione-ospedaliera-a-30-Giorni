"""Metriche di classificazione binaria su dataset sbilanciato.

Si privilegiano metriche **orientate alla classe positiva** (~11% in
Diabetes-130):

- **AUC-PR (average_precision)**: area sotto la curva precision-recall.
  Robusta allo sbilanciamento, sensibile a miglioramenti sulla classe
  positiva.
- **AUC-ROC**: comparazione di "ranking quality" indipendente dalla
  soglia. Meno informativa di AUC-PR su sbilanciamento estremo.
- **Recall** sulla classe positiva: frazione di pazienti ad alto rischio
  effettivamente intercettati. Metrica primaria nel contesto clinico.
- **Precision** sulla classe positiva: frazione di "alert" effettivamente
  veri positivi. Trade-off contro recall.
- **F1, F-beta** (beta=2): combinazione armonica; F2 privilegia recall.

Accuracy e' deliberatamente NON inclusa come metrica primaria: con l'89%
di classe negativa, un classificatore costante "sempre negativo" otterrebbe
~89% accuracy.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


@dataclass
class ClassificationMetrics:
    """Insieme di metriche per classificazione binaria sbilanciata."""
    auc_roc: float
    auc_pr: float
    recall: float
    precision: float
    f1: float
    f2: float
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "auc_roc": self.auc_roc,
            "auc_pr": self.auc_pr,
            "recall": self.recall,
            "precision": self.precision,
            "f1": self.f1,
            "f2": self.f2,
            "threshold": self.threshold,
            "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
        }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> ClassificationMetrics:
    """Calcola tutte le metriche di interesse.

    Args:
        y_true: ground truth binario (0/1).
        y_pred: predizioni binarizzate (0/1) alla soglia `threshold`.
        y_score: probabilita' della classe positiva (per AUC-ROC/PR).
        threshold: soglia usata per `y_pred` (registrata, non riapplicata).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_score = np.asarray(y_score).astype(float)

    auc_roc = float(roc_auc_score(y_true, y_score))
    auc_pr = float(average_precision_score(y_true, y_score))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    pre = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    f2 = float(fbeta_score(y_true, y_pred, beta=2, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return ClassificationMetrics(
        auc_roc=auc_roc, auc_pr=auc_pr,
        recall=rec, precision=pre, f1=f1, f2=f2,
        threshold=float(threshold),
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
    )


def evaluate_on_holdout(
    fitted_estimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> ClassificationMetrics:
    """Calcola le metriche di un estimator gia' fittato sul holdout test set."""
    y_score = fitted_estimator.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= threshold).astype(int)
    metrics = compute_metrics(
        y_true=y_test.to_numpy(),
        y_pred=y_pred,
        y_score=y_score,
        threshold=threshold,
    )
    logger.info(
        "Holdout @t=%.2f: AUC-ROC=%.4f, AUC-PR=%.4f, recall=%.3f, "
        "precision=%.3f, F1=%.3f, F2=%.3f.",
        threshold, metrics.auc_roc, metrics.auc_pr,
        metrics.recall, metrics.precision, metrics.f1, metrics.f2,
    )
    return metrics


def compare_models(
    metrics_by_model: dict[str, ClassificationMetrics],
) -> pd.DataFrame:
    """Tabella comparativa modelli, ordinata per AUC-PR decrescente."""
    df = pd.DataFrame({
        name: m.as_dict() for name, m in metrics_by_model.items()
    }).T
    df.index.name = "model"
    if "auc_pr" in df.columns:
        df = df.sort_values("auc_pr", ascending=False)
    return df


# --- Diagnostica grafica ---

def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str = "Curva ROC",
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot ROC con AUC nel titolo."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str = "Curva Precision-Recall",
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot precision-recall con AUC-PR nel titolo."""
    p, r, _ = precision_recall_curve(y_true, y_score)
    auc_pr = average_precision_score(y_true, y_score)
    base = float(np.mean(y_true))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(r, p, label=f"AUC-PR = {auc_pr:.3f}")
    ax.axhline(base, color="k", linestyle="--", lw=1, alpha=0.5,
               label=f"baseline = {base:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
    return fig


def plot_confusion_matrix(
    metrics: ClassificationMetrics,
    title: str = "Confusion matrix",
    save_path: Path | None = None,
) -> plt.Figure:
    """Heatmap della confusion matrix (2x2)."""
    cm = np.array([[metrics.tn, metrics.fp], [metrics.fn, metrics.tp]])
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], ["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:d}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
    return fig


__all__ = [
    "ClassificationMetrics",
    "compute_metrics",
    "evaluate_on_holdout",
    "compare_models",
    "plot_roc_curve",
    "plot_precision_recall_curve",
    "plot_confusion_matrix",
]
