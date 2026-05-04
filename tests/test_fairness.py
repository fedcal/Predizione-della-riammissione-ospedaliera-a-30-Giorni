"""Smoke test sui calcoli di fairness su mini-DataFrame (richiede fairlearn).

Se fairlearn non e' installato, i test sono skippati (xfail) ma non
fanno crashare la suite.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Skip dell'intero modulo se fairlearn manca.
fairlearn = pytest.importorskip("fairlearn")

from readmit_pipeline.fairness import (  # noqa: E402
    aggregate_fairness_metrics,
    equalized_odds_difference,
    fairness_report,
    selection_rate_difference,
)


def _toy_dataset() -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """Mini-dataset con una disparita' artificiale per controllare i calcoli.

    Gruppo A (10 elementi): 5 positivi, modello predice 5 positivi tutti corretti.
    Gruppo B (10 elementi): 5 positivi, modello predice 2 positivi (3 falsi negativi).
    Risultato atteso: equalized_odds_gap > 0.1, selection_rate gap > 0.
    """
    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                       1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                       1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    sf = pd.Series(["A"] * 10 + ["B"] * 10, name="group")
    return y_true, y_pred, sf


def test_fairness_report_returns_per_group_rows() -> None:
    y_true, y_pred, sf = _toy_dataset()
    report = fairness_report(y_true, y_pred, sf)
    # Dovrebbe avere una riga per gruppo (A, B).
    assert len(report) == 2
    # Colonne attese.
    expected = {"selection_rate", "recall", "precision",
                "false_positive_rate", "false_negative_rate",
                "count", "base_rate"}
    assert expected.issubset(set(report.columns))


def test_selection_rate_difference_positive_when_disparity() -> None:
    y_true, y_pred, sf = _toy_dataset()
    gap = selection_rate_difference(y_true, y_pred, sf)
    # Gruppo A predice 5/10=0.5, gruppo B predice 2/10=0.2 -> gap = 0.3.
    assert gap == pytest.approx(0.3, abs=0.01)


def test_equalized_odds_difference_positive_when_tpr_diverges() -> None:
    y_true, y_pred, sf = _toy_dataset()
    gap = equalized_odds_difference(y_true, y_pred, sf)
    # TPR(A)=1.0, TPR(B)=0.4 -> diff=0.6, FPR(A)=FPR(B)=0 -> max=0.6.
    assert gap == pytest.approx(0.6, abs=0.01)


def test_aggregate_fairness_metrics_handles_multi_attribute() -> None:
    y_true, y_pred, sf = _toy_dataset()
    sf2 = pd.Series(["X"] * 5 + ["Y"] * 15, name="other")
    table = aggregate_fairness_metrics(
        y_true=y_true, y_pred=y_pred,
        sensitive_features_dict={"group": sf, "other": sf2},
    )
    assert set(table["attribute"]) == {"group", "other"}
    assert "demographic_parity_gap" in table.columns
    assert "equalized_odds_gap" in table.columns


def test_fairness_report_with_perfect_model_has_zero_gaps() -> None:
    """Modello perfetto -> gap = 0 sui calcoli."""
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = y_true.copy()
    sf = pd.Series(["A", "A", "A", "A", "B", "B", "B", "B"], name="g")
    gap = equalized_odds_difference(y_true, y_pred, sf)
    assert gap == pytest.approx(0.0, abs=1e-9)
