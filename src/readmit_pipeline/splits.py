"""Split train/test e cross-validation **group-aware**.

Lo stesso `patient_nbr` puo' comparire piu' volte nel dataset (lo stesso
paziente con piu' ricoveri). Lasciar finire ricoveri dello stesso paziente
sia in train sia in test introduce **leakage temporale**: il modello
"riconosce" il paziente invece di generalizzare.

Soluzione: `GroupShuffleSplit` per l'holdout iniziale e `GroupKFold` per
la cross-validation. Entrambe garantiscono che `groups` (= `patient_nbr`)
non si sovrapponga fra fold.

Stratificazione: `GroupKFold` di sklearn non supporta nativamente la
stratificazione. Usiamo `StratifiedGroupKFold` (sklearn>=1.0) per avere
fold con percentuali simili di classe positiva (~11%).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
)

from .config import DEFAULT_CONFIG, PipelineConfig

logger = logging.getLogger(__name__)


def group_aware_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Split holdout train/test in cui `groups` (patient_nbr) NON si sovrappone.

    Args:
        X: feature.
        y: target binario.
        groups: serie di identificatori paziente (un valore per riga di X).
        config: configurazione (test_size, random_state).

    Returns:
        (X_train, X_test, y_train, y_test, groups_train, groups_test).

    Raises:
        ValueError: se `len(X) != len(y) != len(groups)`.

    Note:
        `GroupShuffleSplit` non supporta stratify nativamente: il
        bilanciamento del target nei due split puo' variare (~1-2% in
        pratica). Per avere split stratificati e group-aware insieme
        si puo' usare `StratifiedGroupKFold` con `n_splits=int(1/test_size)`
        e prendere il primo fold; lasciato come estensione.
    """
    if not (len(X) == len(y) == len(groups)):
        raise ValueError(
            f"Lunghezze incoerenti: X={len(X)}, y={len(y)}, groups={len(groups)}"
        )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    (train_idx, test_idx), = splitter.split(X, y, groups=groups)

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()
    g_train = groups.iloc[train_idx].copy()
    g_test = groups.iloc[test_idx].copy()

    overlap = set(g_train.unique()) & set(g_test.unique())
    if overlap:
        # Difensivo: GroupShuffleSplit non dovrebbe mai produrre overlap.
        # Se succede, il dataset ha invariante violato (es. NaN in groups).
        raise RuntimeError(
            f"Group-aware split fallito: {len(overlap)} pazienti in sovrapposizione."
        )

    pos_train = float(y_train.mean())
    pos_test = float(y_test.mean())
    logger.info(
        "Split group-aware: train=%d (pos=%.2f%%), test=%d (pos=%.2f%%). "
        "Pazienti unici: %d train, %d test, 0 overlap.",
        len(X_train), pos_train * 100,
        len(X_test), pos_test * 100,
        g_train.nunique(), g_test.nunique(),
    )
    return X_train, X_test, y_train, y_test, g_train, g_test


def make_group_cv(config: PipelineConfig = DEFAULT_CONFIG) -> StratifiedGroupKFold:
    """Cross-validation stratificata e group-aware.

    `StratifiedGroupKFold` (sklearn>=1.0):
        - Bilancia la classe positiva fra fold (stratificazione).
        - Garantisce che lo stesso `patient_nbr` non finisca in fold
          diversi (group-aware).

    Use:
        cv = make_group_cv(config)
        for train_idx, val_idx in cv.split(X, y, groups=groups):
            ...
    """
    return StratifiedGroupKFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )


def assert_no_group_leakage(
    groups_train: pd.Series,
    groups_test: pd.Series,
) -> None:
    """Sanity check: solleva AssertionError se c'e' leakage.

    Use in test e all'inizio della pipeline come safety net.
    """
    train_set = set(np.asarray(groups_train).tolist())
    test_set = set(np.asarray(groups_test).tolist())
    overlap = train_set & test_set
    assert not overlap, (
        f"Group leakage rilevato: {len(overlap)} pazienti in entrambi gli split."
    )


__all__ = [
    "group_aware_train_test_split",
    "make_group_cv",
    "assert_no_group_leakage",
]
