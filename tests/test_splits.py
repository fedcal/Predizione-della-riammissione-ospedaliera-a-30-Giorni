"""Smoke test su `splits`: garantire che lo stesso patient_nbr non finisca
sia in train sia in test (no leakage)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from readmit_pipeline.config import PipelineConfig
from readmit_pipeline.splits import (
    assert_no_group_leakage,
    group_aware_train_test_split,
    make_group_cv,
)


def _make_synthetic(n_patients: int = 100, encounters_per_patient: int = 3):
    """Crea (X, y, groups) con piu' encounter per paziente."""
    rng = np.random.default_rng(0)
    patient_nbr = np.repeat(np.arange(n_patients), encounters_per_patient)
    n = len(patient_nbr)
    y = rng.integers(0, 2, size=n)
    X = pd.DataFrame({
        "feat_num": rng.normal(size=n),
        "feat_cat": rng.choice(["a", "b", "c"], size=n),
    })
    return X, pd.Series(y, name="y"), pd.Series(patient_nbr, name="patient_nbr")


def test_group_aware_split_zero_overlap() -> None:
    X, y, groups = _make_synthetic(n_patients=50, encounters_per_patient=4)
    cfg = PipelineConfig(test_size=0.2, random_state=42)
    Xtr, Xte, ytr, yte, gtr, gte = group_aware_train_test_split(X, y, groups, cfg)

    # Nessun paziente in entrambi gli split.
    overlap = set(gtr) & set(gte)
    assert overlap == set(), f"Group leakage: {len(overlap)} pazienti in entrambi."

    # Le shape devono sommare al totale.
    assert len(Xtr) + len(Xte) == len(X)
    assert len(ytr) + len(yte) == len(y)


def test_assert_no_group_leakage_passes_when_disjoint() -> None:
    a = pd.Series([1, 2, 3])
    b = pd.Series([4, 5, 6])
    # Non solleva.
    assert_no_group_leakage(a, b)


def test_assert_no_group_leakage_fails_on_overlap() -> None:
    a = pd.Series([1, 2, 3])
    b = pd.Series([3, 4, 5])
    with pytest.raises(AssertionError):
        assert_no_group_leakage(a, b)


def test_group_cv_no_overlap_per_fold() -> None:
    """Verifica che StratifiedGroupKFold non sovrapponga i gruppi fra train e val di ogni fold."""
    X, y, groups = _make_synthetic(n_patients=80, encounters_per_patient=3)
    cfg = PipelineConfig(cv_folds=3, random_state=0)
    cv = make_group_cv(cfg)
    for train_idx, val_idx in cv.split(X, y, groups=groups):
        g_tr = set(groups.iloc[train_idx])
        g_va = set(groups.iloc[val_idx])
        assert g_tr.isdisjoint(g_va), "Group leakage in fold CV"


def test_split_raises_on_length_mismatch() -> None:
    X, y, groups = _make_synthetic(n_patients=10, encounters_per_patient=2)
    bad_groups = groups.iloc[:5]
    cfg = PipelineConfig(test_size=0.2, random_state=0)
    with pytest.raises(ValueError):
        group_aware_train_test_split(X, y, bad_groups, cfg)
