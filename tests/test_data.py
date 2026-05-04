"""Smoke test su `readmit_pipeline.data`: missing handling + binarizzazione."""
from __future__ import annotations

import io

import pandas as pd

from readmit_pipeline.data import (
    basic_clean,
    binarize_target,
    drop_expired_patients,
    drop_unused_columns,
    missing_summary,
)


def _toy_df() -> pd.DataFrame:
    """DataFrame minimo che imita la struttura di Diabetes-130."""
    csv = io.StringIO(
        "encounter_id,patient_nbr,readmitted,weight,examide,citoglipton,"
        "discharge_disposition_id,race,age\n"
        "1,100,<30,?,No,No,1,Caucasian,[60-70)\n"
        "2,100,>30,?,No,No,1,Caucasian,[60-70)\n"
        "3,200,NO,?,No,No,11,AfricanAmerican,[70-80)\n"
        "4,300,<30,?,No,No,1,?,[50-60)\n"
    )
    df = pd.read_csv(csv, na_values=["?"])
    return df


def test_binarize_target_creates_readmitted_30d() -> None:
    df = _toy_df()
    out = binarize_target(df)
    assert "readmitted_30d" in out.columns
    assert out["readmitted_30d"].tolist() == [1, 0, 0, 1]
    # Immutability: input DataFrame non mutato.
    assert "readmitted_30d" not in df.columns


def test_drop_unused_removes_zero_variance_and_high_missing() -> None:
    df = _toy_df()
    out = drop_unused_columns(df)
    for col in ("weight", "examide", "citoglipton"):
        assert col not in out.columns


def test_drop_expired_patients_filters_discharge_id_11() -> None:
    df = binarize_target(_toy_df())
    out = drop_expired_patients(df)
    # Il record 3 ha discharge_disposition_id=11 (Expired) e va rimosso.
    assert len(out) == len(df) - 1
    assert 11 not in out["discharge_disposition_id"].values


def test_basic_clean_pipeline() -> None:
    df = _toy_df()
    out = basic_clean(df)
    assert "readmitted_30d" in out.columns
    assert "weight" not in out.columns
    # 4 record - 1 expired = 3.
    assert len(out) == 3


def test_missing_summary_returns_columns_with_missing() -> None:
    df = _toy_df()
    summary = missing_summary(df)
    # `weight` (4/4 missing) e `race` (1/4 missing) appaiono.
    assert "weight" in summary.index
    assert summary.loc["weight", "n_missing"] == 4
