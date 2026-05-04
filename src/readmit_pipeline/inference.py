"""Inferenza end-to-end su nuovi dati clinici.

Espone `predict_readmission_risk(patient_record)` per stimare la probabilita'
di riammissione a 30 giorni da un singolo record (dict) o da un DataFrame.

Il modello viene caricato dal disco una sola volta tramite `lru_cache`
per evitare I/O ripetuti.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .config import DEFAULT_THRESHOLD, MODELS_DIR

logger = logging.getLogger(__name__)


DEFAULT_MODEL_PATH: Path = MODELS_DIR / "best_model.joblib"


@lru_cache(maxsize=4)
def _load_model(model_path: str) -> Any:
    """Carica un modello serializzato; cached per evitare I/O ripetuti."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Modello non trovato a {path}. "
            "Esegui prima la pipeline di training: `readmit-train`."
        )
    logger.info("Carico modello da %s", path)
    return joblib.load(path)


def _expected_columns(model: Any) -> list[str] | None:
    """Estrae i nomi colonna attesi dal modello serializzato.

    Il primo step della Pipeline (di solito `feature_engineer`) espone
    `feature_names_in_` come array dei nomi colonna pre-FE viste al fit.
    Per `predict_readmission_risk` usiamo questa lista per allineare
    l'input del chiamante.
    """
    try:
        return list(model.feature_names_in_)
    except AttributeError:
        return None


def _align_to_expected(df: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    """Riempie le colonne mancanti con NaN, scarta le extra, riordina."""
    return df.reindex(columns=expected)


def predict_readmission_risk(
    patient_record: dict[str, Any] | pd.DataFrame,
    model_path: Path = DEFAULT_MODEL_PATH,
    threshold: float = DEFAULT_THRESHOLD,
    return_label: bool = False,
) -> float | list[float] | tuple[float, int] | tuple[list[float], list[int]]:
    """Predice la probabilita' di riammissione a 30 giorni.

    Args:
        patient_record: dizionario con le feature di un singolo paziente
            (chiavi = nomi colonna del dataset Diabetes-130:
            'age', 'race', 'gender', 'time_in_hospital', 'num_medications',
            'A1Cresult', 'diag_1', ...), oppure `pd.DataFrame` per batch
            prediction.
        model_path: path del modello serializzato (joblib).
        threshold: soglia decisionale per la classe positiva.
        return_label: se True, restituisce anche la label binaria (0/1)
            calcolata alla soglia.

    Returns:
        - float (single) o list[float] (batch) se `return_label=False`.
        - (float, int) o (list[float], list[int]) se `return_label=True`.

    Note pratiche:
        - Le colonne MANCANTI nell'input vengono trattate come NaN dal
          preprocessor (imputazione automatica). L'utente puo' passare
          anche solo un sottoinsieme delle ~50 feature.
        - Le colonne EXTRA non presenti nel training vengono ignorate.
    """
    model = _load_model(str(model_path))

    if isinstance(patient_record, dict):
        df = pd.DataFrame([patient_record])
        is_single = True
    elif isinstance(patient_record, pd.DataFrame):
        df = patient_record.copy()
        is_single = False
    else:
        raise TypeError(
            "patient_record deve essere dict o DataFrame, ricevuto "
            f"{type(patient_record).__name__}."
        )

    expected = _expected_columns(model)
    if expected is not None:
        df = _align_to_expected(df, expected)

    proba = model.predict_proba(df)[:, 1]
    labels = (proba >= threshold).astype(int)

    if is_single:
        p = float(proba[0])
        if return_label:
            return p, int(labels[0])
        return p
    p_list = [float(x) for x in proba]
    if return_label:
        return p_list, [int(x) for x in labels]
    return p_list


def example_input() -> dict[str, Any]:
    """Esempio di input minimo per smoke-test della funzione di inferenza.

    Paziente fittizio con caratteristiche "tipiche" del dataset Strack 2014.
    """
    return {
        "race": "Caucasian",
        "gender": "Female",
        "age": "[60-70)",
        "admission_type_id": 1,
        "discharge_disposition_id": 1,
        "admission_source_id": 7,
        "time_in_hospital": 4,
        "num_lab_procedures": 41,
        "num_procedures": 0,
        "num_medications": 15,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 1,
        "number_diagnoses": 9,
        "max_glu_serum": "None",
        "A1Cresult": ">7",
        "diag_1": "428",
        "diag_2": "250.83",
        "diag_3": "401",
        "metformin": "No",
        "insulin": "Steady",
        "change": "No",
        "diabetesMed": "Yes",
    }


# --- CLI: readmit-predict ---

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predice la probabilita' di riammissione a 30 giorni "
                    "da un record paziente in JSON.",
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Path a un file JSON con il record paziente. "
             "Se assente, usa `example_input()`.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help=f"Path al modello serializzato. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Soglia decisionale. Default: {DEFAULT_THRESHOLD}",
    )
    return parser


def main_predict() -> int:
    """Entry point CLI per `readmit-predict`."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    args = _build_arg_parser().parse_args()

    if args.input:
        record = json.loads(Path(args.input).read_text())
    else:
        record = example_input()

    proba, label = predict_readmission_risk(
        record,
        model_path=Path(args.model),
        threshold=args.threshold,
        return_label=True,
    )
    out = {
        "readmission_probability": proba,
        "predicted_label": label,
        "threshold": args.threshold,
    }
    print(json.dumps(out, indent=2))
    return 0


__all__ = [
    "predict_readmission_risk",
    "example_input",
    "main_predict",
    "DEFAULT_MODEL_PATH",
]
