"""Pipeline ML + audit di equità per la predizione della riammissione
ospedaliera a 30 giorni su pazienti diabetici (UCI 296, Diabetes 130-US
Hospitals for Years 1999-2008).

API pubbliche principali:

    from readmit_pipeline.pipeline import run_full_pipeline
    from readmit_pipeline.inference import predict_readmission_risk
    from readmit_pipeline.fairness import fairness_report

Per il dettaglio del flusso e delle scelte tecniche, vedi
`docs/scelte_tecniche/architettura.md` e i notebook in `notebooks/`.
"""
from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
