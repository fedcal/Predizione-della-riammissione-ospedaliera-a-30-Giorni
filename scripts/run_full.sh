#!/usr/bin/env bash
# Pipeline completa: training full + esecuzione notebook + smoke test.
# Uso: bash scripts/run_full.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Sanity check: il dataset deve essere presente in data/raw/.
if [[ ! -f "data/raw/diabetic_data.csv" ]]; then
  echo "[ERR] data/raw/diabetic_data.csv non trovato."
  echo "      Scaricalo dal UCI ML Repository (id 296):"
  echo "      https://archive.ics.uci.edu/dataset/296"
  echo "      Estrai diabetic_data.csv e IDS_mapping.csv in data/raw/."
  exit 1
fi

if [[ ! -d "venv" ]]; then
  echo "[setup] creo venv..."
  python3 -m venv venv
  # shellcheck disable=SC1091
  source venv/bin/activate
  pip install --upgrade pip --quiet
  pip install -e ".[notebooks]" --quiet
else
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

echo "[1/4] Generazione notebook didattici..."
python scripts/build_notebooks.py

echo "[2/4] Training pipeline completa (~5-15 min)..."
readmit-train

echo "[3/4] Esecuzione notebook end-to-end..."
for nb in notebooks/01_eda_demographics_target.ipynb \
          notebooks/02_preprocessing_icd9_grouping.ipynb \
          notebooks/03_models_logreg_vs_ensemble.ipynb \
          notebooks/04_fairness_audit.ipynb \
          notebooks/05_interpretability_and_errors.ipynb; do
  echo "    -> $nb"
  jupyter nbconvert --to notebook --execute --inplace \
      "$nb" --ExecutePreprocessor.timeout=1800 \
      --log-level=ERROR
done

echo "[4/4] Smoke test predict_readmission_risk()..."
python -c "
from readmit_pipeline.inference import predict_readmission_risk, example_input
proba, label = predict_readmission_risk(example_input(), return_label=True)
print(f'  Probabilita\\': {proba:.3f}, label predetta: {label}')
assert 0.0 <= proba <= 1.0, f'Probabilita\\' fuori range: {proba}'
print('  [OK] inference smoke test passed')
"

echo
echo "Pipeline completata. Vedi reports/ per i risultati."
