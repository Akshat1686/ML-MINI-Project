"""Run this in Colab AFTER the training notebook finishes, to fetch the models.

The training notebook saves one artifact per run into ``MODEL_DIR``:
  - ``<run_name>.joblib``  for LogisticRegression / XGBoost
    (e.g. ``D_XGBoost_undersampled.joblib``) — contains the fitted model AND
    the fitted preprocessing (imputer / scaler / vectorizer), so the demo
    needs nothing else.
  - ``<run_name>.keras``   for NeuralNetwork runs.

This script zips that folder so you can download it in one click, then unzip
it into ``ML-MINI-Project_Demo/models/`` next to ``demo_app.py``.
"""

import shutil
from pathlib import Path

try:
    MODEL_DIR  # noqa: F821 — defined in the training notebook
except NameError:
    MODEL_DIR = Path("/content/drive/MyDrive/goodreads_reproduction_results/models")

MODEL_DIR = Path(MODEL_DIR)
assert MODEL_DIR.is_dir(), f"models folder not found: {MODEL_DIR}"

zip_path = shutil.make_archive("/content/goodreads_models", "zip", MODEL_DIR)
print(f"zipped {len(list(MODEL_DIR.glob('*')))} files -> {zip_path}")

try:
    from google.colab import files  # noqa
    files.download(zip_path)
    print("download started — unzip into ML-MINI-Project_Demo/models/")
except ImportError:
    print("not in Colab; copy the zip manually:")
    print("  ", zip_path)
