# 📚 Book Review Popularity — Interactive Demo

A Streamlit demo for the **UE24CS352A ML mini-project**
(*Text-Based Prediction of Book Review Popularity*, replication of Daly 2021, Stanford CS229).

Type any book review → the app engineers the paper's 12 features on the fly
(NLTK tokenize / POS-tag / lemmatize + VADER sentiment) → the **trained model
artifact from the training notebook** scores its probability of becoming popular
(like-share > 0.02).

## Files

| File | What it is |
|---|---|
| `demo_app.py` | The Streamlit demo (run this) |
| `demo_core.py` | Prediction logic: feature engineering + artifact loading (no Streamlit dependency) |
| `export_models.py` | Run in Colab after training to zip the `models/` folder for download |
| `requirements.txt` | Python dependencies |
| `models/` | **You create this** — drop the training notebook's `*.joblib` / `*.keras` files here |

## Setup

1. **Get the trained models.** In the Colab training notebook
   (`Notebooks/model_training .ipynb`), after training completes, run
   `export_models.py`. It zips the notebook's `models/` output
   (one `*.joblib` per LogisticRegression/XGBoost run, one `*.keras` per
   neural-net run — each `.joblib` already contains its fitted vectorizer/
   imputer/scaler, so nothing else is needed). Download the zip, unzip it into
   a `models/` folder next to `demo_app.py`:
   ```
   ML-MINI-Project_Demo/
   ├── demo_app.py
   ├── demo_core.py
   └── models/
       ├── D_XGBoost_undersampled.joblib
       ├── D_NeuralNetwork_undersampled.keras
       └── ...
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   (Add `tensorflow` too if you want to demo the NeuralNetwork runs.)

3. **Run the demo:**
   ```bash
   streamlit run demo_app.py
   ```

## Using the demo

- Pick any of the 18 training runs from the sidebar
  (defaults to `D_XGBoost_undersampled`, one of the paper's best runs).
- Paste a review, set the rating / reviewer details, hit **Predict popularity**.
- The verdict, P(popular), and the 12 computed features are shown; the decision
  threshold is adjustable in the sidebar.

## Notes

- Predictions reuse the **exact fitted preprocessing** stored inside each
  artifact, so demo scores match the evaluation pipeline.
- NLTK data (`punkt`, tagger, `wordnet`, `vader_lexicon`, `stopwords`) is
  downloaded automatically on first use.
- The demo expects the artifact schema written by `model_training .ipynb`
  (`feature_meta` with `kind: numeric|text`).
