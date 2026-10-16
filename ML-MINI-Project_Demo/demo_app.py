"""Streamlit demo: predict whether a book review will become popular.

Replicates the Stanford CS229 paper (Daly 2021) pipeline:
type a review -> 12 NLTK features are engineered on the fly ->
the trained run artifact (vectorizer + model) from
``Notebooks/model_training .ipynb`` scores it.

Run with:
    streamlit run demo_app.py

Before the first run, copy the ``*.joblib`` files saved by the training
notebook (the ``models/`` output dir) into ``./models/`` next to this file —
see README.md / export_models.py.
"""

from pathlib import Path

import streamlit as st

import demo_core as core

st.set_page_config(page_title="Book Review Popularity — Demo",
                   page_icon="📚", layout="wide")

st.title("📚 Will this review become popular?")
st.markdown(
    "Replication of **Daly (2021), Stanford CS229** — *Text-Based Prediction of "
    "Book Review Popularity*. Type a review, set a few book/reviewer details, "
    "and the trained model scores its probability of becoming popular "
    "(like-share > 0.02)."
)

# ---------------------------------------------------------------- sidebar ---
st.sidebar.header("Model")
default_dir = str(Path(__file__).parent / "models")
models_dir = st.sidebar.text_input("Models folder", value=default_dir)

runs = core.discover_runs(models_dir)
keras_runs = []
if runs:
    keras_runs = [r for r in runs if r.startswith(("A_NeuralNetwork",
                                                   "B_NeuralNetwork",
                                                   "C_NeuralNetwork",
                                                   "D_NeuralNetwork"))]
    # joblib artifacts only exist for LogReg/XGBoost; NN runs ship as .keras
    from pathlib import Path as _P
    keras_runs = [r for r in keras_runs
                  if (_P(models_dir) / f"{r}.keras").exists()]

if not runs and not keras_runs:
    st.warning("No model artifacts found in the models folder.")
    st.markdown(
        """
        **To enable predictions:**
        1. In the Colab training notebook, run `export_models.py` (included here)
           after training — it zips the notebook's `models/` output.
        2. Unzip it so this folder looks like:
           ```
           ML-MINI-Project_Demo/
           ├── demo_app.py
           └── models/
               ├── D_XGBoost_undersampled.joblib
               ├── D_NeuralNetwork_undersampled.keras
               └── ...
           ```
        3. Refresh this page.
        """
    )
    st.stop()

options = runs + [r for r in keras_runs if r not in runs]
# Prefer the paper's best run when available.
default_idx = 0
for i, r in enumerate(options):
    if r == "D_XGBoost_undersampled":
        default_idx = i
        break
run_name = st.sidebar.selectbox("Training run", options, index=default_idx)
threshold = st.sidebar.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05)
st.sidebar.caption("A review is labelled *popular* when P(popular) ≥ threshold.")

# ------------------------------------------------------------- load model ---
is_keras = run_name.endswith("keras") or (
    "NeuralNetwork" in run_name and not (Path(models_dir) / f"{run_name}.joblib").exists())

try:
    if is_keras:
        base = run_name[:-6] if run_name.endswith(".keras") else run_name
        subset = base.split("_")[0]
        by_subset = {}
        for r in runs:
            art = core.load_artifact(models_dir, r)
            by_subset.setdefault(art.get("subset", r.split("_")[0]), art)
        keras_model, feature_meta = core.load_keras_model(
            models_dir, base, subset, by_subset)
        artifact = {"subset": subset, "model_name": "NeuralNetwork",
                    "undersampled": base.endswith("undersampled"),
                    "threshold": 0.5}
        use_keras = True
    else:
        artifact = core.load_artifact(models_dir, run_name)
        use_keras = False
except Exception as exc:  # noqa: BLE001 — show, don't crash
    st.error(f"Could not load `{run_name}`: {exc}")
    st.stop()

st.sidebar.markdown(
    f"**Subset:** {artifact.get('subset')} &nbsp;|&nbsp; "
    f"**Model:** {artifact.get('model_name')} &nbsp;|&nbsp; "
    f"**Training:** {'undersampled' if artifact.get('undersampled') else 'full sample'}"
)

# ------------------------------------------------------------------ input ---
col1, col2 = st.columns([3, 2])

with col1:
    review_text = st.text_area(
        "Review text",
        height=220,
        placeholder="Paste a book review here — e.g. “An absolutely stunning debut. "
                    "The prose sings on every page and the ending stayed with me for days…”",
    )

with col2:
    user_rating = st.slider("Your rating (stars)", 1, 5, 5)
    avg_rating = st.number_input("Book's average rating", 1.0, 5.0, 4.0, 0.1)
    user_reviews = st.number_input("Reviews written by you", 1, 100000, 12)
    days_since = st.number_input("Days since you posted it", 0, 3650, 30)

go = st.button("🔮 Predict popularity", type="primary", use_container_width=False)

if not core.nltk_resources_ok():
    st.warning(
        "NLTK language data is not fully downloaded in this environment — "
        "predictions will use degraded fallbacks. Install it once with:\n"
        "`python -c \"import nltk; [nltk.download(p) for p in "
        "['punkt','averaged_perceptron_tagger_eng','wordnet','omw-1.4',"
        "'vader_lexicon','stopwords']]\"`"
    )

# ---------------------------------------------------------------- predict ---
if go:
    if not review_text.strip():
        st.warning("Please paste a review first.")
        st.stop()
    with st.spinner("Engineering features & scoring…"):
        feats = core.engineer_review(review_text, user_rating, avg_rating,
                                     user_reviews, days_since)
        if use_keras:
            proba, _, label = core.predict_keras(keras_model, feature_meta,
                                                 feats, threshold)
        else:
            proba, _, label = core.predict(artifact, feats, threshold)

    st.divider()
    m1, m2 = st.columns(2)
    with m1:
        if label == "POPULAR":
            st.success(f"## ✅ {label}")
        else:
            st.info(f"## ➖ {label}")
    with m2:
        st.metric("P(popular)", f"{proba:.1%}")
    st.progress(proba, text=f"Model confidence: {proba:.1%}")

    with st.expander("🔍 Features the model saw (12 paper features)"):
        st.json({k: (round(v, 4) if isinstance(v, float) else v)
                 for k, v in feats.items() if k != "tokenized_words"})
        st.caption("`tokenized_words` (lemmatized, stopwords removed) feeds the "
                   "bag-of-words / TF-IDF vectorizer for subsets C and D.")

    st.caption(
        "Prediction uses the fitted preprocessing (imputer / scaler / vectorizer) "
        "stored inside the training artifact — identical to the evaluation pipeline."
    )
else:
    st.info("👆 Paste a review and hit **Predict popularity** to score it.")
