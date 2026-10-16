"""Core prediction logic for the Goodreads review-popularity demo.

Loads a training-run artifact saved by ``Notebooks/model_training .ipynb``
(``<run_name>.joblib``) and predicts popularity for a single raw review.

The artifact dict schema (written by the training notebook):
    {'model': fitted sklearn/XGB model,
     'subset': 'A'|'B'|'C'|'D',
     'model_name': 'LogisticRegression'|'XGBoost'|'NeuralNetwork',
     'undersampled': bool,
     'feature_meta': {'kind': 'numeric', 'imputer': ..., 'columns': [...]}
                  or {'kind': 'text', 'transformer': ColumnTransformer,
                      'columns': [...], 'text_column': 'tokenized_words'},
     'threshold': 0.5, ...}
"""

from pathlib import Path

import joblib
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer

# ----------------------------------------------------------------------------
# NLTK resources
# ----------------------------------------------------------------------------

_NLTK_READY = False


def ensure_nltk():
    """Download the NLTK resources used by the paper's feature pipeline."""
    global _NLTK_READY
    if _NLTK_READY:
        return
    for pkg in ["punkt", "punkt_tab", "averaged_perceptron_tagger",
                "averaged_perceptron_tagger_eng", "wordnet", "omw-1.4",
                "vader_lexicon", "stopwords"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    _NLTK_READY = True


def _word_tokenize(text):
    try:
        return nltk.word_tokenize(text)
    except LookupError:
        return text.split()


def _sent_tokenize(text):
    try:
        return nltk.sent_tokenize(text)
    except LookupError:
        return [text] if text.strip() else []


def nltk_resources_ok():
    """True when the key NLTK resources are available locally."""
    checks = [("tokenizers/punkt",), ("taggers/averaged_perceptron_tagger_eng",),
              ("taggers/averaged_perceptron_tagger",), ("corpora/wordnet",),
              ("sentiment/vader_lexicon",), ("corpora/stopwords",)]
    ok_tok = any(_exists(p[0]) for p in checks[:1])
    ok_tag = any(_exists(p[0]) for p in checks[1:3])
    return ok_tok and ok_tag and all(_exists(p[0]) for p in checks[3:])


def _exists(resource_path):
    try:
        nltk.data.find(resource_path)
        return True
    except LookupError:
        return False


# ----------------------------------------------------------------------------
# Feature engineering — mirrors engineer_one_review() in the training repo
# ----------------------------------------------------------------------------

A_COLS = ["user_reviews", "days_since_review", "user_rating", "rating_diff"]
B_EXTRA_COLS = ["num_words", "avg_word_len", "avg_sent_len",
                "pct_verbs", "pct_nouns", "pct_adj", "quote", "sentiment"]
B_COLS = A_COLS + B_EXTRA_COLS
TEXT_COL = "tokenized_words"


def engineer_review(review_text, user_rating, avg_rating,
                    user_reviews=1, days_since_review=0):
    """Compute the 12 paper features + tokenized text for one raw review."""
    ensure_nltk()
    raw = "" if review_text is None else str(review_text)
    sentences = _sent_tokenize(raw)
    words = _word_tokenize(raw)

    try:
        pos_tags = nltk.pos_tag(words) if words else []
    except LookupError:
        pos_tags = [(w, "NN") for w in words]

    n_sent = len(sentences)
    n_words = len(words)
    n_letters = sum(len(w) for w in words)

    def count_prefix(prefixes):
        return sum(1 for _, tag in pos_tags if tag and tag[0] in prefixes)

    verbs = count_prefix(["V"])
    nouns = count_prefix(["N"])
    adj_adv = count_prefix(["J", "R"])  # paper groups adjectives+adverbs

    try:
        sid = SentimentIntensityAnalyzer()
        sentiment = float(np.mean([sid.polarity_scores(s)["compound"]
                                   for s in sentences])) if sentences else 0.0
    except LookupError:
        sentiment = 0.0

    try:
        stop = set(stopwords.words("english"))
    except LookupError:
        stop = set()
    lemmatizer = WordNetLemmatizer()
    clean_words = [w for w in words if w.lower() not in stop]
    try:
        clean_words = [lemmatizer.lemmatize(w) for w in clean_words]
    except LookupError:
        pass

    try:
        rating_diff = float(user_rating) - float(avg_rating)
    except (TypeError, ValueError):
        rating_diff = float("nan")

    return {
        "user_reviews": float(user_reviews),
        "days_since_review": float(days_since_review),
        "user_rating": float(user_rating),
        "rating_diff": rating_diff,
        "num_words": n_words,
        "avg_word_len": (n_letters / n_words) if n_words else 0.0,
        "avg_sent_len": (n_words / n_sent) if n_sent else 0.0,
        "pct_verbs": (verbs / n_words) if n_words else 0.0,
        "pct_nouns": (nouns / n_words) if n_words else 0.0,
        "pct_adj": (adj_adv / n_words) if n_words else 0.0,
        "quote": int('"' in raw),
        "sentiment": sentiment,
        "tokenized_words": " ".join(clean_words),
    }


# ----------------------------------------------------------------------------
# Artifacts & prediction
# ----------------------------------------------------------------------------

def discover_runs(models_dir):
    """Return sorted run names for every ``<run>.joblib`` artifact found."""
    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        return []
    return sorted(p.stem for p in models_dir.glob("*.joblib"))


def load_artifact(models_dir, run_name):
    path = Path(models_dir) / f"{run_name}.joblib"
    return joblib.load(path)


def build_matrix(artifact, features):
    """Apply the artifact's stored transformer to one review's features."""
    meta = artifact["feature_meta"]
    if meta["kind"] == "numeric":
        df = pd.DataFrame([{c: features[c] for c in meta["columns"]}])
        return meta["imputer"].transform(df)
    if meta["kind"] == "text":
        row = {meta["text_column"]: features["tokenized_words"]}
        row.update({c: features[c] for c in meta["columns"]})
        df = pd.DataFrame([row])
        return meta["transformer"].transform(df)
    raise ValueError(f"unknown feature_meta kind: {meta.get('kind')}")


def predict_proba(artifact, features):
    """Return P(popular) for one engineered review."""
    X = build_matrix(artifact, features)
    return float(artifact["model"].predict_proba(X)[0, 1])


def predict(artifact, features, threshold=None):
    """Return (probability, label, label_text) for one engineered review."""
    if threshold is None:
        threshold = float(artifact.get("threshold", 0.5))
    proba = predict_proba(artifact, features)
    is_popular = proba >= threshold
    return proba, is_popular, ("POPULAR" if is_popular else "not popular")


def load_keras_model(models_dir, run_name, subset, artifacts_by_subset):
    """Optionally load a ``<run>.keras`` NN, reusing a same-subset transformer.

    The training notebook saves NN weights as ``<run>.keras`` without the
    feature meta, so we borrow the fitted transformer from any joblib
    artifact of the same subset (the transformer is subset-specific, not
    model-specific).
    """
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("tensorflow is required for NeuralNetwork runs") from exc
    keras_path = Path(models_dir) / f"{run_name}.keras"
    if not keras_path.exists():
        raise FileNotFoundError(f"{keras_path} not found")
    donor = artifacts_by_subset.get(subset)
    if donor is None:
        raise RuntimeError(
            f"need a .joblib artifact for subset {subset} to supply the transformer")
    model = tf.keras.models.load_model(keras_path)
    return model, donor["feature_meta"]


def predict_keras(keras_model, feature_meta, features, threshold=0.5):
    X = build_matrix({"feature_meta": feature_meta}, features)
    if hasattr(X, "toarray"):
        X = X.toarray()
    proba = float(keras_model.predict(np.asarray(X, dtype=np.float32), verbose=0)[0, 0])
    is_popular = proba >= threshold
    return proba, is_popular, ("POPULAR" if is_popular else "not popular")
