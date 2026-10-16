import re
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

st.set_page_config(page_title="Goodreads Review Popularity", page_icon="📚")
st.title("Goodreads Review Popularity")
st.caption("Demo for the Stanford paper reproduction. Predictions are experimental, not guarantees.")

@st.cache_resource
def load_artifact():
    return joblib.load("best_sklearn_demo_model.joblib")

@st.cache_resource
def load_nlp():
    for path, pkg in [
        ("tokenizers/punkt", "punkt"), ("tokenizers/punkt_tab", "punkt_tab"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ("corpora/stopwords", "stopwords"), ("corpora/wordnet", "wordnet"),
        ("sentiment/vader_lexicon", "vader_lexicon")
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)
    return SentimentIntensityAnalyzer(), WordNetLemmatizer(), set(stopwords.words("english"))

def features(text):
    sid, lem, stops = load_nlp()
    try:
        sentences = nltk.sent_tokenize(text) if text.strip() else []
    except LookupError:
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    try:
        raw_tokens = nltk.word_tokenize(text) if text.strip() else []
    except LookupError:
        raw_tokens = nltk.tokenize.TreebankWordTokenizer().tokenize(text)
    words = [w.lower() for w in raw_tokens if w.isalpha()]
    try:
        tags = nltk.pos_tag(words) if words else []
    except LookupError:
        tags = [(w, "NN") for w in words]
    nwords, nsent = len(words), len(sentences)
    nletters = sum(len(w) for w in words)
    pct = lambda prefixes: sum(1 for _, tag in tags if tag and tag[0] in prefixes) / nwords if nwords else 0.0
    sentiment = float(np.mean([sid.polarity_scores(s)["compound"] for s in sentences])) if sentences else 0.0
    clean = [lem.lemmatize(w) for w in words if w not in stops]
    return {
        "rating_diff": float(user_rating - avg_rating),
        "quote": int('"' in text),
        "num_words": nwords,
        "avg_sent_len": nwords / nsent if nsent else 0.0,
        "avg_word_len": nletters / nwords if nwords else 0.0,
        "pct_verbs": pct(["V"]), "pct_nouns": pct(["N"]), "pct_adj": pct(["J", "R"]),
        "sentiment": sentiment, "tokenized_words": " ".join(clean)
    }

artifact = load_artifact()
model = artifact["model"]
subset = artifact["subset"]
meta = artifact["feature_meta"]

review_text = st.text_area("Review text", "The characters were compelling and the story was enjoyable.")
c1, c2 = st.columns(2)
with c1:
    user_reviews = st.number_input("Reviews written by user", min_value=0, value=10)
    days_since_review = st.number_input("Days since review posted", min_value=0, value=30)
with c2:
    user_rating = st.slider("User rating", min_value=1.0, max_value=5.0, value=4.0, step=0.5)
    avg_rating = st.slider("Average book rating", min_value=1.0, max_value=5.0, value=3.8, step=0.1)

if st.button("Predict popularity", type="primary"):
    f = features(review_text)
    row = {
        "user_reviews": user_reviews, "days_since_review": days_since_review,
        "user_rating": user_rating, "avg_rating": avg_rating,
        **f
    }
    frame = pd.DataFrame([row])
    if meta["kind"] == "numeric":
        X = meta["imputer"].transform(frame[meta["columns"]])
    else:
        X = meta["transformer"].transform(frame[[meta["text_column"]] + meta["columns"]])
    probability = float(model.predict_proba(X)[:, 1][0])
    st.metric("Predicted probability of popular", f"{probability:.1%}")
    st.write("Predicted class:", "Popular" if probability >= artifact.get("threshold", 0.5) else "Unpopular")
    st.caption(f"Model subset: {subset}; model: {artifact['model_name']}. This demo is only as valid as the trained dataset and target definition.")
