"""
nlp_pitch_analysis.py
----------------------
"Pitch analysis" module.

Every claim carries a free-text narrative ("pitch") written to justify the
treatment/billing. Fraudulent pitches tend to differ linguistically from
genuine clinical notes: they lean on urgency, vague justification, guarantees,
and pressure language rather than clinical specificity. This module scores
that linguistic signal in two complementary ways:

1. Rule-based lexicon scoring   -> fast, explainable, no training needed.
2. TF-IDF + Logistic Regression -> learns data-driven suspicious phrasing.
3. VADER sentiment              -> captures unusually charged / urgent tone.

Complexity
----------
- Lexicon scoring:      O(n * L)      n = num claims, L = avg tokens/claim
- TF-IDF vectorization: O(n * L)      building sparse term matrix
- LogisticRegression:   O(n * d)      d = vocabulary size (bounded to 300)
- VADER sentiment:      O(n * L)
Overall linear in dataset size -> "medium" complexity, scales to 10^5+ rows
comfortably on a laptop.
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

SUSPICIOUS_LEXICON = [
    "guarantee", "guaranteed", "no questions asked", "cash settlement", "cash",
    "immediately", "urgent", "no delay", "trust the provider", "bulk",
    "maximum allowable", "resubmitted", "convenience", "please approve",
    "process quickly", "paperwork will follow", "precaution only",
]

_analyzer = SentimentIntensityAnalyzer()


def lexicon_score(text: str) -> float:
    """Fraction of suspicious-lexicon hits normalized by phrase count (0-1)."""
    text_l = text.lower()
    hits = sum(1 for phrase in SUSPICIOUS_LEXICON if phrase in text_l)
    return min(hits / 3.0, 1.0)  # cap at 1.0, 3+ hits = maximally suspicious


def sentiment_features(text: str) -> dict:
    scores = _analyzer.polarity_scores(text)
    return {
        "sentiment_compound": scores["compound"],
        "sentiment_neg": scores["neg"],
        "sentiment_pos": scores["pos"],
    }


def urgency_flag(text: str) -> int:
    return int(bool(re.search(r"\b(immediately|urgent|no delay|asap|right away)\b",
                               text.lower())))


class PitchAnalyzer:
    """
    Fits a lightweight TF-IDF + Logistic Regression classifier on claim
    narratives (when labels are available) and exposes a unified
    `transform` method that returns a DataFrame of NLP-derived features,
    including a single blended `pitch_fraud_score` in [0, 1].
    """

    def __init__(self, max_features: int = 300):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=(1, 2), stop_words="english"
        )
        self.clf = LogisticRegression(max_iter=500, class_weight="balanced")
        self._fitted = False

    def fit(self, texts: pd.Series, labels: pd.Series):
        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, labels)
        self._fitted = True
        return self

    def _tfidf_proba(self, texts: pd.Series) -> np.ndarray:
        if not self._fitted:
            return np.zeros(len(texts))
        X = self.vectorizer.transform(texts)
        return self.clf.predict_proba(X)[:, 1]

    def transform(self, texts: pd.Series) -> pd.DataFrame:
        texts = texts.fillna("")
        lex = texts.apply(lexicon_score)
        urgency = texts.apply(urgency_flag)
        sentiment = texts.apply(sentiment_features).apply(pd.Series)
        tfidf_proba = self._tfidf_proba(texts)

        out = pd.DataFrame({
            "lexicon_suspicion": lex.values,
            "urgency_flag": urgency.values,
            "tfidf_fraud_proba": tfidf_proba,
        })
        out = pd.concat([out, sentiment.reset_index(drop=True)], axis=1)

        # Blended pitch score: weighted combination, tunable.
        out["pitch_fraud_score"] = (
            0.35 * out["lexicon_suspicion"]
            + 0.40 * out["tfidf_fraud_proba"]
            + 0.15 * out["urgency_flag"]
            + 0.10 * (out["sentiment_neg"])
        ).clip(0, 1)
        return out

    def analyze_single(self, text: str) -> dict:
        """Convenience method for the live 'test a claim narrative' widget."""
        row = self.transform(pd.Series([text])).iloc[0].to_dict()
        row["top_lexicon_hits"] = [p for p in SUSPICIOUS_LEXICON if p in text.lower()]
        return row
