"""
fraud_model.py
----------------
Fuses three independent signal families into one supervised classifier:

  1. Statistical anomaly score   (utils/anomaly_detection.py)
  2. Graph ML structural features (utils/graph_features.py)
  3. NLP pitch-analysis features  (utils/nlp_pitch_analysis.py)

A Random Forest is used as the fusion model: it handles mixed-scale
features without heavy preprocessing, gives free feature-importance
ranking (useful for the "why was this flagged" explainability the
dashboard shows), and trains in O(n log n) time -- consistent with the
project's medium time/space complexity target.

Complexity
----------
Training: O(T * n * log(n) * m)   T = n_estimators, n = rows, m = features
Inference: O(T * depth) per row -> effectively O(log n) per prediction
Space: O(T * n) for the stored trees (bounded, T fixed at 200)
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

FEATURE_COLUMNS = [
    "ClaimAmount", "ClaimDurationDays", "NumProceduresBilled",
    "ChronicConditions", "PatientAge",
    "anomaly_score", "is_statistical_outlier",
    "graph_degree", "graph_weighted_degree", "graph_clustering_coeff",
    "graph_avg_neighbor_degree", "graph_betweenness",
    "graph_avg_claim_amount_per_patient",
    "lexicon_suspicion", "urgency_flag", "tfidf_fraud_proba",
    "sentiment_compound", "sentiment_neg", "sentiment_pos",
    "pitch_fraud_score",
]


def build_fused_dataframe(claims_df, anomaly_df, graph_df, nlp_df) -> pd.DataFrame:
    df = claims_df.copy()
    df["anomaly_score"] = anomaly_df["anomaly_score"].values
    df["is_statistical_outlier"] = anomaly_df["is_statistical_outlier"].values
    df = df.merge(graph_df, on="ProviderID", how="left")
    df = pd.concat([df.reset_index(drop=True), nlp_df.reset_index(drop=True)], axis=1)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0)
    return df


def train_fraud_model(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y = df["PotentialFraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    metrics = {
        "roc_auc": roc_auc_score(y_test, proba),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
    }

    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    importances = importances.sort_values(ascending=False)

    full_proba = model.predict_proba(X)[:, 1]
    df_scored = df.copy()
    df_scored["fraud_probability"] = full_proba

    return model, metrics, importances, df_scored
