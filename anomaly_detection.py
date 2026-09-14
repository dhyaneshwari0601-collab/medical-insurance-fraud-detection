"""
anomaly_detection.py
----------------------
Unsupervised anomaly detection over numeric claim features using
Isolation Forest. Captures statistical outliers (billing spikes, unusual
procedure counts, abnormal claim durations) independent of the graph and
NLP signals, so the final model combines three genuinely different views
of the same claim.

Complexity
----------
Isolation Forest build:  O(t * n * log(n))   t = n_estimators, n = rows
Scoring:                 O(t * log(n)) per sample
This is the standard reason Isolation Forest is preferred over distance-based
methods (e.g. LOF/O(n^2)) for medium-sized tabular fraud data.
"""

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

NUMERIC_FEATURES = [
    "ClaimAmount",
    "ClaimDurationDays",
    "NumProceduresBilled",
    "ChronicConditions",
    "PatientAge",
]


def run_isolation_forest(df: pd.DataFrame, contamination: float = 0.08):
    X = df[NUMERIC_FEATURES].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    raw_scores = model.decision_function(X_scaled)  # higher = more normal
    anomaly_score = (raw_scores.max() - raw_scores) / (raw_scores.max() - raw_scores.min())
    is_outlier = (model.predict(X_scaled) == -1).astype(int)

    result = df.copy()
    result["anomaly_score"] = anomaly_score
    result["is_statistical_outlier"] = is_outlier
    return result, model, scaler
