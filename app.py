"""
app.py
-------
Medical Insurance Fraud Detection Engine
Anomaly Detection + Graph Machine Learning + NLP Pitch Analysis

Run with:
    streamlit run app.py
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from utils.anomaly_detection import run_isolation_forest
from utils.graph_features import build_provider_patient_graph, compute_provider_graph_features
from utils.nlp_pitch_analysis import PitchAnalyzer
from utils.fraud_model import build_fused_dataframe, train_fraud_model

DATA_PATH = "data/claims.csv"

st.set_page_config(
    page_title="MedFraud Detection Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------- THEME CSS
st.markdown("""
<style>
    html, body, [class*="css"] { font-family: 'Segoe UI', 'Inter', sans-serif; }
    .main-header {
        padding: 1.4rem 1.8rem; border-radius: 14px;
        background: linear-gradient(135deg, #0F2437 0%, #12324A 60%, #0B4A47 100%);
        border: 1px solid #1E3A52; margin-bottom: 1.2rem;
    }
    .main-header h1 { color: #F1F5F9; margin-bottom: 0.15rem; font-weight: 700; }
    .main-header p { color: #9FB4C7; margin: 0; font-size: 0.95rem; }
    .metric-card {
        background: #111A2E; border: 1px solid #1E2C44; border-radius: 12px;
        padding: 1rem 1.2rem;
    }
    .badge-fraud { background:#4A1518; color:#FF8A8A; padding:2px 10px; border-radius:20px; font-size:0.78rem; }
    .badge-clean { background:#123324; color:#7CE8B0; padding:2px 10px; border-radius:20px; font-size:0.78rem; }
    section[data-testid="stSidebar"] { background-color: #0D1524; }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1> Medical Insurance Fraud Detection Engine</h1>
    <p>Anomaly Detection &nbsp;•&nbsp; Graph Machine Learning &nbsp;•&nbsp; NLP Pitch Analysis</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- PIPELINE
@st.cache_data(show_spinner=False)
def load_claims():
    if not os.path.exists(DATA_PATH):
        from data.generate_dataset import generate
        df = generate()
        os.makedirs("data", exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
    return pd.read_csv(DATA_PATH, parse_dates=["ClaimStartDt", "ClaimEndDt"])


@st.cache_data(show_spinner=False)
def run_pipeline(claims_df: pd.DataFrame):
    # 1. Anomaly detection
    anomaly_df, _, _ = run_isolation_forest(claims_df)

    # 2. Graph ML
    G = build_provider_patient_graph(claims_df)
    graph_features_df = compute_provider_graph_features(G)

    # 3. NLP pitch analysis
    analyzer = PitchAnalyzer()
    analyzer.fit(claims_df["ClaimNarrative"], claims_df["PotentialFraud"])
    nlp_df = analyzer.transform(claims_df["ClaimNarrative"])

    # 4. Fusion + supervised model
    fused_df = build_fused_dataframe(claims_df, anomaly_df, graph_features_df, nlp_df)
    model, metrics, importances, scored_df = train_fraud_model(fused_df)

    return {
        "graph": G,
        "graph_features": graph_features_df,
        "scored_df": scored_df,
        "metrics": metrics,
        "importances": importances,
        "model": model,
    }


@st.cache_resource(show_spinner=False)
def get_analyzer(claims_df: pd.DataFrame):
    analyzer = PitchAnalyzer()
    analyzer.fit(claims_df["ClaimNarrative"], claims_df["PotentialFraud"])
    return analyzer


claims_df = load_claims()

with st.sidebar:
    st.markdown("###  Controls")
    if st.button(" Regenerate synthetic dataset"):
        from data.generate_dataset import generate
        df = generate()
        df.to_csv(DATA_PATH, index=False)
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    st.markdown(f"**Claims loaded:** {len(claims_df):,}")
    st.markdown(f"**Providers:** {claims_df['ProviderID'].nunique():,}")
    st.markdown(f"**Patients:** {claims_df['PatientID'].nunique():,}")
    st.markdown(f"**Fraud rate:** {claims_df['PotentialFraud'].mean():.2%}")
    st.markdown("---")
    st.caption(
        "Dataset is synthetically generated to mirror the schema of the "
        "Kaggle 'Healthcare Provider Fraud Detection Analysis' dataset. "
        "See README.md to swap in the real Kaggle CSVs."
    )

with st.spinner("Running anomaly detection, graph ML, and NLP pitch analysis..."):  
    results = run_pipeline(claims_df)

scored_df = results["scored_df"]
metrics = results["metrics"]
importances = results["importances"]
G = results["graph"]

tabs = st.tabs([
    " Overview", " Graph Network", " Anomaly Detection",
    " NLP Pitch Analysis", " Fraud Model", " Provider Risk Leaderboard",
])

# ---------------------------------------------------------------- OVERVIEW
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Claims", f"{len(claims_df):,}")
    c2.metric("Flagged High-Risk", f"{(scored_df['fraud_probability'] > 0.5).sum():,}")
    c3.metric("Model ROC-AUC", f"{metrics['roc_auc']:.3f}")
    c4.metric("Model F1-Score", f"{metrics['f1']:.3f}")

    st.markdown("#### Claim Amount Distribution: Fraud vs Genuine")
    fig = px.histogram(
        claims_df, x="ClaimAmount", color="PotentialFraud", nbins=60,
        color_discrete_map={0: "#2DD4BF", 1: "#F87171"}, barmode="overlay", opacity=0.7,
    )
    fig.update_layout(template="plotly_white", height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Architecture")
    st.markdown("""
    | Layer | Technique | Purpose |
    |---|---|---|
    | Statistical | Isolation Forest | Flags outlier billing behavior per claim |
    | Structural | Bipartite Provider–Patient Graph + centrality/clustering | Detects fraud rings & referral clusters |
    | Linguistic | TF-IDF + Logistic Regression + lexicon + VADER sentiment | Scores the claim "pitch" narrative |
    | Fusion | Random Forest Classifier | Combines all signals into one fraud probability |
    """)

# ---------------------------------------------------------------- GRAPH
with tabs[1]:
    st.markdown("#### Provider ↔ Patient Network (sampled for readability)")
    top_providers = (
        scored_df.groupby("ProviderID")["fraud_probability"].mean()
        .sort_values(ascending=False).head(18).index.tolist()
    )
    sub_nodes = set()
    for p in top_providers:
        node = f"P::{p}"
        if node in G:
            sub_nodes.add(node)
            sub_nodes.update(G.neighbors(node))
    SG = G.subgraph(sub_nodes)

    pos = nx.spring_layout(SG, seed=42, k=0.35)
    edge_x, edge_y = [], []
    for u, v in SG.edges():
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]

    node_x, node_y, node_color, node_text, node_size = [], [], [], [], []
    risk_lookup = scored_df.groupby("ProviderID")["fraud_probability"].mean().to_dict()
    for n, d in SG.nodes(data=True):
        node_x.append(pos[n][0]); node_y.append(pos[n][1])
        if d.get("node_type") == "provider":
            pid = n.replace("P::", "")
            risk = risk_lookup.get(pid, 0)
            node_color.append(risk)
            node_text.append(f"Provider {pid}<br>Fraud risk: {risk:.2f}")
            node_size.append(22)
        else:
            node_color.append(0)
            node_text.append(f"Patient {n.replace('T::', '')}")
            node_size.append(10)

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color="#2C3E55"),
                             hoverinfo="none", mode="lines")
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers", hoverinfo="text", text=node_text,
        marker=dict(size=node_size, color=node_color, colorscale="Reds",
                    line=dict(width=1, color="#0B1220"), colorbar=dict(title="Fraud risk")),
    )
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(template="plotly_white", height=560, showlegend=False,
                       margin=dict(l=10, r=10, t=10, b=10),
                       xaxis=dict(visible=False), yaxis=dict(visible=False))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Dense stars around a single provider node = classic fraud-ring signature "
               "(one provider recycling the same small patient pool across many claims).")

    st.markdown("#### Graph Feature Table (per provider)")
    st.dataframe(results["graph_features"].sort_values("graph_weighted_degree", ascending=False),
                 use_container_width=True, height=280)

# ---------------------------------------------------------------- ANOMALY
with tabs[2]:
    st.markdown("#### Isolation Forest — Statistical Outliers")
    fig = px.scatter(
        scored_df, x="ClaimAmount", y="anomaly_score", color="PotentialFraud",
        color_discrete_map={0: "#2DD4BF", 1: "#F87171"}, opacity=0.6,
        hover_data=["ClaimID", "ProviderID"],
    )
    fig.update_layout(template="plotly_white", height=420)
    st.plotly_chart(fig, use_container_width=True)

    outlier_rate = scored_df["is_statistical_outlier"].mean()
    st.metric("Flagged as statistical outliers", f"{outlier_rate:.2%}")
    st.dataframe(
        scored_df.sort_values("anomaly_score", ascending=False)
        [["ClaimID", "ProviderID", "ClaimAmount", "NumProceduresBilled",
          "anomaly_score", "PotentialFraud"]].head(15),
        use_container_width=True,
    )

# ---------------------------------------------------------------- NLP
with tabs[3]:
    st.markdown("#### Claim Narrative ('Pitch') Fraud Scoring")
    fig = px.histogram(
        scored_df, x="pitch_fraud_score", color="PotentialFraud", nbins=40,
        color_discrete_map={0: "#2DD4BF", 1: "#F87171"}, barmode="overlay", opacity=0.7,
    )
    fig.update_layout(template="plotly_white", height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("####  Test a Claim Narrative Live")
    default_text = "Emergency treatment required immediately, please approve claim with no delay, guaranteed reimbursement."
    user_text = st.text_area("Paste or edit a claim narrative:", value=default_text, height=100)
    if st.button("Analyze Pitch"):
        analyzer = get_analyzer(claims_df)
        result = analyzer.analyze_single(user_text)
        c1, c2, c3 = st.columns(3)
        c1.metric("Pitch Fraud Score", f"{result['pitch_fraud_score']:.2f}")
        c2.metric("Lexicon Suspicion", f"{result['lexicon_suspicion']:.2f}")
        c3.metric("Sentiment (compound)", f"{result['sentiment_compound']:.2f}")
        if result["top_lexicon_hits"]:
            st.warning("Suspicious phrases detected: " + ", ".join(result["top_lexicon_hits"]))
        else:
            st.success("No suspicious lexicon phrases detected.")

    st.markdown("#### Sample Narratives")
    st.dataframe(
        scored_df[["ClaimID", "ClaimNarrative", "pitch_fraud_score", "PotentialFraud"]]
        .sort_values("pitch_fraud_score", ascending=False).head(10),
        use_container_width=True,
    )

# ---------------------------------------------------------------- MODEL
with tabs[4]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    c2.metric("Precision", f"{metrics['precision']:.3f}")
    c3.metric("Recall", f"{metrics['recall']:.3f}")
    c4.metric("F1-Score", f"{metrics['f1']:.3f}")

    st.markdown("#### Feature Importance (Random Forest)")
    imp_df = importances.reset_index()
    imp_df.columns = ["feature", "importance"]
    fig = px.bar(imp_df.head(12), x="importance", y="feature", orientation="h",
                 color="importance", color_continuous_scale="Teal")
    fig.update_layout(template="plotly_white", height=440, yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Complexity Summary")
    st.markdown("""
    | Component | Time Complexity | Space Complexity |
    |---|---|---|
    | Isolation Forest | O(t·n·log n) | O(t·n) |
    | Graph construction | O(E) | O(V + E) |
    | Graph centrality (sampled) | O(k·(V+E)) | O(V) |
    | TF-IDF + Logistic Regression | O(n·d) | O(n·d) (sparse) |
    | Random Forest fusion | O(T·n·log n·m) | O(T·n) |

    *n = claims, V/E = graph nodes/edges, t/T = ensemble size, d = vocab size, k = betweenness sample.*
    All components are near-linear in dataset size — no quadratic all-pairs
    operations are used, which keeps the engine at "medium" complexity and
    responsive on a laptop for tens of thousands of claims.
    """)

# ---------------------------------------------------------------- LEADERBOARD
with tabs[5]:
    st.markdown("#### Highest-Risk Providers")
    provider_risk = (
        scored_df.groupby("ProviderID")
        .agg(avg_fraud_probability=("fraud_probability", "mean"),
             total_claims=("ClaimID", "count"),
             total_billed=("ClaimAmount", "sum"),
             avg_pitch_score=("pitch_fraud_score", "mean"))
        .reset_index()
        .merge(results["graph_features"], on="ProviderID", how="left")
        .sort_values("avg_fraud_probability", ascending=False)
    )
    st.dataframe(provider_risk.head(25), use_container_width=True, height=560)
    st.download_button(
        " Download full risk report (CSV)",
        provider_risk.to_csv(index=False).encode("utf-8"),
        file_name="provider_risk_report.csv",
        mime="text/csv",
    )
