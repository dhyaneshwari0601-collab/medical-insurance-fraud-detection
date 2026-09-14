"""
graph_features.py
-------------------
Graph Machine Learning module.

Builds a bipartite Provider <-> Patient graph from the claims table and
derives per-provider structural features. Fraud rings show up as dense,
tightly-shared patient clusters around a small set of providers, so
graph-theoretic centrality/clustering measures give strong, cheap signal
that plain tabular features miss.

Complexity
----------
- Graph construction:            O(E)            E = number of claims
- Degree / weighted degree:      O(V + E)
- Clustering coefficient:        O(V * d_avg^2)   d_avg = average degree
- Approximate betweenness:       O(k * (V + E))   k = sample size (bounded)
                                  (exact betweenness is O(V*E), we sample
                                  k nodes via networkx's `k` param to keep
                                  this "medium" complexity on larger graphs)
Overall near-linear in claim volume for the graphs this project targets
(hundreds of providers, thousands of patients).
"""

import networkx as nx
import pandas as pd
import numpy as np


def build_provider_patient_graph(df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for _, row in df.iterrows():
        p_node = f"P::{row['ProviderID']}"
        pt_node = f"T::{row['PatientID']}"
        G.add_node(p_node, node_type="provider")
        G.add_node(pt_node, node_type="patient")
        if G.has_edge(p_node, pt_node):
            G[p_node][pt_node]["weight"] += 1
            G[p_node][pt_node]["total_amount"] += row["ClaimAmount"]
        else:
            G.add_edge(p_node, pt_node, weight=1, total_amount=row["ClaimAmount"])
    return G


def compute_provider_graph_features(G: nx.Graph, sample_k: int = 200) -> pd.DataFrame:
    provider_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "provider"]

    degree = dict(G.degree(provider_nodes))
    weighted_degree = dict(G.degree(provider_nodes, weight="weight"))
    clustering = nx.clustering(G, nodes=provider_nodes)
    avg_neighbor_deg = nx.average_neighbor_degree(G, nodes=provider_nodes)

    # Approximate betweenness centrality (sampled) keeps this tractable
    # (O(k*(V+E)) instead of exact O(V*E)) while still surfacing "hub"
    # providers that bridge many patient clusters.
    k = min(sample_k, G.number_of_nodes())
    betweenness = nx.betweenness_centrality(G, k=k, seed=42)

    rows = []
    for n in provider_nodes:
        pid = n.replace("P::", "")
        shared_patient_amounts = [
            G[n][nbr]["total_amount"] for nbr in G.neighbors(n)
        ]
        rows.append({
            "ProviderID": pid,
            "graph_degree": degree.get(n, 0),
            "graph_weighted_degree": weighted_degree.get(n, 0),
            "graph_clustering_coeff": clustering.get(n, 0.0),
            "graph_avg_neighbor_degree": avg_neighbor_deg.get(n, 0.0),
            "graph_betweenness": betweenness.get(n, 0.0),
            "graph_avg_claim_amount_per_patient": (
                np.mean(shared_patient_amounts) if shared_patient_amounts else 0.0
            ),
        })
    return pd.DataFrame(rows)
