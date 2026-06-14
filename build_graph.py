"""
build_graph.py
──────────────
Builds a knowledge graph from drugs.csv, foods.csv, and interactions.csv
using NetworkX. Saves the graph in GML format for reuse.

Run: python build_graph.py
"""

import os
import pandas as pd
import networkx as nx
from networkx.readwrite import json_graph
import json

DATA_DIR    = "data"
GRAPH_DIR   = "graph"
GRAPH_PATH  = os.path.join(GRAPH_DIR, "knowledge_graph.json")


def load_data():
    drugs        = pd.read_csv(os.path.join(DATA_DIR, "drugs.csv"))
    foods        = pd.read_csv(os.path.join(DATA_DIR, "foods.csv"))
    interactions = pd.read_csv(os.path.join(DATA_DIR, "interactions.csv"))
    return drugs, foods, interactions


def build_graph(drugs: pd.DataFrame, foods: pd.DataFrame,
                interactions: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()

    # Add drug nodes
    for _, row in drugs.iterrows():
        G.add_node(
            row["drug_id"],
            label=row["drug_name"],
            node_type="drug",
            category=row["category"],
            common_use=row["common_use"],
        )

    # Add food nodes
    for _, row in foods.iterrows():
        G.add_node(
            row["food_id"],
            label=row["food_name"],
            node_type="food",
            category=row["category"],
            key_compound=row["key_compound"],
        )

    # Add interaction edges
    for _, row in interactions.iterrows():
        G.add_edge(
            row["source_id"],
            row["target_id"],
            interaction_id=row["interaction_id"],
            severity=row["severity"],
            mechanism=row["mechanism"],
            recommendation=row["recommendation"],
        )

    return G


def print_summary(G: nx.Graph):
    drug_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "drug"]
    food_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "food"]

    print("📊 Knowledge Graph Summary")
    print(f"   Total nodes      : {G.number_of_nodes()}")
    print(f"   - Drug nodes     : {len(drug_nodes)}")
    print(f"   - Food nodes     : {len(food_nodes)}")
    print(f"   Total edges      : {G.number_of_edges()}")

    severities = [d["severity"] for _, _, d in G.edges(data=True)]
    for sev in ["High", "Moderate", "Low"]:
        count = severities.count(sev)
        bar = "█" * count
        print(f"   {sev:<10} {bar} ({count})")


if __name__ == "__main__":
    print("📂 Loading datasets...")
    drugs, foods, interactions = load_data()
    print(f"   Drugs: {len(drugs)} | Foods: {len(foods)} | Interactions: {len(interactions)}")

    print("\n🔗 Building knowledge graph...")
    G = build_graph(drugs, foods, interactions)

    print()
    print_summary(G)

    os.makedirs(GRAPH_DIR, exist_ok=True)
    data = json_graph.node_link_data(G)
    with open(GRAPH_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"\n💾 Graph saved → {GRAPH_PATH}")
    print("\n✅ Done! Next: run 'python kg_engine.py' to test queries.")
