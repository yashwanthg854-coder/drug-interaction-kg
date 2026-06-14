"""
kg_engine.py
────────────
Query engine for the drug-food interaction knowledge graph.
Provides functions to:
- Find a node by name (fuzzy matching)
- Get all interactions for a drug/food
- Find interaction paths between multiple drugs/foods
- Format results for RAG context

Run: python kg_engine.py   (runs test queries)
"""

import os
import json
import networkx as nx
from networkx.readwrite import json_graph
from difflib import get_close_matches

GRAPH_PATH = "graph/knowledge_graph.json"


# ── Load Graph ───────────────────────────────────────────────────────────────
def load_graph(path: str = GRAPH_PATH) -> nx.Graph:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Graph not found at {path}. Run 'python build_graph.py' first."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return json_graph.node_link_graph(data)


# ── Find Node by Name (fuzzy match) ─────────────────────────────────────────
def find_node(G: nx.Graph, name: str) -> str | None:
    """Find a node ID by drug/food name using case-insensitive fuzzy match."""
    name_lower = name.strip().lower()

    # Build label -> id map
    label_map = {}
    for node_id, data in G.nodes(data=True):
        label = data.get("label", "").lower()
        label_map[label] = node_id

    # Exact match
    if name_lower in label_map:
        return label_map[name_lower]

    # Substring match
    for label, node_id in label_map.items():
        if name_lower in label or label in name_lower:
            return node_id

    # Fuzzy match
    matches = get_close_matches(name_lower, label_map.keys(), n=1, cutoff=0.6)
    if matches:
        return label_map[matches[0]]

    return None


# ── Get All Interactions for a Node ──────────────────────────────────────────
def get_interactions(G: nx.Graph, node_id: str) -> list[dict]:
    """Return all interactions (edges) connected to a node."""
    results = []
    node_label = G.nodes[node_id].get("label", node_id)

    for neighbor in G.neighbors(node_id):
        edge_data    = G.edges[node_id, neighbor]
        neighbor_data = G.nodes[neighbor]

        results.append({
            "entity":         node_label,
            "interacts_with": neighbor_data.get("label", neighbor),
            "interacts_type": neighbor_data.get("node_type", "unknown"),
            "severity":       edge_data.get("severity", "Unknown"),
            "mechanism":      edge_data.get("mechanism", ""),
            "recommendation": edge_data.get("recommendation", ""),
        })

    # Sort by severity: High > Moderate > Low
    severity_order = {"High": 0, "Moderate": 1, "Low": 2}
    results.sort(key=lambda r: severity_order.get(r["severity"], 3))
    return results


# ── Multi-Entity Query: find pairwise interactions among a list ─────────────
def find_combination_interactions(G: nx.Graph, entity_names: list[str]) -> dict:
    """
    For a list of drug/food names, find:
    - direct interactions between each pair
    - all known interactions for each individual entity
    """
    node_ids   = {}
    not_found  = []

    for name in entity_names:
        node_id = find_node(G, name)
        if node_id:
            node_ids[name] = node_id
        else:
            not_found.append(name)

    # Pairwise direct interactions
    pairwise = []
    names = list(node_ids.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_id, b_id = node_ids[names[i]], node_ids[names[j]]
            if G.has_edge(a_id, b_id):
                edge_data = G.edges[a_id, b_id]
                pairwise.append({
                    "entity_a":       G.nodes[a_id].get("label"),
                    "entity_b":       G.nodes[b_id].get("label"),
                    "severity":       edge_data.get("severity"),
                    "mechanism":      edge_data.get("mechanism"),
                    "recommendation": edge_data.get("recommendation"),
                })

    # All interactions for each entity (for broader context)
    all_interactions = {}
    for name, node_id in node_ids.items():
        all_interactions[name] = get_interactions(G, node_id)

    return {
        "matched_entities":    {k: G.nodes[v].get("label") for k, v in node_ids.items()},
        "not_found":           not_found,
        "direct_interactions": pairwise,
        "all_interactions":    all_interactions,
    }


# ── Format Results as Context String for LLM ─────────────────────────────────
def format_context(query_result: dict) -> str:
    """Convert query results into a readable text block for RAG context."""
    lines = []

    if query_result["not_found"]:
        lines.append(f"⚠️ Could not identify: {', '.join(query_result['not_found'])}")

    if query_result["direct_interactions"]:
        lines.append("\n=== DIRECT INTERACTIONS FOUND ===")
        for inter in query_result["direct_interactions"]:
            lines.append(
                f"\n• {inter['entity_a']} ↔ {inter['entity_b']}\n"
                f"  Severity: {inter['severity']}\n"
                f"  Mechanism: {inter['mechanism']}\n"
                f"  Recommendation: {inter['recommendation']}"
            )
    else:
        lines.append("\n=== NO DIRECT INTERACTIONS FOUND BETWEEN THE GIVEN ITEMS ===")

    lines.append("\n=== OTHER KNOWN INTERACTIONS FOR EACH ITEM ===")
    for entity_name, interactions in query_result["all_interactions"].items():
        if interactions:
            lines.append(f"\n{entity_name}:")
            for inter in interactions[:3]:  # top 3 by severity
                lines.append(
                    f"  - With {inter['interacts_with']} ({inter['interacts_type']}): "
                    f"{inter['severity']} severity. {inter['recommendation']}"
                )

    return "\n".join(lines)


# ── Get Graph Stats (for UI) ──────────────────────────────────────────────────
def get_all_labels(G: nx.Graph, node_type: str = None) -> list[str]:
    labels = []
    for _, data in G.nodes(data=True):
        if node_type is None or data.get("node_type") == node_type:
            labels.append(data.get("label", ""))
    return sorted(labels)


# ── Main / Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    G = load_graph()
    print(f"📊 Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

    # Test 1: Single drug query
    print("=" * 60)
    print("TEST 1: Interactions for 'Warfarin'")
    print("=" * 60)
    node_id = find_node(G, "Warfarin")
    interactions = get_interactions(G, node_id)
    for inter in interactions:
        print(f"  {inter['interacts_with']:<20} | {inter['severity']:<8} | {inter['mechanism'][:60]}...")

    # Test 2: Combination query
    print("\n" + "=" * 60)
    print("TEST 2: Combination — Warfarin + Ibuprofen + Spinach")
    print("=" * 60)
    result = find_combination_interactions(G, ["Warfarin", "Ibuprofen", "Spinach"])
    print(format_context(result))
