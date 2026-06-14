"""
chatbot.py
──────────
RAG pipeline: combines knowledge graph retrieval with Claude API
to answer natural language drug-food interaction questions.

Usage:
    from chatbot import ask_chatbot
    answer = ask_chatbot("Can I take Warfarin with spinach and ibuprofen?", G)
"""

import os
import re
import requests

from kg_engine import (
    load_graph, find_combination_interactions, format_context, get_all_labels
)

# ── Config ───────────────────────────────────────────────────────────────────
# Set your Anthropic API key as an environment variable:
#   export ANTHROPIC_API_KEY="your_key_here"
API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
API_URL  = "https://api.anthropic.com/v1/messages"
MODEL    = "claude-sonnet-4-6"


# ── 1. Extract Drug/Food Names from User Query ───────────────────────────────
def extract_entities(query: str, G) -> list[str]:
    """
    Find which known drugs/foods are mentioned in the user's query
    by checking against all labels in the graph.
    """
    query_lower = query.lower()
    all_labels  = get_all_labels(G)

    found = []
    for label in all_labels:
        if label.lower() in query_lower:
            found.append(label)

    return found


# ── 2. Call Claude API ────────────────────────────────────────────────────────
def call_claude(prompt: str) -> str:
    if not API_KEY:
        return _fallback_answer(prompt)

    headers = {
        "Content-Type":      "application/json",
        "x-api-key":         API_KEY,
        "anthropic-version": "2023-06-01",
    }
    body = {
        "model":      MODEL,
        "max_tokens": 600,
        "messages":   [{"role": "user", "content": prompt}],
    }

    try:
        response = requests.post(API_URL, headers=headers, json=body, timeout=30)
        response.raise_for_status()
        data = response.json()
        text_blocks = [b["text"] for b in data["content"] if b["type"] == "text"]
        return "\n".join(text_blocks)
    except requests.exceptions.RequestException as e:
        return f"⚠️ API Error: {e}\n\n" + _fallback_answer(prompt)


# ── 3. Fallback (no API key) — template-based answer from graph data ─────────
def _fallback_answer(prompt: str) -> str:
    return (
        "ℹ️ No ANTHROPIC_API_KEY set — showing raw knowledge graph data below.\n"
        "(Set your API key for a natural-language explanation.)\n\n"
        + prompt.split("CONTEXT:")[-1].strip()
    )


# ── 4. Main RAG Pipeline ───────────────────────────────────────────────────────
def ask_chatbot(query: str, G) -> dict:
    """
    Full RAG pipeline:
    1. Extract drug/food entities from the query
    2. Query knowledge graph for interactions
    3. Build context and prompt Claude API
    4. Return answer + structured graph data (for visualization)
    """
    entities = extract_entities(query, G)

    if not entities:
        return {
            "answer":   "I couldn't identify any known drugs or foods in your question. "
                        "Try mentioning specific names like 'Warfarin', 'Grapefruit', or 'Aspirin'.",
            "entities": [],
            "graph_data": None,
        }

    kg_result = find_combination_interactions(G, entities)
    context   = format_context(kg_result)

    prompt = f"""You are a pharmacy assistant AI. A user asked a question about drug-food interactions.
Use ONLY the knowledge graph data provided below to answer. Be clear, concise, and practical.
If severity is High, emphasize caution. Always include the recommendation.
Do not invent information not present in the context. If no interaction is found, say so clearly
but still mention any general safety notes from the context.

USER QUESTION: {query}

CONTEXT:
{context}

Provide a clear, helpful answer in 3-5 sentences, written for a patient (not a doctor)."""

    answer = call_claude(prompt)

    return {
        "answer":     answer,
        "entities":   entities,
        "graph_data": kg_result,
    }


# ── Main / Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    G = load_graph()

    test_queries = [
        "Can I take Warfarin with spinach and ibuprofen?",
        "What should I avoid while on Simvastatin?",
        "Is it safe to drink milk with Tetracycline?",
    ]

    for q in test_queries:
        print("=" * 70)
        print(f"Q: {q}")
        print("=" * 70)
        result = ask_chatbot(q, G)
        print(f"Entities found: {result['entities']}")
        print(f"\nAnswer:\n{result['answer']}\n")
