"""
app.py
──────
Streamlit dashboard for the Drug-Drug-Food Interaction
Knowledge Graph + RAG Assistant.

Run: streamlit run app.py
"""

import streamlit as st
import streamlit.components.v1 as components
import json

from kg_engine import load_graph, get_all_labels, get_interactions, find_node
from chatbot import ask_chatbot

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Drug-Food Interaction Assistant",
    page_icon="💊",
    layout="wide",
)

st.markdown("""
<style>
    .main-title { font-size: 2rem; font-weight: 700; color: #1565C0; margin-bottom: 0; }
    .subtitle { font-size: 0.95rem; color: #555; margin-top: 0; }
    .severity-high { background:#ffebee; border-left:4px solid #C62828; padding:0.7rem 1rem; border-radius:8px; margin-bottom:0.5rem; }
    .severity-moderate { background:#fff8e1; border-left:4px solid #F9A825; padding:0.7rem 1rem; border-radius:8px; margin-bottom:0.5rem; }
    .severity-low { background:#e8f5e9; border-left:4px solid #2E7D32; padding:0.7rem 1rem; border-radius:8px; margin-bottom:0.5rem; }
    .disclaimer { background:#eceff1; border-radius:8px; padding:0.8rem 1rem; font-size:0.85rem; color:#444; margin-top:1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">💊 Drug-Food Interaction Knowledge Graph Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered interaction checker using Knowledge Graphs + RAG | Educational demo</p>', unsafe_allow_html=True)
st.divider()

# ── Load Graph ───────────────────────────────────────────────────────────────
@st.cache_resource
def get_graph():
    return load_graph()

try:
    G = get_graph()
except FileNotFoundError:
    st.error("⚠️ Knowledge graph not found! Run `python build_graph.py` first.")
    st.code("python build_graph.py", language="bash")
    st.stop()

drug_labels = get_all_labels(G, "drug")
food_labels = get_all_labels(G, "food")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Knowledge Graph Stats")
    st.metric("Drugs in database", len(drug_labels))
    st.metric("Foods in database", len(food_labels))
    st.metric("Total interactions", G.number_of_edges())

    st.divider()
    st.subheader("💊 Available Drugs")
    st.caption(", ".join(drug_labels))

    st.subheader("🍽️ Available Foods")
    st.caption(", ".join(food_labels))

    st.divider()
    st.caption("🔑 Set ANTHROPIC_API_KEY environment variable for AI-generated answers. Without it, raw graph data is shown.")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💬 Ask the Assistant", "🔍 Browse Interactions", "🕸️ Knowledge Graph"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Chat Interface
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Ask about drug-food combinations")

    examples = [
        "Can I take Warfarin with spinach and ibuprofen?",
        "What should I avoid while on Simvastatin?",
        "Is it safe to drink milk with Tetracycline?",
        "Can I combine Lisinopril and Spironolactone?",
    ]

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Your question:", placeholder="e.g. Can I take Aspirin with grapefruit juice?")
    with col2:
        st.write("")
        st.write("")
        ask_clicked = st.button("Ask", type="primary", use_container_width=True)

    st.caption("💡 Try: " + " | ".join(f"_{e}_" for e in examples[:2]))

    selected_example = st.selectbox("Or pick an example question:", ["-- Select --"] + examples)
    if selected_example != "-- Select --":
        query = selected_example
        ask_clicked = True

    if ask_clicked and query:
        with st.spinner("🔍 Searching knowledge graph and generating answer..."):
            result = ask_chatbot(query, G)

        st.markdown("### 🤖 Answer")
        st.info(result["answer"])

        if result["entities"]:
            st.markdown(f"**Identified entities:** {', '.join(result['entities'])}")

        if result["graph_data"] and result["graph_data"]["direct_interactions"]:
            st.markdown("### ⚠️ Direct Interactions Found")
            for inter in result["graph_data"]["direct_interactions"]:
                sev = inter["severity"]
                css_class = f"severity-{sev.lower()}"
                st.markdown(f"""
                <div class="{css_class}">
                    <b>{inter['entity_a']} ↔ {inter['entity_b']}</b> — {sev} Severity<br>
                    <b>Mechanism:</b> {inter['mechanism']}<br>
                    <b>Recommendation:</b> {inter['recommendation']}
                </div>
                """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
    ⚕️ <b>Disclaimer:</b> This is an educational demo project using a small sample dataset.
    It is NOT a substitute for professional medical advice. Always consult a pharmacist
    or doctor before making decisions about medications and diet.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Browse Interactions
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Browse all interactions for a specific drug or food")

    all_labels = sorted(drug_labels + food_labels)
    selected = st.selectbox("Select a drug or food:", all_labels)

    if selected:
        node_id = find_node(G, selected)
        interactions = get_interactions(G, node_id)

        node_data = G.nodes[node_id]
        if node_data["node_type"] == "drug":
            st.markdown(f"**{selected}** — {node_data.get('category', '')}")
            st.caption(f"Common use: {node_data.get('common_use', '')}")
        else:
            st.markdown(f"**{selected}** — {node_data.get('category', '')}")
            st.caption(f"Key compound: {node_data.get('key_compound', '')}")

        st.markdown(f"#### Found {len(interactions)} interaction(s)")

        for inter in interactions:
            sev = inter["severity"]
            css_class = f"severity-{sev.lower()}"
            icon = "🔴" if sev == "High" else "🟡" if sev == "Moderate" else "🟢"
            st.markdown(f"""
            <div class="{css_class}">
                {icon} <b>{inter['interacts_with']}</b> ({inter['interacts_type']}) — {sev} Severity<br>
                <b>Mechanism:</b> {inter['mechanism']}<br>
                <b>Recommendation:</b> {inter['recommendation']}
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Knowledge Graph Visualization
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Interactive Knowledge Graph")
    st.caption("Drugs (blue) and Foods (green) connected by interaction edges. Red = High severity, Orange = Moderate, Gray = Low.")

    # Build vis.js network data
    nodes_js = []
    edges_js = []

    severity_colors = {"High": "#E53935", "Moderate": "#FB8C00", "Low": "#9E9E9E"}

    for node_id, data in G.nodes(data=True):
        color = "#1565C0" if data["node_type"] == "drug" else "#2E7D32"
        shape = "dot" if data["node_type"] == "drug" else "diamond"
        nodes_js.append({
            "id": node_id,
            "label": data.get("label", node_id),
            "color": color,
            "shape": shape,
            "title": data.get("common_use", data.get("key_compound", "")),
            "size": 18,
        })

    for u, v, data in G.edges(data=True):
        sev = data.get("severity", "Low")
        edges_js.append({
            "from": u,
            "to": v,
            "color": severity_colors.get(sev, "#9E9E9E"),
            "width": 4 if sev == "High" else 2 if sev == "Moderate" else 1,
            "title": f"{sev}: {data.get('recommendation', '')[:80]}...",
        })

    nodes_json = json.dumps(nodes_js)
    edges_json = json.dumps(edges_js)

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.6/vis-network.min.js"></script>
      <style>
        #network {{ width: 100%; height: 600px; border: 1px solid #ddd; border-radius: 8px; background: #fafafa; }}
        .legend {{ font-family: Arial, sans-serif; font-size: 13px; margin-bottom: 8px; }}
        .legend span {{ display: inline-block; margin-right: 16px; }}
        .dot {{ height: 12px; width: 12px; border-radius: 50%; display: inline-block; margin-right: 4px; vertical-align: middle; }}
      </style>
    </head>
    <body>
      <div class="legend">
        <span><span class="dot" style="background:#1565C0;"></span>Drug</span>
        <span><span class="dot" style="background:#2E7D32;"></span>Food</span>
        <span><span class="dot" style="background:#E53935;"></span>High severity link</span>
        <span><span class="dot" style="background:#FB8C00;"></span>Moderate severity link</span>
        <span><span class="dot" style="background:#9E9E9E;"></span>Low severity link</span>
      </div>
      <div id="network"></div>
      <script>
        var nodes = new vis.DataSet({nodes_json});
        var edges = new vis.DataSet({edges_json});
        var container = document.getElementById('network');
        var data = {{ nodes: nodes, edges: edges }};
        var options = {{
          physics: {{ stabilization: true, barnesHut: {{ gravitationalConstant: -3000, springLength: 120 }} }},
          interaction: {{ hover: true, tooltipDelay: 100 }},
          nodes: {{ font: {{ size: 12, color: '#222' }} }}
        }};
        var network = new vis.Network(container, data, options);
      </script>
    </body>
    </html>
    """

    components.html(html_code, height=650, scrolling=False)

st.divider()
st.caption("💊 Drug-Food Interaction Knowledge Graph Assistant | Built with NetworkX + Claude API + Streamlit | Academic Project")
