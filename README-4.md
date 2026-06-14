# 💊 Drug-Drug-Food Interaction Knowledge Graph + RAG Assistant

> **AI-powered interaction checker combining Knowledge Graphs with LLM-based reasoning**
> Built with NetworkX · Claude API · Streamlit · Python

---

## 📌 About the Project

Patients often take multiple medications alongside everyday foods, and dangerous interactions are easy to miss. This project models drugs, foods, and their interactions as a **knowledge graph** — a network of connected entities — and uses an AI assistant (RAG-based) to answer natural-language safety questions.

Instead of simple keyword search, the system **traverses relationships** between entities (e.g. *Warfarin → interacts with → Spinach → severity: High*) to give explainable, structured answers.

---

## 🖥️ App Features

- 💬 **Chat interface** — ask questions in plain English about multiple drugs/foods at once
- 🔍 **Browse mode** — select any drug or food to see all its known interactions, sorted by severity
- 🕸️ **Interactive knowledge graph** — visualizes drugs (blue), foods (green), and interaction edges color-coded by severity (red = high, orange = moderate, gray = low)
- ⚕️ Built-in medical disclaimer for responsible use

---

## 🧠 How It Works

```
User Question
     ↓
Entity Extraction → identifies drugs/foods mentioned
     ↓
Knowledge Graph (NetworkX) → finds direct + related interactions
     ↓
Context Formatting → structures graph data as text
     ↓
Claude API → generates clear, patient-friendly explanation
     ↓
Streamlit UI → displays answer + severity cards + graph
```

---

## 🗂️ Project Structure

```
drug-interaction-kg/
├── build_graph.py       ← Builds knowledge graph from CSV data
├── kg_engine.py          ← Graph query + traversal logic
├── chatbot.py            ← RAG pipeline (Graph + Claude API)
├── app.py                ← Streamlit UI with graph visualization
├── requirements.txt
├── data/
│   ├── drugs.csv          (15 common medications)
│   ├── foods.csv          (15 common foods/compounds)
│   └── interactions.csv   (30 documented interactions)
└── graph/
    └── knowledge_graph.json
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Build the Knowledge Graph
```bash
python build_graph.py
```

### 3. (Optional) Set Claude API Key
For AI-generated natural language answers:
```bash
export ANTHROPIC_API_KEY="your_key_here"
```
Without a key, the app shows structured graph data directly (still fully functional).

### 4. Launch the App
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 📊 Dataset Overview

| Category | Count | Examples |
|---|---|---|
| Drugs | 15 | Warfarin, Metformin, Simvastatin, Ciprofloxacin |
| Foods | 15 | Grapefruit, Spinach, Dairy, Alcohol |
| Interactions | 30 | Drug-food and drug-drug pairs with severity & recommendations |

Each interaction includes: **severity level** (High/Moderate/Low), **mechanism of interaction**, and a **practical recommendation**.

---

## 🏭 Industry Applications

| Sector | Use Case |
|---|---|
| Pharmacy apps & telemedicine | Real-time interaction warnings for patients |
| Hospital pharmacy systems | Decision support during prescription review |
| Pharmacovigilance teams | Structured interaction knowledge base |
| Health insurance / PBMs | Medication safety review tools |

---

## 🛠️ Tech Stack

`Python` · `NetworkX` · `Pandas` · `Streamlit` · `Claude API` · `vis.js` (graph visualization)

---

## ⚕️ Disclaimer

This is an **educational/academic demo** using a small sample dataset (15 drugs, 15 foods, 30 interactions). It is **not a medical device** and must not be used for real clinical decisions. Always consult a licensed pharmacist or physician.

---

## 👤 Author

**Yashwanth Kumar H C**
B.E. Electrical & Electronics Engineering
Sri Sairam College of Engineering, Bengaluru — 2026

📧 Connect on [LinkedIn](https://www.linkedin.com/in/yashwanth-gowda-3aabb8307) | 💻 [GitHub](https://github.com/yashwanthg854-coder)

---

## 📄 License

MIT License — free to use for academic, research, and personal projects.
