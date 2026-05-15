# ConvoMind — Conversational Intelligence System

Offline-first conversational AI system with:
- Chronological RAG
- Persona Extraction
- Persona Drift Detection
- Offline Intent Classification
- Conflict-Aware Retrieval

---

# Features

## Chronological Conversational RAG
- Topic checkpoint generation
- Timeline checkpoint summaries
- BM25 retrieval
- Chronological memory reasoning

## Persona Engine
Extracts:
- habits
- personality traits
- communication style
- behavioral patterns

## Persona Drift Detection
Tracks:
- evolving conversational tone
- personality shifts over time
- drift triggers

## Offline Intent Classification
Fully offline lightweight NLP model:
- reminder
- emotional-support
- action-item
- small-talk
- unknown

## Conflict-Aware RAG
- contradiction detection
- emotional chunk ranking
- merged memory synthesis

---

# Architecture

```text
User Query
     ↓
Query Router
 ┌────────────┬──────────────┬──────────────┐
 ↓            ↓              ↓
Persona    Drift Engine   Conflict RAG
 ↓
Intent Classifier
 ↓
Final Response
```

---

# Tech Stack

| Technology | Usage |
|---|---|
| Python | Core system |
| Streamlit | Web UI |
| BM25Okapi | Retrieval |
| scikit-learn | Offline ML |
| TF-IDF | Vectorization |
| Logistic Regression | Intent classification |
| JSON / SQLite | Local storage |

---

# Project Structure

```text
src/
├── process_conversations.py
├── persona_extractor.py
├── persona_drift.py
├── intent_classifier.py
├── intent_router.py
├── query_router.py
├── rag_resolver.py
└── app.py
```

---

# How Persona Drift Works

The system:
1. Processes conversations chronologically
2. Extracts conversational traits per conversation/day
3. Compares evolving behavioral signals
4. Detects personality drift
5. Identifies drift triggers

Example:

```text
Day 1 → curious & formal
Day 4 → casual & frustrated
Day 7 → playful
```

---

# How Retrieval Works

The RAG system:
- creates topic checkpoints
- generates timeline summaries
- indexes conversational chunks using BM25
- retrieves memory chronologically
- resolves contradictions using conflict-aware ranking

---

# How Persona Extraction Works

Persona is built using:
- evidence-based behavioral signals
- communication statistics
- emotional markers
- repeated conversational patterns

The system avoids unsupported psychological assumptions.

---

# Running Locally

## Install dependencies

```bash
pip install -r requirements.txt
```

## Train intent classifier

```bash
python src/intent_classifier.py
```

## Start chatbot

```bash
streamlit run src/app.py
```

---

# Demo Questions

## Persona

```text
What kind of person is this user?
```

## Drift

```text
How did the user's mood change over time?
```

## Conflict RAG

```text
Did I mention anything about my sister?
```

## Intent

```text
Remind me tomorrow
```

---

# Design Principles

- Offline-first
- Privacy-preserving
- Lightweight local inference
- Modular AI architecture
- Conflict-aware memory reasoning

---

# Author

Sathiya Priyan
