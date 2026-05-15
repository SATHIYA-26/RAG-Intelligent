
# Advanced Conversational RAG + Persona Intelligence System

## Features
- Chronological message processing
- Dynamic semantic topic segmentation
- Topic checkpoint summaries
- 100-message timeline checkpoints
- Hybrid retrieval (Semantic + BM25)
- Evidence-based persona extraction
- Query classification
- Hierarchical retrieval
- Streamlit chatbot UI
- Local lightweight models only

---

# Installation

pip install -r requirements.txt

python -m spacy download en_core_web_sm

---

# Run Processing Pipeline

python src/process_conversations.py

---

# Run Persona Extraction

python src/persona_extractor.py

---

# Launch Streamlit Chatbot

streamlit run src/app.py

---

# CSV FORMAT

conversation_id,conversation

Each row contains multiline chat messages.
