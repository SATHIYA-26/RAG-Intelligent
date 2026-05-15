import re
import json
import pandas as pd
import numpy as np

from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# ==========================================
# CONFIG
# ==========================================

CSV_PATH = "/content/drive/MyDrive/Colab Notebooks/modified-rag/advanced_rag_persona_system/advanced_rag_persona_system/data/conversations.csv"

TOPIC_THRESHOLD = 0.50
TIMELINE_WINDOW = 100
CHUNK_SIZE = 20
ROLLING_WINDOW = 5

BATCH_SIZE = 256

# ==========================================
# LOAD MODEL
# ==========================================

print("Loading embedding model...")

embed_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ==========================================
# SUMMARIZER
# ==========================================

def summarize_text(text, max_sentences=3):

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text
    )

    sentences = [
        s.strip()
        for s in sentences
        if len(s.split()) > 4
    ]

    summary = " ".join(
        sentences[:max_sentences]
    )

    if not summary.strip():
        summary = text[:300]

    return summary

# ==========================================
# LOAD CSV
# ==========================================

print("\nLoading CSV dataset...")

df = pd.read_csv(
    CSV_PATH,
    header=None
)

df.columns = ["conversation"]

print(f"Dataset Loaded: {len(df)} conversations")

messages = []

global_msg_id = 0

# ==========================================
# EXTRACT MESSAGES
# ==========================================

print("\nExtracting messages...")

for idx, row in tqdm(df.iterrows(), total=len(df)):

    conv_id = idx + 1

    conversation_text = str(
        row["conversation"]
    )

    lines = conversation_text.split("\n")

    for line in lines:

        line = line.strip()

        if not line:
            continue

        global_msg_id += 1

        speaker = "unknown"

        match = re.match(
            r"(User \d+):",
            line
        )

        if match:
            speaker = match.group(1)

        messages.append({

            "global_msg_id": global_msg_id,

            "conversation_id": conv_id,

            "speaker": speaker,

            "text": line
        })

print(f"\nTotal Messages: {len(messages)}")

# ==========================================
# BATCH EMBEDDINGS
# ==========================================

print("\nGenerating embeddings in batches...")

all_texts = [
    m["text"]
    for m in messages
]

all_embeddings = embed_model.encode(

    all_texts,

    batch_size=BATCH_SIZE,

    show_progress_bar=True,

    convert_to_numpy=True,

    normalize_embeddings=True
)

print("Embeddings completed.")

# ==========================================
# TOPIC SEGMENTATION
# ==========================================

print("\nRunning topic segmentation...")

topic_summaries = []

current_topic = []

current_embeddings = []

topic_id = 1

for idx in tqdm(range(len(messages))):

    msg = messages[idx]

    emb = all_embeddings[idx]

    if len(current_topic) == 0:

        current_topic.append(msg)

        current_embeddings.append(emb)

        continue

    recent_embeddings = current_embeddings[
        -ROLLING_WINDOW:
    ]

    centroid = np.mean(
        recent_embeddings,
        axis=0
    )

    sim = np.dot(
        emb,
        centroid
    )

    if sim < TOPIC_THRESHOLD:

        topic_text = " ".join(
            [m["text"] for m in current_topic]
        )

        summary = summarize_text(
            topic_text
        )

        topic_data = {

            "topic_id": topic_id,

            "conversation_id":
                current_topic[0]["conversation_id"],

            "start_msg":
                current_topic[0]["global_msg_id"],

            "end_msg":
                current_topic[-1]["global_msg_id"],

            "summary": summary
        }

        topic_summaries.append(
            topic_data
        )

        topic_id += 1

        current_topic = [msg]

        current_embeddings = [emb]

    else:

        current_topic.append(msg)

        current_embeddings.append(emb)

# ==========================================
# FINAL TOPIC
# ==========================================

if current_topic:

    topic_text = " ".join(
        [m["text"] for m in current_topic]
    )

    summary = summarize_text(
        topic_text
    )

    topic_data = {

        "topic_id": topic_id,

        "conversation_id":
            current_topic[0]["conversation_id"],

        "start_msg":
            current_topic[0]["global_msg_id"],

        "end_msg":
            current_topic[-1]["global_msg_id"],

        "summary": summary
    }

    topic_summaries.append(
        topic_data
    )

print(f"\nTotal Topics Created: {len(topic_summaries)}")

# ==========================================
# TIMELINE CHECKPOINTS
# ==========================================

print("\nCreating timeline checkpoints...")

timeline_summaries = []

for i in tqdm(

    range(0, len(messages), TIMELINE_WINDOW)
):

    checkpoint_msgs = messages[
        i:i+TIMELINE_WINDOW
    ]

    checkpoint_text = " ".join(

        [m["text"] for m in checkpoint_msgs]
    )

    summary = summarize_text(
        checkpoint_text
    )

    timeline_summaries.append({

        "checkpoint_id": i,

        "start_msg":
            checkpoint_msgs[0]["global_msg_id"],

        "end_msg":
            checkpoint_msgs[-1]["global_msg_id"],

        "summary": summary
    })

# ==========================================
# CHUNKS
# ==========================================

print("\nCreating chunks...")

chunk_docs = []

for i in tqdm(

    range(0, len(messages), CHUNK_SIZE)
):

    chunk_msgs = messages[
        i:i+CHUNK_SIZE
    ]

    chunk_text = "\n".join([

        f"{m['speaker']}: {m['text']}"

        for m in chunk_msgs
    ])

    chunk_docs.append(chunk_text)

# ==========================================
# BM25
# ==========================================

print("\nCreating BM25 index...")

tokenized = [
    x.split()
    for x in chunk_docs
]

bm25 = BM25Okapi(tokenized)

# ==========================================
# SAVE OUTPUTS
# ==========================================

print("\nSaving outputs...")

with open(
    "/content/drive/MyDrive/Colab Notebooks/modified-rag/advanced_rag_persona_system/advanced_rag_persona_system/outputs/topic_summaries.json",
    "w"
) as f:

    json.dump(
        topic_summaries,
        f,
        indent=4
    )

with open(
    "/content/drive/MyDrive/Colab Notebooks/modified-rag/advanced_rag_persona_system/advanced_rag_persona_system/outputs/timeline_summaries.json",
    "w"
) as f:

    json.dump(
        timeline_summaries,
        f,
        indent=4
    )

with open(
    "/content/drive/MyDrive/Colab Notebooks/modified-rag/advanced_rag_persona_system/advanced_rag_persona_system/outputs/messages.json",
    "w"
) as f:

    json.dump(
        messages,
        f,
        indent=4
    )

with open(
    "/content/drive/MyDrive/Colab Notebooks/modified-rag/advanced_rag_persona_system/advanced_rag_persona_system/outputs/bm25_chunks.json",
    "w"
) as f:

    json.dump(
        chunk_docs,
        f,
        indent=4
    )

print("\nALL DONE SUCCESSFULLY 🚀")