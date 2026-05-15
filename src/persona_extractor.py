
import json
import re
from collections import Counter

with open("D:\RAG-ConvMind\RAG-ConvMind\outputs\messages.json") as f:
    messages = json.load(f)

all_text = " ".join(
    [m["text"] for m in messages]
).lower()

persona = {
    "habits": [],
    "personal_facts": [],
    "personality_traits": [],
    "communication_style": {}
}

# ==========================================
# EVIDENCE COUNTS
# ==========================================

habit_rules = {
    "late sleeper": [
        "2am", "3am", "late night",
        "can't sleep", "awake"
    ],

    "travel enthusiast": [
        "travel", "vacation",
        "trip", "explore"
    ]
}

for trait, patterns in habit_rules.items():

    count = 0

    for p in patterns:
        count += all_text.count(p)

    if count >= 2:

        persona["habits"].append({
            "trait": trait,
            "confidence": round(min(0.95, 0.5 + count*0.05), 2),
            "evidence_count": count
        })

# ==========================================
# PERSONAL FACTS
# ==========================================

fact_patterns = {
    "student": "education",
    "college": "education",
    "dog": "pet ownership",
    "cat": "pet ownership",
    "band": "music involvement"
}

for pattern, category in fact_patterns.items():

    matches = re.findall(
        rf"\b{pattern}\b",
        all_text
    )

    count = len(matches)

    if count > 0:

        persona["personal_facts"].append({
            "fact": category,
            "evidence": pattern,
            "frequency": count
        })

# ==========================================
# PERSONALITY
# ==========================================
lengths = []

question_count = 0

emoji_count = len(
    re.findall(r'[😂🤣😊😭❤️]', all_text)
)




# Friendly personality

friendly_words = [
    "thanks",
    "thank you",
    "bro",
    "buddy",
    "nice"
]

friendly_count = sum(
    [all_text.count(x) for x in friendly_words]
)

if friendly_count >= 15:

    persona["personality_traits"].append({

        "trait": "friendly",

        "confidence": 0.77,

        "evidence_count": friendly_count
    })

humor_markers = [
    "lol", "haha", "😂", "🤣"
]

humor_count = sum(
    [all_text.count(x) for x in humor_markers]
)

if humor_count >= 3:

    persona["personality_traits"].append({
        "trait": "humorous",
        "confidence": 0.78,
        "evidence_count": humor_count
    })

emotion_markers = [
    "love", "happy", "miss",
    "great", "excited"
]

emotion_count = sum(
    [all_text.count(x) for x in emotion_markers]
)

if emotion_count >= 10:

    persona["personality_traits"].append({
        "trait": "emotionally expressive",
        "confidence": 0.82,
        "evidence_count": emotion_count
    })

# ==========================================
# COMMUNICATION STYLE
# ==========================================


for msg in messages:

    lengths.append(
        len(msg["text"].split())
    )

    question_count += msg["text"].count("?")
# Curious personality

if question_count > 500:

    persona["personality_traits"].append({

        "trait": "curious",

        "confidence": 0.80,

        "evidence_count": question_count
    })

avg_len = sum(lengths) / len(lengths)

tone = "casual"

if avg_len > 20:
    tone = "descriptive"

persona["communication_style"] = {
    "average_message_length": round(avg_len, 2),
    "emoji_usage_count": emoji_count,
    "question_frequency": question_count,
    "tone": tone
}

with open("/content/drive/MyDrive/Colab Notebooks/modified-rag/advanced_rag_persona_system/advanced_rag_persona_system/outputs/persona.json", "w") as f:
    json.dump(persona, f, indent=4)

print("Persona extraction complete.")
