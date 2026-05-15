import re
from collections import Counter

# ==========================================
# EMOTIONAL WORDS
# ==========================================

emotion_words = [

    "love",
    "hate",
    "fight",
    "cry",
    "stress",
    "angry",
    "happy",
    "sad"
]

# ==========================================
# SCORE CHUNKS
# ==========================================

def score_chunk(text, recency):

    emotion_score = sum(

        text.lower().count(word)

        for word in emotion_words
    )

    final_score = (

        emotion_score * 2
        +
        recency
    )

    return final_score

# ==========================================
# DETECT CONTRADICTIONS
# ==========================================

def detect_contradictions(chunks):

    combined = " ".join(
        chunks
    ).lower()

    contradictions = []

    contradiction_pairs = [

        (
            "moved abroad",
            "lives with me"
        ),

        (
            "single",
            "relationship"
        ),

        (
            "happy",
            "depressed"
        )
    ]

    for a, b in contradiction_pairs:

        if a in combined and b in combined:

            contradictions.append(
                f"{a} vs {b}"
            )

    return contradictions

# ==========================================
# CLEAN TEXT
# ==========================================

def clean_chunk(text):

    text = re.sub(
        r'User \d+:',
        '',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    return text.strip()

# ==========================================
# MAIN RESOLUTION
# ==========================================

def resolve_conflicts(chunks):

    # ======================================
    # RANK CHUNKS
    # ======================================

    ranked = []

    for idx, chunk in enumerate(chunks):

        score = score_chunk(
            chunk,
            idx
        )

        ranked.append(
            (score, chunk)
        )

    ranked.sort(
        reverse=True
    )

    top_chunks = [

        clean_chunk(x[1])

        for x in ranked[:5]
    ]

    # ======================================
    # CONTRADICTIONS
    # ======================================

    contradictions = detect_contradictions(
        top_chunks
    )

    # ======================================
    # TOPIC EXTRACTION
    # ======================================

    combined = " ".join(
        top_chunks
    ).lower()

    words = re.findall(

        r'\b[a-zA-Z]{4,}\b',

        combined
    )

    stopwords = {

        "that",
        "this",
        "with",
        "have",
        "they",
        "them",
        "what",
        "your",
        "about",
        "there",
        "their",
        "would",
        "could",
        "should",
        "really",
        "because",
        "conversation",
        "conversations",
        "hello",
        "thanks",
        "great",
        "love",
        "like",
        "good",
        "nice",
        "yeah",
        "okay",
        "just",
        "very",
        "much"
    }

    filtered = [

        w

        for w in words

        if w not in stopwords
    ]

    freq = Counter(filtered)

    top_topics = [

        w

        for w, c in
        freq.most_common(5)
    ]

    # ======================================
    # FINAL RESPONSE
    # ======================================

    response = ""

    if top_topics:

        response += (
            "Relevant conversations discuss topics related to "
        )

        response += (
            ", ".join(top_topics)
        )

        response += ". "
    else:

        response += (
            "Relevant memories were retrieved. "
        )

    # ======================================
    # MEMORY SUMMARY
    # ======================================

    response += "\n\n"

    for chunk in top_chunks[:3]:

        response += f"• {chunk[:220]}\n\n"

    # ======================================
    # CONTRADICTIONS
    # ======================================

    if contradictions:

        response += (
            "Some retrieved conversations "
            "contain conflicting details."
        )

    return response