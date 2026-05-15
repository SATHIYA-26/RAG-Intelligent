import json
from collections import defaultdict

# ==========================================
# LOAD MESSAGES
# ==========================================

with open(
    "outputs/messages.json",
    "r",
    encoding="utf-8"
) as f:

    messages = json.load(f)

# ==========================================
# GROUP DAILY CONVERSATIONS
# ==========================================

daily_conversations = defaultdict(list)

for msg in messages:

    conv_id = msg["conversation_id"]

    daily_conversations[conv_id].append(
        msg["text"]
    )

# ==========================================
# TRAIT KEYWORDS
# ==========================================

trait_keywords = {

    "curious": [
        "why",
        "how",
        "what",
        "when",
        "?"
    ],

    "formal": [
        "thank you",
        "please",
        "appreciate",
        "regards"
    ],

    "casual": [
        "bro",
        "lol",
        "haha",
        "yo",
        "dude"
    ],

    "playful": [
        "😂",
        "🤣",
        "fun",
        "excited",
        "yay"
    ],

    "frustrated": [
        "hate",
        "annoying",
        "stress",
        "angry",
        "upset"
    ],

    "emotional": [
        "love",
        "miss",
        "happy",
        "sad",
        "cry"
    ]
}

# ==========================================
# TRIGGER TOPICS
# ==========================================

trigger_topics = {

    "travel discussions": [
        "travel",
        "trip",
        "vacation",
        "flight"
    ],

    "relationship discussions": [
        "boyfriend",
        "girlfriend",
        "love",
        "breakup"
    ],

    "work stress discussions": [
        "work",
        "job",
        "office",
        "deadline",
        "stress"
    ],

    "education discussions": [
        "college",
        "student",
        "study",
        "exam"
    ],

    "gaming discussions": [
        "game",
        "gaming",
        "xbox",
        "playstation"
    ],

    "family discussions": [
        "mom",
        "dad",
        "sister",
        "brother",
        "family"
    ]
}

# ==========================================
# TRAIT SCORING
# ==========================================

def compute_trait_scores(text):

    scores = {}

    for trait, keywords in trait_keywords.items():

        score = sum(
            text.count(word)
            for word in keywords
        )

        scores[trait] = score

    return scores

# ==========================================
# TRAIT DETECTION
# ==========================================

def detect_traits(scores):

    traits = []

    for trait, score in scores.items():

        threshold = 3

        if trait == "curious":
            threshold = 10

        if score >= threshold:

            traits.append(trait)

    return traits

# ==========================================
# TRIGGER DETECTION
# ==========================================

def detect_trigger(text):

    topic_scores = {}

    for topic, words in trigger_topics.items():

        score = sum(
            text.count(w)
            for w in words
        )

        topic_scores[topic] = score

    best_topic = max(
        topic_scores,
        key=topic_scores.get
    )

    if topic_scores[best_topic] == 0:

        return "general conversations"

    return best_topic

# ==========================================
# CONFIDENCE
# ==========================================

def compute_confidence(scores):

    total = sum(scores.values())

    if total == 0:
        return 0.0

    strongest = max(scores.values())

    return round(
        strongest / total,
        2
    )

# ==========================================
# PERSONA DRIFT ANALYSIS
# ==========================================

persona_timeline = []

previous_traits = set()

for day in sorted(daily_conversations.keys()):

    combined_text = " ".join(
        daily_conversations[day]
    ).lower()

    # ======================================
    # TRAIT ANALYSIS
    # ======================================

    trait_scores = compute_trait_scores(
        combined_text
    )

    traits = detect_traits(
        trait_scores
    )

    # ======================================
    # DRIFT DETECTION
    # ======================================

    current_traits = set(traits)

    drift_detected = (
        current_traits != previous_traits
    )

    # ======================================
    # TRIGGER
    # ======================================

    trigger = detect_trigger(
        combined_text
    )

    # ======================================
    # CONFIDENCE
    # ======================================

    confidence = compute_confidence(
        trait_scores
    )

    # ======================================
    # STORE
    # ======================================

    persona_timeline.append({

        "day": int(day),

        "traits": traits,

        "drift_detected": drift_detected,

        "trigger": trigger,

        "confidence": confidence
    })

    previous_traits = current_traits

# ==========================================
# SAVE
# ==========================================

with open(
    "outputs/persona_drift.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        persona_timeline,
        f,
        indent=4
    )

print(
    f"Persona drift analysis created for {len(persona_timeline)} conversations."
)