import joblib

# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load(
    "outputs/intent_model.pkl"
)

# ==========================================
# STRONG RULES
# ==========================================

rule_patterns = {

    "reminder": [

        "remind",
        "reminder",
        "don't let me forget"
    ],

    "emotional-support": [

        "i feel",
        "sad",
        "stressed",
        "anxious",
        "lonely",
        "overwhelming",
        "upset"
    ],

    "action-item": [

        "book",
        "send",
        "finish",
        "submit",
        "prepare",
        "schedule",
        "call"
    ],

    "small-talk": [

        "how are you",
        "what's up",
        "hello",
        "hey",
        "good morning"
    ]
}

# ==========================================
# MAIN CLASSIFIER
# ==========================================

def classify_intent(text):

    query = text.lower().strip()

    # ======================================
    # RULE-BASED CHECK
    # ======================================

    for intent, patterns in rule_patterns.items():

        for pattern in patterns:

            if pattern in query:

                return {

                    "intent": intent,

                    "confidence": 1.0,

                    "source": "rule-based"
                }

    # ======================================
    # ML PREDICTION
    # ======================================

    probabilities = model.predict_proba(
        [text]
    )[0]

    prediction = model.predict(
        [text]
    )[0]

    confidence = max(probabilities)

    # ======================================
    # CONFIDENCE THRESHOLD
    # ======================================

    if confidence < 0.55:

        return {

            "intent": "unknown",

            "confidence": round(
                float(confidence),
                2
            ),

            "source": "low-confidence"
        }

    return {

        "intent": prediction,

        "confidence": round(
            float(confidence),
            2
        ),

        "source": "ml-model"
    }