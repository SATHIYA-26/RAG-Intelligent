import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# ==========================================
# TRAINING DATA
# ==========================================

dataset = {

    "text": [

        # ==================================
        # REMINDER
        # ==================================

        "remind me tomorrow",
        "set a reminder",
        "please remind me later",
        "don't let me forget",
        "remind me to call mom",
        "set reminder for assignment",
        "can you remind me tonight",
        "remind me about the meeting",

        # ==================================
        # EMOTIONAL SUPPORT
        # ==================================

        "i feel sad",
        "i am stressed",
        "i feel terrible",
        "i feel anxious",
        "i'm emotionally exhausted",
        "i had a terrible day",
        "everything feels overwhelming",
        "i feel lonely lately",

        # ==================================
        # ACTION ITEM
        # ==================================

        "book tickets",
        "send the email",
        "finish the project",
        "submit the assignment",
        "prepare the presentation",
        "schedule the interview",
        "clean the dataset",
        "call the client",

        # ==================================
        # SMALL TALK
        # ==================================

        "how are you",
        "what's up",
        "hello bro",
        "good morning",
        "nice weather today",
        "how was your day",
        "hey what's happening",
        "lol that's funny",

        # ==================================
        # UNKNOWN
        # ==================================

        "quantum processors rotate internally",
        "blue triangles collapse unexpectedly",
        "bananas operate under moonlight",
        "random meaningless text",
        "neural matrix instability",
        "statistical entropy collapsed"
    ],

    "intent": [

        # reminder
        "reminder",
        "reminder",
        "reminder",
        "reminder",
        "reminder",
        "reminder",
        "reminder",
        "reminder",

        # emotional
        "emotional-support",
        "emotional-support",
        "emotional-support",
        "emotional-support",
        "emotional-support",
        "emotional-support",
        "emotional-support",
        "emotional-support",

        # action
        "action-item",
        "action-item",
        "action-item",
        "action-item",
        "action-item",
        "action-item",
        "action-item",
        "action-item",

        # small talk
        "small-talk",
        "small-talk",
        "small-talk",
        "small-talk",
        "small-talk",
        "small-talk",
        "small-talk",
        "small-talk",

        # unknown
        "unknown",
        "unknown",
        "unknown",
        "unknown",
        "unknown",
        "unknown"
    ]
}

df = pd.DataFrame(dataset)

# ==========================================
# PIPELINE
# ==========================================

pipeline = Pipeline([

    (
        "tfidf",

        TfidfVectorizer(

            ngram_range=(1, 2),

            lowercase=True,

            stop_words="english",

            max_features=3000
        )
    ),

    (
        "classifier",

        LogisticRegression(

            max_iter=3000,

            class_weight="balanced"
        )
    )
])

# ==========================================
# TRAIN MODEL
# ==========================================

pipeline.fit(

    df["text"],

    df["intent"]
)

# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(

    pipeline,

    "outputs/intent_model.pkl",

    compress=3
)

print(
    "Offline intent classifier trained successfully."
)