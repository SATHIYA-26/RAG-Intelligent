import streamlit as st
import json
import re

from collections import Counter
from rank_bm25 import BM25Okapi

from intent_router import classify_intent
from query_router import detect_query_type
from rag_resolver import resolve_conflicts

# ==========================================
# LOAD DATA
# ==========================================

with open(
    "outputs/persona.json",
    "r",
    encoding="utf-8"
) as f:

    persona = json.load(f)

with open(
    "outputs/persona_drift.json",
    "r",
    encoding="utf-8"
) as f:

    persona_drift = json.load(f)

with open(
    "outputs/bm25_chunks.json",
    "r",
    encoding="utf-8"
) as f:

    bm25_chunks = json.load(f)

with open(
    "outputs/timeline_summaries.json",
    "r",
    encoding="utf-8"
) as f:

    timeline_summaries = json.load(f)

# ==========================================
# BM25
# ==========================================

bm25 = BM25Okapi(
    [x.split() for x in bm25_chunks]
)

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(

    page_title="ConvoMind",

    layout="wide"
)

st.title("ConvoMind")

st.caption(
    "Conversational Intelligence System"
)

# ==========================================
# QUERY INPUT
# ==========================================

query = st.text_input(
    "Ask a question"
)

# ==========================================
# QUERY HANDLING
# ==========================================

if query:

    q = query.lower()

    # ======================================
    # QUERY ROUTER
    # ======================================

    query_type = detect_query_type(query)

    

    # ======================================
    # PERSONA
    # ======================================

    if query_type == "persona":

        st.subheader(
            "Persona Analysis"
        )

        response = ""

        traits = [

            x["trait"]

            for x in persona[
                "personality_traits"
            ]
        ]

        habits = [

            x["trait"]

            for x in persona[
                "habits"
            ]
        ]

        if traits:

            response += (
                "The conversations suggest "
                "the user is "
            )

            response += (
                ", ".join(traits)
            )

            response += ". "

        if habits:

            response += (
                "Observed habits include "
            )

            response += (
                ", ".join(habits)
            )

            response += ". "

        style = persona[
            "communication_style"
        ]

        response += (
            f"The communication style "
            f"appears {style['tone']} "
            f"and interactive."
        )

        st.success(response)

    # ======================================
    # PERSONA DRIFT
    # ======================================

    elif query_type == "persona-drift":

        st.subheader(
            "Persona Drift Timeline"
        )

        drift_entries = [

            x for x in persona_drift

            if x["drift_detected"]
        ]

        if not drift_entries:

            st.success(
                "No major personality drifts detected."
            )

        else:

            response = ""

            for drift in drift_entries[:10]:

                traits = ", ".join(
                    drift["traits"]
                )

                response += (

                    f"Day {drift['day']} → "

                    f"{traits}. "

                    f"Trigger: "

                    f"{drift['trigger']}.\n\n"
                )

            st.success(response)

    # ======================================
    # TIMELINE
    # ======================================

    elif query_type == "timeline":

        st.subheader(
            "Timeline Retrieval"
        )

        query_words = q.split()

        matched = []

        for checkpoint in timeline_summaries:

            summary = checkpoint["summary"]

            if any(

                word in summary.lower()

                for word in query_words
            ):

                matched.append(summary)

        if matched:

            cleaned = re.sub(

                r'User \d+:',

                '',

                matched[0]
            )

            cleaned = re.sub(

                r'\s+',

                ' ',

                cleaned
            ).strip()

            response = (

                "Earlier conversations "
                "included discussions about "
            )

            response += cleaned[:400]

            st.success(response)

        else:

            st.success(
                "No relevant timeline memories found."
            )

    # ======================================
    # CONFLICT RAG
    # ======================================

    elif query_type == "conflict-rag":

        st.subheader(
            "Conflict-Aware Retrieval"
        )

        bm25_scores = bm25.get_scores(
            query.split()
        )

        ranked_chunks = sorted(

            zip(
                bm25_scores,
                bm25_chunks
            ),

            reverse=True

        )[:10]

        retrieved_chunks = [

            x[1]

            for x in ranked_chunks
        ]

        response = resolve_conflicts(
            retrieved_chunks
        )

        st.success(response)

    # ======================================
    # LIGHTWEIGHT INTENTS
    # ======================================

    else:

        intent_result = classify_intent(
            query
        )

        predicted_intent = intent_result[
            "intent"
        ]

        confidence = intent_result[
            "confidence"
        ]

        source = intent_result[
            "source"
        ]

        

        # ==================================
        # REMINDER
        # ==================================

        if predicted_intent == "reminder":

            st.subheader(
                "Reminder Intent"
            )

            st.success(
                "The message appears "
                "to contain a reminder request."
            )

        # ==================================
        # EMOTIONAL SUPPORT
        # ==================================

        elif predicted_intent == "emotional-support":

            st.subheader(
                "Emotional Support Intent"
            )

            st.success(
                "The message appears "
                "emotionally expressive "
                "or support-seeking."
            )

        # ==================================
        # ACTION ITEM
        # ==================================

        elif predicted_intent == "action-item":

            st.subheader(
                "Action Item Intent"
            )

            st.success(
                "The message appears "
                "to contain an actionable task."
            )

        # ==================================
        # SMALL TALK
        # ==================================

        elif predicted_intent == "small-talk":

            st.subheader(
                "Small Talk Intent"
            )

            st.success(
                "The message appears "
                "to be casual conversation."
            )

        # ==================================
        # UNKNOWN
        # ==================================

        else:

            st.subheader(
                "Memory Retrieval"
            )

            bm25_scores = bm25.get_scores(
                query.split()
            )

            ranked_chunks = sorted(

                zip(
                    bm25_scores,
                    bm25_chunks
                ),

                reverse=True

            )[:8]

            retrieved_chunks = [

                x[1]

                for x in ranked_chunks
            ]

            combined = " ".join(
                retrieved_chunks
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

            topics = [

                w

                for w, c in
                freq.most_common(6)
            ]

            response = (

                "Relevant conversations "
                "discuss recurring themes "
                "related to: "
            )

            response += ", ".join(topics)

            st.success(response)