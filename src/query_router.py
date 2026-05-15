# ==========================================
# QUERY ROUTER
# ==========================================

def detect_query_type(query):

    q = query.lower()

    # ======================================
    # PERSONA DRIFT
    # ======================================

    drift_patterns = [

        "mood change",
        "personality evolve",
        "over time",
        "drift",
        "tone change",
        "mood shifts",
        "timeline of mood",
        "playful",
        "frustrated",
        "casual over time",
        "emotional shifts"
    ]

    if any(
        p in q
        for p in drift_patterns
    ):

        return "persona-drift"

    # ======================================
    # CONFLICT RAG
    # ======================================

    conflict_patterns = [

        "did i mention",
        "sister",
        "brother",
        "family",
        "conflicting",
        "contradictory",
        "emotionally significant",
        "opinions change",
        "relationship discussions"
    ]

    if any(
        p in q
        for p in conflict_patterns
    ):

        return "conflict-rag"

    # ======================================
    # PERSONA
    # ======================================

    persona_patterns = [

        "what kind of person",
        "habits",
        "communication style",
        "how do they talk",
        "personality"
    ]

    if any(
        p in q
        for p in persona_patterns
    ):

        return "persona"

    # ======================================
    # TIMELINE
    # ======================================

    timeline_patterns = [

        "earlier",
        "before",
        "history",
        "historical"
    ]

    if any(
        p in q
        for p in timeline_patterns
    ):

        return "timeline"

    # ======================================
    # FALLBACK
    # ======================================

    return "intent"