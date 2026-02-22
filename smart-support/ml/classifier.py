# ml/classifier.py – Fixed & improved (no default distilbert warning)

import re
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# =====================================================
# Milestone 1 — Baseline ML Component
# =====================================================

def baseline_classify(text: str) -> str:
    text_lower = text.lower()
    if any(word in text_lower for word in ['bill', 'payment', 'invoice']):
        return 'Billing'
    elif any(word in text_lower for word in ['legal', 'contract', 'terms']):
        return 'Legal'
    else:
        return 'Technical'

def baseline_urgency(text: str) -> float:
    text_lower = text.lower()
    keywords = ['asap', 'broken', 'urgent', 'immediately']
    return 1.0 if any(k in text_lower for k in keywords) else 0.0

# =====================================================
# Milestone 2 — Transformer ML Component
# =====================================================

# Load embedding model once (unchanged – good model)
st_model = SentenceTransformer("all-MiniLM-L6-v2")

# Zero-shot classification pipeline (unchanged – bart-large-mnli is fine)
classifier_pipeline = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

# Modern sentiment model – NO DEFAULT WARNING + realistic scores
# Trained on Twitter/X data → handles short, urgent, neutral messages well
_sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    revision="main"  # explicit revision = warning gone forever
)

# ==================================================
# CATEGORY CLASSIFICATION (unchanged – semantic similarity)
# ==================================================

CATEGORY_ANCHORS = {
    "Billing": [
        "billing issue",
        "payment failed",
        "invoice error",
        "refund request",
        "charged incorrectly"
    ],
    "Technical": [
        "system error",
        "application crashed",
        "bug in software",
        "feature not working",
        "login problem",
        "server not responding",
        "system outage"
    ],
    "Legal": [
        "contract dispute",
        "terms and conditions issue",
        "legal complaint",
        "compliance issue"
    ]
}

category_embeddings = {
    label: st_model.encode(texts, convert_to_tensor=True)
    for label, texts in CATEGORY_ANCHORS.items()
}

def transformer_classify(text: str) -> str:
    """
    Semantic similarity classification using Sentence Transformer
    """
    text_embedding = st_model.encode(text, convert_to_tensor=True)

    scores = {}
    for label, embeddings in category_embeddings.items():
        sim = util.cos_sim(text_embedding, embeddings)
        scores[label] = float(sim.max())

    return max(scores, key=scores.get)

# ==================================================
# URGENCY REGRESSION – FIXED & SHARPENED
# ==================================================

# Your original anchors (kept – good idea)
URGENCY_ANCHORS = [
    ("production system completely down", 1.0),
    ("service outage affecting all users", 1.0),
    ("critical system failure immediate fix required", 0.98),
    ("server not responding urgent", 0.95),
    ("application crashed cannot access", 0.9),
    ("security breach suspected", 0.95),
    ("unauthorized access detected", 0.92),
    ("please respond as soon as possible", 0.75),
    ("there is an issue please check", 0.5),
    ("minor inconvenience", 0.2),
    ("general inquiry", 0.0),
]

anchor_texts = [t for t, _ in URGENCY_ANCHORS]
anchor_scores = torch.tensor([s for _, s in URGENCY_ANCHORS])

anchor_embeddings = st_model.encode(anchor_texts, convert_to_tensor=True)

def transformer_urgency(text: str, temperature: float = 0.05) -> float:
    """
    Continuous urgency score S ∈ [0,1]
    Uses sharpened semantic similarity + keyword boost
    """
    text_embedding = st_model.encode(text, convert_to_tensor=True)

    similarities = util.cos_sim(text_embedding, anchor_embeddings)[0]

    # Normalize cosine similarity from [-1,1] → [0,1]
    similarities = (similarities + 1) / 2

    # Dominant-anchor override (prevents averaging dilution)
    max_sim = float(similarities.max())
    if max_sim > 0.85:
        idx = similarities.argmax()
        base_score = float(anchor_scores[idx])
    else:
        # Temperature scaling → sharper softmax
        scaled_sim = similarities / temperature
        weights = torch.softmax(scaled_sim, dim=0)
        base_score = float((weights * anchor_scores).sum())

    # Keyword boost – tuned for support tickets
    lower_text = text.lower()
    boost = 0.0

    urgent_keywords = [
        "urgent", "asap", "immediately", "now", "broken", "crashed", "down",
        "emergency", "critical", "fix", "help", "problem", "issue", "error",
        "not working", "can't", "failure", "outage", "refund", "cancel"
    ]

    for kw in urgent_keywords:
        if kw in lower_text:
            boost += 0.12

    # Extra boost for strong emotion / shouting
    if "!!" in text or "!!!" in text or text.isupper():
        boost += 0.18

    # Final score
    urgency = min(1.0, base_score + boost)
    urgency = max(0.0, urgency)

    return round(urgency, 3)