# ml/classifier.py
import re
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# Global pipelines
classifier_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
sentiment_pipeline = pipeline("sentiment-analysis")

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
# Sentence Transformer based routing + urgency regression
# =====================================================

# --------------------------------------------------
# Load embedding model once
# --------------------------------------------------
st_model = SentenceTransformer("all-MiniLM-L6-v2")

# ==================================================
# CATEGORY CLASSIFICATION
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
# URGENCY REGRESSION (SHARPENED)
# ==================================================

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
    Uses sharpened semantic similarity
    """
    text_embedding = st_model.encode(text, convert_to_tensor=True)

    similarities = util.cos_sim(text_embedding, anchor_embeddings)[0]

    # Normalize cosine similarity from [-1,1] → [0,1]
    similarities = (similarities + 1) / 2

    # Dominant-anchor override (prevents averaging dilution)
    max_sim = float(similarities.max())
    if max_sim > 0.85:
        idx = similarities.argmax()
        return float(anchor_scores[idx])

    # Temperature scaling → sharper softmax
    scaled_sim = similarities / temperature
    weights = torch.softmax(scaled_sim, dim=0)

    score = float((weights * anchor_scores).sum())
    return round(score, 3)