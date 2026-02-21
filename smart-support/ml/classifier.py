# ml/classifier.py
import re
from transformers import pipeline

# Global pipelines
classifier_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
sentiment_pipeline = pipeline("sentiment-analysis")

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

def transformer_classify(text: str) -> str:
    result = classifier_pipeline(text, candidate_labels=["Billing issue", "Technical problem", "Legal concern"])
    mapping = {'Billing issue': 'Billing', 'Technical problem': 'Technical', 'Legal concern': 'Legal'}
    return mapping[result['labels'][0]]

def transformer_urgency(text: str) -> float:
    sent = sentiment_pipeline(text)[0]
    return 1 - sent['score'] if sent['label'] == 'POSITIVE' else sent['score']