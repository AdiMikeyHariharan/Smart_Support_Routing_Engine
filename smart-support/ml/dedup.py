# ml/dedup.py
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta

embedder = SentenceTransformer('all-MiniLM-L6-v2')

SIMILARITY_THRESHOLD = 0.9
STORM_WINDOW_MINUTES = 5
STORM_TICKET_COUNT = 10

# Shared state (could be moved to a class or external store like Redis)
recent_tickets = []  # List of (timestamp, embedding, ticket_id)
master_incidents = []

def get_embedding(text: str) -> np.ndarray:
    return embedder.encode(text)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def check_for_storm(embedding: np.ndarray, timestamp: datetime) -> bool:
    global recent_tickets, master_incidents
    similar_count = 0
    window_start = timestamp - timedelta(minutes=STORM_WINDOW_MINUTES)
    to_remove = []
    for i, (ts, emb, tid) in enumerate(recent_tickets):
        if ts < window_start:
            to_remove.append(i)
            continue
        if cosine_similarity(embedding, emb) > SIMILARITY_THRESHOLD:
            similar_count += 1
    
    # Clean old tickets
    for i in sorted(to_remove, reverse=True):
        del recent_tickets[i]
    
    if similar_count >= STORM_TICKET_COUNT:
        master_incidents.append({"timestamp": timestamp, "description": "Master Incident from storm"})
        return True  # Suppress individual
    return False

def add_to_recent(timestamp: datetime, embedding: np.ndarray, ticket_id: str):
    global recent_tickets
    recent_tickets.append((timestamp, embedding, ticket_id))