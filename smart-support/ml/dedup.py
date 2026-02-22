# ml/dedup.py
import os
import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta

from redis import Redis

# Embedding model (local process cache)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

SIMILARITY_THRESHOLD = 0.9
STORM_WINDOW_MINUTES = 5
STORM_TICKET_COUNT = 10

# Shared state (could be moved to a class or external store like Redis)
recent_tickets = []  # List of (timestamp, embedding, ticket_id)

# Redis keys
_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_conn = Redis.from_url(_REDIS_URL, decode_responses=True)
RECENT_KEY = "dedup:recent"
MASTER_KEY = "dedup:master_incidents"
LOCK_KEY = "dedup:lock"

# In-memory fallback (kept for compatibility/testing)
master_incidents = []

def get_embedding(text: str) -> np.ndarray:
    return embedder.encode(text)

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def _serialize_entry(ts: float, emb: list, ticket_id: str) -> str:
    return json.dumps({"ts": ts, "emb": emb, "ticket_id": ticket_id})

def _deserialize_entry(s: str):
    d = json.loads(s)
    return d["ts"], np.array(d["emb"], dtype=float), d.get("ticket_id")

def check_for_storm(embedding: np.ndarray, timestamp: datetime, ticket_id: str = None) -> bool:
    """
    Atomic check/update using Redis lock so concurrent workers coordinate.
    Returns True if a master incident exists/was created and this ticket should be suppressed.
    """
    ts = timestamp.timestamp()

    lock = redis_conn.lock(LOCK_KEY, blocking_timeout=3, timeout=10)
    try:
        acquired = lock.acquire(blocking=True)
        if not acquired:
            # Could not obtain lock — be conservative and do not suppress
            return False

        # Fetch recent entries and keep only those within window
        raw = redis_conn.lrange(RECENT_KEY, 0, -1)
        window_start_ts = ts - (STORM_WINDOW_MINUTES * 60)

        recent = []  # list of (ts, emb, ticket_id)
        for item in raw:
            try:
                r_ts, r_emb, r_tid = _deserialize_entry(item)
            except Exception:
                continue
            if r_ts >= window_start_ts:
                recent.append((r_ts, r_emb, r_tid))

        # Compute similarity counts
        similar_count = 0
        for _, r_emb, _ in recent:
            if _cosine_similarity(embedding, r_emb) > SIMILARITY_THRESHOLD:
                similar_count += 1

        # Append current ticket to recent (store embedding as list)
        redis_conn.rpush(RECENT_KEY, _serialize_entry(ts, embedding.tolist(), ticket_id or ""))

        # Trim old entries atomically by rewriting the list with only windowed items
        # (simple approach while holding lock)
        # Rebuild list: include previous windowed items plus the current one
        new_list = [ _serialize_entry(r_ts, r_emb.tolist(), r_tid) for r_ts, r_emb, r_tid in recent ]
        new_list.append(_serialize_entry(ts, embedding.tolist(), ticket_id or ""))
        # Replace list
        if new_list:
            redis_conn.delete(RECENT_KEY)
            redis_conn.rpush(RECENT_KEY, *new_list)

        # Check threshold: spec requires strictly more than 10
        if similar_count > STORM_TICKET_COUNT:
            # Ensure we create only one master incident per window by checking a flag key
            created_key = f"dedup:master_created:{int(window_start_ts)}"
            if not redis_conn.exists(created_key):
                incident = {"timestamp": timestamp.isoformat(), "description": "Master Incident from storm"}
                redis_conn.rpush(MASTER_KEY, json.dumps(incident))
                # Keep in-memory copy for local inspection too
                master_incidents.append(incident)
                # Set marker with TTL equal to window length to avoid duplicate masters
                redis_conn.set(created_key, "1", ex=STORM_WINDOW_MINUTES * 60)
            return True

        return False
    finally:
        try:
            lock.release()
        except Exception:
            pass


def add_to_recent(timestamp: datetime, embedding: np.ndarray, ticket_id: str):
    ts = timestamp.timestamp()
    redis_conn.rpush(RECENT_KEY, _serialize_entry(ts, embedding.tolist(), ticket_id or ""))
    # Also keep short in-memory copy (optional)
    return