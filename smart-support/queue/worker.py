import time
import requests
import os
import sys
from datetime import datetime
from typing import Optional
from fastapi import HTTPException
import asyncio
import aiohttp

from dotenv import load_dotenv
load_dotenv()

from ..ml.classifier import baseline_classify, baseline_urgency
from ..agents import route_to_agent
from .queue import add_to_priority_queue, add_to_priority_queue_2
from ..models import SupportTicket

# ==========================================================
# Transformer & Embedding Models (Load ONCE)
# ==========================================================
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# 🔥 Load transformer pipeline once at worker startup
_transformer_pipeline = pipeline(
    "text-classification",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

# 🔥 Load sentence-transformer embedding model once
_embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Wrapper functions for reuse
def transformer_classify(text: str):
    result = _transformer_pipeline(text)[0]
    return result["label"]

def transformer_urgency(text: str):
    result = _transformer_pipeline(text)[0]
    return float(result["score"])

def get_embedding(text: str):
    return _embedding_model.encode(text)

# ==========================================================
# Circuit Breaker
# ==========================================================

CB_CLOSED = "CLOSED"
CB_OPEN = "OPEN"
CB_HALF_OPEN = "HALF_OPEN"

circuit_state = CB_CLOSED
model_latency_threshold = 0.5  # 500ms
cooldown_seconds = 30
last_failure_time = None

def _log(message: str):
    print(f"[WORKER DEBUG] {message}")
    sys.stdout.flush()  # 🔥 force flush inside docker

def _should_use_fallback() -> bool:
    global circuit_state, last_failure_time
    if circuit_state == CB_OPEN:
        if last_failure_time and (time.time() - last_failure_time > cooldown_seconds):
            circuit_state = CB_HALF_OPEN
            _log("Circuit moving to HALF_OPEN")
            return False
        _log("Circuit OPEN → using baseline fallback")
        return True
    return False

def _record_latency(latency: float):
    global circuit_state, last_failure_time
    _log(f"Model latency: {latency:.4f}s")
    if latency > model_latency_threshold:
        circuit_state = CB_OPEN
        last_failure_time = time.time()
        _log("Latency threshold exceeded → Circuit OPEN")
    else:
        if circuit_state == CB_HALF_OPEN:
            circuit_state = CB_CLOSED
            _log("Model healthy → Circuit CLOSED")

# ==========================================================
# Model Inference
# ==========================================================

def _classify_and_score(description: str):
    global circuit_state, last_failure_time
    _log("Starting classification pipeline")

    if _should_use_fallback():
        _log("[CB] Using BASELINE model (circuit OPEN)")
        return baseline_classify(description), float(baseline_urgency(description))

    try:
        _log("Using TRANSFORMER model")
        start_time = time.time()
        category = transformer_classify(description)
        urgency = transformer_urgency(description)
        latency = time.time() - start_time
        _record_latency(latency)
        _log(f"Final classification → Category: {category}, Urgency: {urgency}")
        return category, urgency
    except Exception as e:
        circuit_state = CB_OPEN
        last_failure_time = time.time()
        _log(f"[CB] Transformer failure: {e}")
        _log("[CB] Switching to BASELINE fallback")
        return baseline_classify(description), float(baseline_urgency(description))

# ==========================================================
# Webhook
# ==========================================================

def _send_webhook(message: str):
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        _log("Webhook URL not configured.")
        return
    try:
        requests.post(webhook_url, json={"text": message}, timeout=2)
        _log("Webhook sent successfully.")
    except Exception as e:
        _log(f"Webhook failed: {e}")

# Milestone - 2
async def _send_webhook_async(message: str):
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        _log("Webhook URL not configured.")
        return

    try:
        async with aiohttp.ClientSession() as session:  # Use aiohttp for async HTTP (pip install aiohttp)
            async with session.post(webhook_url, json={"text": message}, timeout=2) as resp:
                if resp.status == 200:
                    _log("Webhook sent successfully.")
    except Exception as e:
        _log(f"Webhook failed: {e}")

# ==========================================================
# Milestone 1
# ==========================================================

def process_ticket_m1(ticket_id: str, subject: Optional[str], description: str):
    _log(f"[M1] Processing ticket {ticket_id}")
    category = baseline_classify(description)
    urgency = baseline_urgency(description)
    _log(f"[M1] Category: {category}, Urgency: {urgency}")
    ticket = SupportTicket(
        ticket_id=ticket_id,
        subject=subject,
        description=description,
        category=category,
        urgency_score=urgency,
        processing_status="routed",
    )
    add_to_priority_queue(ticket)
    return {"status": "added", "category": category, "urgency": urgency}

# ==========================================================
# Milestone 2
# ==========================================================

def process_ticket_m2(ticket_id: str, subject: Optional[str], description: str):
    _log(f"[M2] Processing ticket {ticket_id}")
    try:
        category, urgency = _classify_and_score(description)
        ticket = SupportTicket(
            ticket_id=ticket_id,
            subject=subject,
            description=description,
            category=category,
            urgency_score=urgency,
            processing_status="routed"
        )
        add_to_priority_queue_2(ticket)
        _log(f"[M2] Ticket {ticket_id} added to priority queue")

        if urgency > 0.8:
            _log(f"[M2] High urgency detected: {urgency}")
            _send_webhook(
                f"🚨 High urgency ticket {ticket_id}\n"
                f"Category: {category}\n"
                f"Urgency: {urgency:.3f}"
            )

        _log(f"[M2] COMPLETED → Category: {category}, Urgency: {urgency}")
        return {"status": "added", "category": category, "urgency": urgency}
    except Exception as e:
        _log(f"[M2 ERROR] Ticket {ticket_id}: {e}")
        raise

# ==========================================================
# Milestone 3
# ==========================================================

def process_ticket_m3(ticket_id: str, subject: Optional[str], description: str):
    _log(f"[M3] Processing ticket {ticket_id}")
    try:
        timestamp = datetime.utcnow()
        embedding = get_embedding(description)

        if check_for_storm(embedding, timestamp):
            _log(f"[M3] Storm detected → Suppressing ticket {ticket_id}")
            return {"status": "suppressed", "reason": "storm_detected"}

        add_to_recent(timestamp, embedding, ticket_id)
        category, urgency = _classify_and_score(description)
        agent_id = route_to_agent(category, urgency)

        if agent_id == -1:
            _log(f"[M3] No available agents for ticket {ticket_id}")
            raise HTTPException(status_code=503, detail="No available agents")

        _log(f"[M3] Routed ticket {ticket_id} → Agent {agent_id}")

        ticket = SupportTicket(
            ticket_id=ticket_id,
            subject=subject,
            description=description,
            category=category,
            urgency_score=urgency,
            embedding_vector=embedding.tolist(),
            assigned_agent=agent_id,
            processing_status="routed",
        )
        add_to_priority_queue(ticket)

        if urgency > 0.8:
            _log(f"[M3] Critical urgency detected for {ticket_id}")
            _send_webhook(
                f"🚨 Critical Ticket {ticket_id}\n"
                f"Assigned Agent: {agent_id}\n"
                f"Category: {category}\n"
                f"Urgency: {urgency:.3f}"
            )

        _log(f"[M3] COMPLETED → Ticket {ticket_id}")
        return {
            "status": "routed",
            "ticket_id": ticket_id,
            "category": category,
            "urgency": urgency,
            "assigned_agent": agent_id
        }
    except HTTPException:
        raise
    except Exception as e:
        _log(f"[M3 ERROR] Ticket {ticket_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal processing error")