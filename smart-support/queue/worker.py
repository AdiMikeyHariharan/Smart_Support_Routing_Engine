import time
import requests
import os
import sys
import torch
from datetime import datetime
from typing import Optional
from fastapi import HTTPException
import asyncio
import aiohttp

from dotenv import load_dotenv

load_dotenv()

# ==========================================================
# 🔥 CPU STABILITY (IMPORTANT FOR MULTI-WORKER)
# ==========================================================
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ==========================================================
# Internal Imports
# ==========================================================
from ..ml.classifier import baseline_classify, baseline_urgency
from ..agents import route_to_agent
from .queue import add_to_priority_queue, add_to_priority_queue_2
from ..models import SupportTicket

# ==========================================================
# 🔥 LOAD MODELS ONCE PER WORKER PROCESS
# ==========================================================
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from ..ml.dedup import get_embedding, check_for_storm, add_to_recent


print(f"[WORKER INIT | PID {os.getpid()}] Loading ML models...")

_transformer_pipeline = pipeline(
    "text-classification",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

_embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print(f"[WORKER INIT | PID {os.getpid()}] Models loaded successfully.")

# ==========================================================
# Logging Helper
# ==========================================================
def _log(message: str):
    print(f"[WORKER PID {os.getpid()}] {message}")
    sys.stdout.flush()

# ==========================================================
# 🔥 CIRCUIT BREAKER CLASS (REFINED)
# ==========================================================
class CircuitBreaker:
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

    def __init__(self, latency_threshold: float = 0.5, cooldown: int = 30):
        self.state = self.CLOSED
        self.latency_threshold = latency_threshold
        self.cooldown = cooldown
        self.last_failure_time = None

    def allow_request(self) -> bool:
        if self.state == self.OPEN:
            if self.last_failure_time and \
               (time.time() - self.last_failure_time > self.cooldown):
                self.state = self.HALF_OPEN
                _log("Circuit → HALF_OPEN (Testing model)")
                return True

            _log("Circuit OPEN → Using BASELINE fallback")
            return False

        return True

    def record_success(self):
        if self.state == self.HALF_OPEN:
            self.state = self.CLOSED
            _log("Model healthy → Circuit CLOSED")

    def record_failure(self):
        self.state = self.OPEN
        self.last_failure_time = time.time()
        _log("Failure detected → Circuit OPEN")

    def record_latency(self, latency: float):
        _log(f"Model latency: {latency:.4f}s")

        if latency > self.latency_threshold:
            self.record_failure()
        else:
            self.record_success()


# Create breaker instance per worker process
circuit_breaker = CircuitBreaker(
    latency_threshold=0.5,   # 500ms
    cooldown=30              # 30 sec cooldown
)

# Optional: enable latency simulation for testing
SIMULATE_SLOW_MODEL = True

# ==========================================================
# 🔥 CLASSIFICATION (WITH CIRCUIT PROTECTION)
# ==========================================================
def _classify_and_score(description: str):

    # If circuit open → immediate fallback
    if not circuit_breaker.allow_request():
        return (
            baseline_classify(description),
            float(baseline_urgency(description))
        )

    try:
        start_time = time.time()

        # Optional slow simulation (FOR TESTING ONLY)
        if SIMULATE_SLOW_MODEL:
            time.sleep(0.7)

        result = _transformer_pipeline(description)[0]

        category = result["label"]
        urgency = float(result["score"])

        latency = time.time() - start_time
        circuit_breaker.record_latency(latency)

        _log(f"Classification → {category} | Urgency: {urgency:.4f}")

        return category, urgency

    except Exception as e:
        _log(f"Transformer failure → {e}")
        circuit_breaker.record_failure()

        return (
            baseline_classify(description),
            float(baseline_urgency(description))
        )

# ==========================================================
# Embedding
# ==========================================================
def get_embedding(text: str):
    return _embedding_model.encode(text)

# ==========================================================
# Webhook
# ==========================================================
def _send_webhook(message: str):
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")

    if not webhook_url:
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

    _log(f"[M1] Processing {ticket_id}")

    category = baseline_classify(description)
    urgency = baseline_urgency(description)

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

    _log(f"[M2] Processing {ticket_id}")

    category, urgency = _classify_and_score(description)

    ticket = SupportTicket(
        ticket_id=ticket_id,
        subject=subject,
        description=description,
        category=category,
        urgency_score=urgency,
        processing_status="routed"
    )

    add_to_priority_queue(ticket)

    if urgency > 0.8:
        _send_webhook(
            f"🚨 High Urgency Ticket {ticket_id}\n"
            f"Category: {category}\n"
            f"Urgency: {urgency:.3f}"
        )
        add_to_priority_queue_2(ticket)
        _log(f"[M2] Ticket {ticket_id} added to priority queue")

    _log(f"[M2] COMPLETED → {ticket_id}")

    return {
        "status": "added",
        "category": category,
        "urgency": urgency
    }

# ==========================================================
# Milestone 3
# ==========================================================
def process_ticket_m3(ticket_id: str, subject: Optional[str], description: str):

    _log(f"[M3] Processing {ticket_id}")

    try:
        embedding = get_embedding(description)

        category, urgency = _classify_and_score(description)

        agent_id = route_to_agent(category, urgency)

        if agent_id == -1:
            _log("No available agents")
            raise HTTPException(status_code=503, detail="No available agents")

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
            _send_webhook(
                f"🚨 Critical Ticket {ticket_id}\n"
                f"Assigned Agent: {agent_id}\n"
                f"Category: {category}\n"
                f"Urgency: {urgency:.3f}"
            )

        _log(f"[M3] COMPLETED → {ticket_id}")

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
        _log(f"[M3 ERROR] {ticket_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal processing error")