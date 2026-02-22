# queue/worker.py – Fixed fallback scope + real Slack webhook

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
# CPU STABILITY (IMPORTANT FOR MULTI-WORKER)
# ==========================================================
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ==========================================================
# GLOBAL STATE – Circuit Breaker & Fallback Flag
# ==========================================================
use_fallback = False
model_latency_threshold = 0.5  # 500 ms – as per hackathon

# ==========================================================
# Internal Imports
# ==========================================================
from ..ml.classifier import baseline_classify, baseline_urgency, transformer_classify, transformer_urgency
from ..agents import route_to_agent
from .queue import add_to_priority_queue, add_to_priority_queue_2
from ..models import SupportTicket
from ..ml.dedup import get_embedding, check_for_storm, add_to_recent

# ==========================================================
# LOAD MODELS ONCE PER WORKER PROCESS
# ==========================================================
from transformers import pipeline
from sentence_transformers import SentenceTransformer

print(f"[WORKER INIT | PID {os.getpid()}] Loading ML models...")

# Modern model – no default warning (already fixed)
_sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    revision="main"
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
# CIRCUIT BREAKER CLASS (matches hackathon description)
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
            if self.last_failure_time and (time.time() - self.last_failure_time > self.cooldown):
                self.state = self.HALF_OPEN
                _log("Circuit → HALF_OPEN (testing model)")
                return True
            _log("Circuit OPEN → using BASELINE fallback")
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

circuit_breaker = CircuitBreaker(latency_threshold=0.5, cooldown=30)

# ==========================================================
# CLASSIFICATION (WITH CIRCUIT PROTECTION)
# ==========================================================
def _classify_and_score(description: str):
    # If circuit open → immediate fallback to Milestone 1
    if not circuit_breaker.allow_request():
        return (
            baseline_classify(description),
            float(baseline_urgency(description))
        )

    try:
        start_time = time.time()
        if SIMULATE_SLOW_MODEL:
            time.sleep(0.7)

        # Use your real transformer functions from classifier.py
        category = transformer_classify(description)
        urgency = transformer_urgency(description)

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
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

def _send_webhook(message: str):
    if not SLACK_WEBHOOK_URL:
        _log("No SLACK_WEBHOOK_URL configured – skipping webhook")
        return

    try:
        response = requests.post(SLACK_WEBHOOK_URL, json={"text": message}, timeout=2)
        if response.status_code == 200:
            _log("Webhook sent successfully")
        else:
            _log(f"Webhook failed – HTTP {response.status_code}")
    except Exception as e:
        _log(f"Webhook failed: {e}")

# ==========================================================
# Milestone 1 – synchronous baseline
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
# Milestone 2 – Transformer + continuous urgency + webhook
# ==========================================================
def process_ticket_m2(ticket_id: str, subject: Optional[str], description: str) -> dict:
    _log(f"[M2] Processing {ticket_id}")

    global use_fallback  # needed because we assign to it

    start_time = time.time()

    try:
        if use_fallback:
            category = baseline_classify(description)
            urgency = float(baseline_urgency(description))
        else:
            category, urgency = _classify_and_score(description)

        latency = time.time() - start_time

        if latency > 0.5:
            use_fallback = True
            _log(f"High latency ({latency:.3f}s) → fallback enabled")

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
            message = (
                f"🚨 *HIGH URGENCY TICKET* {ticket_id}\n"
                f"• Category: {category}\n"
                f"• Urgency Score: {urgency:.3f}\n"
                f"• Description: {description[:300]}...\n\n"
                f"*Action needed immediately*"
            )
            _send_webhook(message)
            add_to_priority_queue_2(ticket)
            _log(f"[M2] High urgency ticket added to priority queue 2")

        _log(f"[M2] COMPLETED → {ticket_id}")

    except Exception as e:
        _log(f"[M2 ERROR] {ticket_id}: {e}")
        return {"status": "error", "ticket_id": ticket_id, "error": str(e)}

    return {"status": "processed", "ticket_id": ticket_id, "category": category}

# ==========================================================
# Milestone 3 – full autonomous processing
# ==========================================================
def process_ticket_m3(ticket_id: str, subject: Optional[str], description: str) -> dict:
    _log(f"[M3] Processing {ticket_id}")

    global use_fallback  # ← FIXED: now declared here so it can be read & written

    timestamp = datetime.now()
    embedding = get_embedding(description)

    if check_for_storm(embedding, timestamp):
        _log(f"Ticket {ticket_id} suppressed due to storm")
        return {"status": "suppressed", "ticket_id": ticket_id, "reason": "storm detected"} 

    add_to_recent(timestamp, embedding, ticket_id)

    start_time = time.time()

    try:
        if use_fallback:
            category = baseline_classify(description)
            urgency = float(baseline_urgency(description))
        else:
            category, urgency = _classify_and_score(description)

        latency = time.time() - start_time

        if latency > 0.5:
            use_fallback = True
            _log(f"High latency ({latency:.3f}s) → fallback enabled")

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
            message = (
                f"🚨 *CRITICAL TICKET* {ticket_id}\n"
                f"• Assigned Agent: {agent_id}\n"
                f"• Category: {category}\n"
                f"• Urgency Score: {urgency:.3f}\n"
                f"• Description: {description[:300]}...\n\n"
                f"*Immediate action required*"
            )
            _send_webhook(message)

        _log(f"[M3] COMPLETED → {ticket_id}")

    except HTTPException:
        raise
    except Exception as e:
        _log(f"[M3 ERROR] {ticket_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal processing error")

    return {"status": "processed", "ticket_id": ticket_id, "category": category}