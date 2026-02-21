import time
import requests
import os
from datetime import datetime
from typing import Optional
from fastapi import HTTPException

# OPTIONAL (for local development)
from dotenv import load_dotenv
load_dotenv()

from ..ml.classifier import (
    baseline_classify,
    baseline_urgency,
    transformer_classify,
    transformer_urgency,
)
from ..ml.dedup import get_embedding, check_for_storm, add_to_recent
from ..agents import route_to_agent
from .queue import add_to_priority_queue
from ..models import SupportTicket


# ==========================================================
# Circuit Breaker (Production-style lightweight version)
# ==========================================================

CB_CLOSED = "CLOSED"
CB_OPEN = "OPEN"
CB_HALF_OPEN = "HALF_OPEN"

circuit_state = CB_CLOSED
model_latency_threshold = 0.5  # 500ms
cooldown_seconds = 30
last_failure_time = None


def _should_use_fallback() -> bool:
    global circuit_state, last_failure_time

    if circuit_state == CB_OPEN:
        if last_failure_time and (time.time() - last_failure_time > cooldown_seconds):
            circuit_state = CB_HALF_OPEN
            return False
        return True

    return False


def _record_latency(latency: float):
    global circuit_state, last_failure_time

    if latency > model_latency_threshold:
        circuit_state = CB_OPEN
        last_failure_time = time.time()
    else:
        if circuit_state == CB_HALF_OPEN:
            circuit_state = CB_CLOSED


# ==========================================================
# Helper: Model Inference
# ==========================================================

def _classify_and_score(description: str):
    if _should_use_fallback():
        category = baseline_classify(description)
        urgency = float(baseline_urgency(description))
    else:
        start_time = time.time()
        category = transformer_classify(description)
        urgency = transformer_urgency(description)
        latency = time.time() - start_time
        _record_latency(latency)

    return category, urgency


# ==========================================================
# Helper: Secure Webhook Sender
# ==========================================================

def _send_webhook(message: str):
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")

    if not webhook_url:
        print("[WEBHOOK] SLACK_WEBHOOK_URL not configured.")
        return

    payload = {"text": message}

    try:
        requests.post(webhook_url, json=payload, timeout=2)
    except Exception as e:
        print(f"[WEBHOOK ERROR] {e}")


# ==========================================================
# Milestone 1 – Synchronous Baseline
# ==========================================================

def process_ticket_m1(ticket_id: str, subject: Optional[str], description: str):
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
# Milestone 2 – Async Transformer Processing
# ==========================================================

def process_ticket_m2(ticket_id: str, subject: Optional[str], description: str) -> None:
    """
    Milestone 2:
    - Transformer classification
    - Continuous urgency scoring
    - Latency-based fallback (circuit breaker)
    - Webhook for high urgency
    """

    try:
        # Uses circuit breaker internally
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

        # 🔥 High urgency alert
        if urgency > 0.8:
            _send_webhook(
                f"🚨 High urgency ticket {ticket_id}\n"
                f"Category: {category}\n"
                f"Urgency Score: {urgency:.3f}\n"
                f"{description[:200]}..."
            )

        return {
            "status": "added",
            "category": category,
            "urgency": urgency
        }

    except Exception as e:
        print(f"[M2 ERROR] Ticket {ticket_id}: {e}")

# ==========================================================
# Milestone 3 – Autonomous Orchestrator
# ==========================================================

def process_ticket_m3(ticket_id: str, subject: Optional[str], description: str) -> None:
    try:
        timestamp = datetime.now()
        embedding = get_embedding(description)

        if check_for_storm(embedding, timestamp):
            print(f"[M3] Ticket {ticket_id} suppressed (storm detected)")
            return

        add_to_recent(timestamp, embedding, ticket_id)

        category, urgency = _classify_and_score(description)

        agent_id = route_to_agent(category, urgency)

        if agent_id == -1:
            raise HTTPException(status_code=503, detail="No available agents")

        ticket = SupportTicket(
            ticket_id=ticket_id,
            subject=subject,
            description=description,
            category=category,
            urgency_score=urgency,
            embedding_vector=embedding.tolist(),
            processing_status="routed",
        )

        add_to_priority_queue(ticket)

        if urgency > 0.8:
            _send_webhook(
                f"🚨 Critical ticket {ticket_id} assigned to agent {agent_id} "
                f"(urgency {urgency:.3f})\n"
                f"{description[:200]}..."
            )

    except Exception as e:
        print(f"[M3 ERROR] Ticket {ticket_id}: {e}")