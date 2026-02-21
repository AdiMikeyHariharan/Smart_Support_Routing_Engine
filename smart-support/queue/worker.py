# queue/worker.py — fixed imports

import time
import requests
from datetime import datetime
from typing import Optional
from fastapi import HTTPException

# Go up one level (..) then into ml/
from ..ml.classifier import baseline_classify, baseline_urgency, transformer_classify, transformer_urgency
from ..ml.dedup import get_embedding, check_for_storm, add_to_recent

# agents.py is one level up
from ..agents import route_to_agent

# queue.py is in the same folder → relative import with single dot
from .queue import add_to_priority_queue

# models.py is one level up
from ..models import SupportTicket

# Circuit breaker state – normal module-level variables
use_fallback = False
model_latency_threshold = 0.5  # 500ms

# ────────────────────────────────────────────────
# the rest of the functions (process_ticket_m1 / m2 / m3)
# ────────────────────────────────────────────────

def process_ticket_m1(ticket_id: str, subject: Optional[str], description: str):
    cat = baseline_classify(description)
    urg = baseline_urgency(description)
    ticket = SupportTicket(
        ticket_id=ticket_id,
        subject=subject,
        description=description,
        category=cat,
        urgency_score=urg,
        processing_status="routed"
    )
    add_to_priority_queue(ticket)
    return {"status": "added", "category": cat, "urgency": urg}


# ────────────────────────────────────────────────
# Milestone 2 – Transformer + continuous urgency
# ────────────────────────────────────────────────

def process_ticket_m2(ticket_id: str, subject: Optional[str], description: str) -> None:
    """
    Milestone 2: Transformer model + continuous urgency score
    Called asynchronously (BackgroundTasks or RQ worker)
    """
    global use_fallback

    start_time = time.time()

    try:
        if use_fallback:
            category = baseline_classify(description)
            urgency = float(baseline_urgency(description))  # 0 or 1
        else:
            category = transformer_classify(description)
            urgency = transformer_urgency(description)      # [0,1]

        latency = time.time() - start_time

        if latency > model_latency_threshold:
            use_fallback = True

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
            # Mock webhook – replace with real Slack/Discord URL in production
            webhook_url = "https://hooks.slack.com/services/XXX/YYY/ZZZ"  # ← change this
            payload = {
                "text": f"🚨 High urgency ticket {ticket_id} (score {urgency:.3f})\n"
                        f"Category: {category}\n"
                        f"Description: {description[:200]}..."
            }
            try:
                requests.post(webhook_url, json=payload, timeout=5)
            except Exception as e:
                print(f"Webhook failed: {e}")

    except Exception as e:
        print(f"Error in m2 processing for {ticket_id}: {e}")


# ────────────────────────────────────────────────
# Milestone 3 – full autonomous processing
# ────────────────────────────────────────────────

def process_ticket_m3(ticket_id: str, subject: Optional[str], description: str) -> None:
    """
    Milestone 3: Deduplication + circuit breaker + skill-based agent routing
    """
    global use_fallback

    timestamp = datetime.now()
    embedding = get_embedding(description)

    # Storm / deduplication check
    if check_for_storm(embedding, timestamp):
        print(f"Ticket {ticket_id} suppressed – part of storm incident")
        return

    add_to_recent(timestamp, embedding, ticket_id)

    start_time = time.time()

    try:
        if use_fallback:
            category = baseline_classify(description)
            urgency = float(baseline_urgency(description))
        else:
            category = transformer_classify(description)
            urgency = transformer_urgency(description)

        latency = time.time() - start_time

        if latency > model_latency_threshold:
            use_fallback = True

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
            processing_status="routed"
        )

        add_to_priority_queue(ticket)

        if urgency > 0.8:
            webhook_url = "https://hooks.slack.com/services/XXX/YYY/ZZZ"  # ← change
            payload = {
                "text": f"🚨 Critical ticket {ticket_id} assigned to agent {agent_id} "
                        f"(urgency {urgency:.3f})\n{description[:200]}..."
            }
            try:
                requests.post(webhook_url, json=payload, timeout=5)
            except Exception as e:
                print(f"Webhook failed: {e}")

    except Exception as e:
        print(f"Error in m3 processing for {ticket_id}: {e}")