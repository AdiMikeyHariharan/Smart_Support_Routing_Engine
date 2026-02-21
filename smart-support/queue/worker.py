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


def process_ticket_m2(ticket_id: str, subject: Optional[str], description: str):
    global use_fallback
    start_time = time.time()
    
    if use_fallback:
        cat = baseline_classify(description)
        urg = baseline_urgency(description)
    else:
        cat = transformer_classify(description)
        urg = transformer_urgency(description)
    
    latency = time.time() - start_time
    
    if latency > model_latency_threshold:
        use_fallback = True
    
    ticket = SupportTicket(
        ticket_id=ticket_id,
        subject=subject,
        description=description,
        category=cat,
        urgency_score=urg,
        processing_status="routed"
    )
    add_to_priority_queue(ticket)
    
    if urg > 0.8:
        requests.post("https://mock-webhook.example.com", json={
            "message": f"High urgency ticket: {description}"
        })


def process_ticket_m3(ticket_id: str, subject: Optional[str], description: str):
    global use_fallback
    timestamp = datetime.now()
    embedding = get_embedding(description)
    
    if check_for_storm(embedding, timestamp):
        print(f"Ticket {ticket_id} suppressed due to storm")
        return
    
    add_to_recent(timestamp, embedding, ticket_id)
    
    start_time = time.time()
    
    try:
        if use_fallback:
            cat = baseline_classify(description)
            urg = baseline_urgency(description)
        else:
            cat = transformer_classify(description)
            urg = transformer_urgency(description)
        
        latency = time.time() - start_time
        
        if latency > model_latency_threshold:
            use_fallback = True
        
        agent_id = route_to_agent(cat, urg)
        if agent_id == -1:
            raise HTTPException(status_code=503, detail="No available agents")
        
        ticket = SupportTicket(
            ticket_id=ticket_id,
            subject=subject,
            description=description,
            category=cat,
            urgency_score=urg,
            embedding_vector=embedding.tolist(),
            processing_status="routed"
        )
        add_to_priority_queue(ticket)
        
        if urg > 0.8:
            requests.post("https://mock-webhook.example.com", json={
                "message": f"High urgency ticket assigned to agent {agent_id}: {description}"
            })
    
    except Exception as e:
        print(f"Error processing ticket {ticket_id}: {e}")