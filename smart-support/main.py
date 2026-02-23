# smart-support/main.py

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from threading import Lock
from rq.job import Job
from redis import Redis
import asyncio
import os
import subprocess
import json

from .models import InputTicket
from .queue.worker import (
    process_ticket_m1,
    process_ticket_m2,
    process_ticket_m3,
)
from .queue.queue import view_queue_2, ticket_queue
from .queue import worker  # for circuit reset
from .ml import dedup


app = FastAPI(title="Smart Support Routing Engine")

origins = [
    "https://smart-support-routing-engine-dgtd.onrender.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

redis_conn = Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))



@app.get("/job/{job_id}")
def get_job_status(job_id: str):
    job = Job.fetch(job_id, connection=redis_conn)

    return {
        "status": job.get_status(),
        "result": job.result
    }
# ==========================================================
# Ticket ID Generator (Thread-safe)
# ==========================================================

ticket_counter = 0
counter_lock = Lock()


def generate_ticket_id():
    global ticket_counter
    with counter_lock:
        ticket_counter += 1
        return f"TKT-{ticket_counter:06d}"

def generate_ticket_id_2():
    counter = redis_conn.incr("ticket_counter")  # Atomic increment, starts at 1 if key doesn't exist
    return f"TKT-{counter:06d}"

# ==========================================================
# Milestone 1 – Synchronous
# ==========================================================

@app.post("/ticket_m1")
def add_ticket_m1(ticket: InputTicket):
    ticket_id = generate_ticket_id()
    return process_ticket_m1(ticket_id, ticket.subject, ticket.description)


# ==========================================================
# Milestone 2 – Redis-based Async Processing
# ==========================================================

@app.post("/ticket_m2", status_code=202)
async def add_ticket_m2(ticket: InputTicket):
    ticket_id = generate_ticket_id_2()

    async def enqueue_job():
        return await asyncio.to_thread(
            ticket_queue.enqueue,
            process_ticket_m2,
            ticket_id,
            ticket.subject,
            ticket.description,
        )

    job = await enqueue_job()

    return {
        "status": "enqueued",
        "ticket_id": ticket_id,
        "job_id": job.id,
        "message": "Ticket accepted for background intelligent processing (Milestone 2)"
    }


# ==========================================================
# Milestone 3 – Full Autonomous (Redis Worker)
# ==========================================================

@app.post("/ticket", status_code=202)
def add_ticket(ticket: InputTicket):
    ticket_id = generate_ticket_id()

    job = ticket_queue.enqueue(
        process_ticket_m3,
        ticket_id,
        ticket.subject,
        ticket.description,
    )

    return {
        "status": "enqueued",
        "ticket_id": ticket_id,
        "job_id": job.id,
        "message": "Ticket accepted for autonomous routing (Milestone 3)"
    }


# ==========================================================
# Reset Circuit Breaker
# ==========================================================

@app.get("/reset_fallback")
def reset_fallback():
    worker.circuit_state = worker.CB_CLOSED
    worker.last_failure_time = None

    return {
        "status": "circuit breaker reset",
        "state": worker.circuit_state
    }


# ==========================================================
# View Priority Queue
# ==========================================================

@app.get("/queue")
def get_queue():
    return {"queue": view_queue_2()}


@app.get("/master_incidents")
def get_master_incidents():
    try:
        raw = dedup.redis_conn.lrange(dedup.MASTER_KEY, 0, -1)
        incidents = [json.loads(i) for i in raw]
    except Exception:
        incidents = dedup.master_incidents
    return {"master_incidents": incidents}


# ==========================================================
# Run Server
# ==========================================================

if __name__ == "__main__":
    uvicorn.run("smart-support.main:app", host="0.0.0.0", port=8000)