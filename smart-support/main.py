# main.py
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException
from threading import Lock
from .models import InputTicket
from .queue.worker import process_ticket_m1, process_ticket_m2, process_ticket_m3
from .queue.queue import view_queue

app = FastAPI()

ticket_counter = 0
counter_lock = Lock()

# For Milestone 1 (synchronous)
@app.post("/ticket_m1")
def add_ticket_m1(ticket: InputTicket):
    global ticket_counter
    with counter_lock:
        ticket_counter += 1
        ticket_id = f"TKT-{ticket_counter:06d}"
    return process_ticket_m1(ticket_id, ticket.subject, ticket.description)

# For Milestone 2 (real async broker)
@app.post("/ticket_m2", status_code=202)
def add_ticket_m2(ticket: InputTicket):
    global ticket_counter
    with counter_lock:
        ticket_counter += 1
        ticket_id = f"TKT-{ticket_counter:06d}"

    # Enqueue to real Redis queue – returns immediately
    job = ticket_queue.enqueue(
        process_ticket_m2,
        ticket_id,
        ticket.subject,
        ticket.description
    )

    return {
        "status": "enqueued",
        "ticket_id": ticket_id,
        "job_id": job.id,
        "message": "Ticket accepted for background intelligent processing (Milestone 2)"
    }

# For Milestone 3 (full)
@app.post("/ticket", status_code=202)
def add_ticket(ticket: InputTicket, background_tasks: BackgroundTasks):
    global ticket_counter
    with counter_lock:
        ticket_counter += 1
        ticket_id = f"TKT-{ticket_counter:06d}"
    background_tasks.add_task(process_ticket_m3, ticket_id, ticket.subject, ticket.description)
    return {"status": "accepted"}

# Reset fallback
@app.get("/reset_fallback")
def reset_fallback():
    from .worker import use_fallback
    use_fallback = False
    return {"status": "reset"}

# View queue
@app.get("/queue")
def get_queue():
    return {"queue": view_queue()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)