# queue.py

import heapq
import datetime
import json
from threading import Lock
from typing import List, Tuple
from ..models import SupportTicket

# 🔹 Milestone 2 – Redis RQ Queue
import redis
from rq import Queue

redis_conn = redis.Redis(host="localhost", port=6379)
ticket_queue = Queue("default", connection=redis_conn)

# 🔹 Milestone 1 – In-memory priority queue
_priority_queue: List[Tuple[float, str, SupportTicket]] = []
_queue_lock = Lock()


def add_to_priority_queue(ticket: SupportTicket) -> None:
    if ticket.urgency_score is None:
        raise ValueError(f"Cannot prioritize ticket {ticket.ticket_id}: urgency_score is None")
    
    raw = ticket.model_dump(mode="json", exclude=None)               # ← this is the Pydantic v2 way
    ticket_data = {}
    for k, v in raw.items():
        if v is None:
            ticket_data[k] = ""
        else:
            ticket_data[k] = v

    print(f"[DEBUG add_to_priority_queue] ticket {ticket.ticket_id} → {len(ticket_data)} fields")
    if ticket_data:
        print("Fields:", ", ".join(sorted(ticket_data.keys())))
    else:
        print("WARNING: ticket_data is empty!")

    if not ticket_data:
        raise RuntimeError(f"Cannot store ticket {ticket.ticket_id}: no serializable fields")
    
    redis_conn.hset(f"ticket:{ticket.ticket_id}", mapping=ticket_data)
    redis_conn.zadd("priority_queue", {ticket.ticket_id: -ticket.urgency_score})  # max-heap


def get_next_ticket() -> SupportTicket | None:
    popped = redis_conn.zpopmax("priority_queue")
    if not popped:
        return None
    ticket_id, score = popped[0]
    raw_data = redis_conn.hgetall(f"ticket:{ticket_id}")
    if not raw_data:
        return None

    return SupportTicket(**raw_data)


def queue_size() -> int:
    return redis_conn.zcard("priority_queue")


def view_queue() -> list:
    items = redis_conn.zrange("priority_queue", 0, -1, desc=True, withscores=True)
    return [(abs(score), tid) for tid, score in items]
