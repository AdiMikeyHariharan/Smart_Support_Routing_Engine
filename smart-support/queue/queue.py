# queue.py

import heapq
from threading import Lock
from typing import List, Tuple
from ..models import SupportTicket

# 🔹 Milestone 1 – In-memory priority queue
_priority_queue: List[Tuple[float, str, SupportTicket]] = []
_queue_lock = Lock()


def add_to_priority_queue(ticket: SupportTicket) -> None:
    with _queue_lock:
        heapq.heappush(_priority_queue, (-ticket.urgency_score, ticket.ticket_id, ticket))


def get_next_ticket() -> SupportTicket | None:
    with _queue_lock:
        if _priority_queue:
            _, _, ticket = heapq.heappop(_priority_queue)
            return ticket
        return None


def queue_size() -> int:
    with _queue_lock:
        return len(_priority_queue)


def view_queue() -> list:
    with _queue_lock:
        return [
            (urg, tid)
            for urg, tid, _ in sorted(_priority_queue)
        ]


# 🔹 Milestone 2 – Redis RQ Queue
import redis
from rq import Queue

redis_conn = redis.Redis(host="localhost", port=6379)
ticket_queue = Queue("default", connection=redis_conn)