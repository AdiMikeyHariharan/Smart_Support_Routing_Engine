# queue.py
import heapq
from threading import Lock
from typing import List, Tuple
from ..models import SupportTicket

# Global in-memory priority queue
# tuple: (-urgency_score, ticket_id, SupportTicket instance)
_priority_queue: List[Tuple[float, str, SupportTicket]] = []
_queue_lock = Lock()


def add_to_priority_queue(ticket: SupportTicket) -> None:
    """Add ticket to priority queue (higher urgency = comes out first)"""
    with _queue_lock:
        heapq.heappush(_priority_queue, (-ticket.urgency_score, ticket.ticket_id, ticket))


def get_next_ticket() -> SupportTicket | None:
    """Get highest priority ticket (or None if empty)"""
    with _queue_lock:
        if _priority_queue:
            _, _, ticket = heapq.heappop(_priority_queue)
            return ticket
        return None


def queue_size() -> int:
    with _queue_lock:
        return len(_priority_queue)


def view_queue() -> list:
    """For debugging / monitoring — returns copy"""
    with _queue_lock:
        return list(_priority_queue)