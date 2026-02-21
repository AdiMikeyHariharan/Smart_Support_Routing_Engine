from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

class InputTicket(BaseModel):
    subject: Optional[str] = None
    description: str

class SupportTicket(BaseModel):
    ticket_id: str = Field(..., description="Unique ticket identifier (e.g., TKT-000123)")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    subject: Optional[str] = None
    description: str
    category: Optional[str] = None  # "Billing", "Technical", "Legal"
    urgency_score: Optional[float] = None  # 0.0 to 1.0
    embedding_vector: Optional[List[float]] = None
    priority_level: Optional[int] = None  # e.g., 1-10 (derived or explicit)
    processing_status: str = "received"  # "received", "queued", "processing", "routed", "done"