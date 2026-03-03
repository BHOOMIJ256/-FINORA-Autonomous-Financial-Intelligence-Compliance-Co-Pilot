"""
Pydantic schemas for request/response validation.
TODO: Define schemas as each feature is built.
"""
from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class HealthResponse(BaseModel):
    status: str
    app: str


class TransactionBase(BaseModel):
    transaction_id: str
    user_id: str
    amount: float
    merchant: str
    category: str
    timestamp: datetime
    location: str


class AuditLog(BaseModel):
    agent: str
    action: str
    reasoning: str
    confidence: float
    timestamp: datetime
    input_ref: Optional[str] = None
