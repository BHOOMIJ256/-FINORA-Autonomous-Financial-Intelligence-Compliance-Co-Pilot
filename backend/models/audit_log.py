from sqlalchemy import Column, String, Float, Text, DateTime
from sqlalchemy.sql import func
from .base import Base


class AuditLog(Base):
    __tablename__ = "audit_logs"

    log_id              = Column(String(50),  primary_key=True)
    agent_name          = Column(String(50),  nullable=False, index=True)
    trigger_type        = Column(String(50),  nullable=False)   # scheduled / manual / event
    input_summary       = Column(Text,         nullable=True)
    output_summary      = Column(Text,         nullable=True)
    full_reasoning      = Column(Text,         nullable=True)
    confidence_score    = Column(Float,        nullable=True)
    action_taken        = Column(String(50),  nullable=True)    # BLOCK / HOLD / ESCALATE / REPORT
    timestamp           = Column(String(25),  nullable=False)
    related_entity_id   = Column(String(50),  nullable=True)    # transaction_id / policy_id etc
    related_entity_type = Column(String(30),  nullable=True)    # transaction / policy / client
    query_text          = Column(Text,         nullable=True)    # if triggered by NL query

    def __repr__(self):
        return f"<AuditLog {self.log_id} | {self.agent_name} | {self.action_taken}>"
