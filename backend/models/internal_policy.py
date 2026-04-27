from sqlalchemy import Column, String, Text
from .base import Base


class InternalPolicy(Base):
    __tablename__ = "internal_policies"

    policy_id               = Column(String(20),  primary_key=True)
    policy_name             = Column(String(255), nullable=False)
    policy_category         = Column(String(50),  nullable=False, index=True)
    regulated_entity_type   = Column(String(100), nullable=False)
    policy_version          = Column(String(10),  nullable=False)
    effective_date          = Column(String(15),  nullable=True)
    last_reviewed_date      = Column(String(15),  nullable=True)
    policy_content          = Column(Text,         nullable=False)   # Goes into Qdrant too
    status                  = Column(String(20),  nullable=False, default="Active")
    owner_team              = Column(String(100), nullable=True)

    def __repr__(self):
        return f"<InternalPolicy {self.policy_id} | {self.policy_name} | {self.status}>"
