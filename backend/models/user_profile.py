from sqlalchemy import Column, String, Float
from .base import Base


class UserProfile(Base):
    __tablename__ = "user_profiles"

    user_id                     = Column(String(10),  primary_key=True)
    user_name                   = Column(String(100), nullable=False)
    home_city                   = Column(String(100), nullable=False)
    home_state                  = Column(String(100), nullable=False)
    account_type                = Column(String(20),  nullable=False)
    avg_txn_amount_90d          = Column(Float,        nullable=False)
    stddev_txn_amount_90d       = Column(Float,        nullable=False)
    top_merchant_categories     = Column(String(255), nullable=True)   # pipe-separated
    active_since                = Column(String(15),  nullable=True)
    risk_tier                   = Column(String(10),  nullable=False)

    def __repr__(self):
        return f"<UserProfile {self.user_id} | {self.user_name} | {self.risk_tier}>"
