from sqlalchemy import Column, String, Float, Boolean, DateTime, Text
from sqlalchemy.sql import func
from .base import Base


class Transaction(Base):
    __tablename__ = "transactions"

    transaction_id          = Column(String(20),  primary_key=True)
    user_id                 = Column(String(10),  nullable=False, index=True)
    timestamp               = Column(String(25),  nullable=False)   # stored as string from CSV
    amount                  = Column(Float,        nullable=False)
    currency                = Column(String(5),    nullable=False, default="INR")
    merchant_name           = Column(String(255),  nullable=False)
    merchant_category_code  = Column(String(10),   nullable=False)
    merchant_city           = Column(String(100),  nullable=True)
    merchant_country        = Column(String(100),  nullable=False, default="India")
    channel                 = Column(String(20),   nullable=True)
    device_id               = Column(String(50),   nullable=True)
    ip_country              = Column(String(100),  nullable=True)
    is_anomaly              = Column(Boolean,       nullable=False, default=False)
    anomaly_type            = Column(String(50),   nullable=True)
    anomaly_reason          = Column(Text,          nullable=True)

    def __repr__(self):
        return f"<Transaction {self.transaction_id} | {self.user_id} | ₹{self.amount} | anomaly={self.is_anomaly}>"
