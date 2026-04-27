from sqlalchemy import Column, String, Float
from .base import Base


class Client(Base):
    __tablename__ = "clients"

    client_id                   = Column(String(10),  primary_key=True)
    client_name                 = Column(String(255), nullable=False)
    sector                      = Column(String(100), nullable=False, index=True)
    product_type                = Column(String(100), nullable=False)
    exposure_short_term         = Column(Float,        nullable=True)
    exposure_long_term          = Column(Float,        nullable=True)
    credit_line_total           = Column(Float,        nullable=True)
    credit_line_utilized_pct    = Column(Float,        nullable=True)
    collateral_type             = Column(String(20),  nullable=True)
    risk_rating                 = Column(String(10),  nullable=False)
    last_reviewed_date          = Column(String(15),  nullable=True)
    payment_history             = Column(String(30),  nullable=True)
    regulatory_sensitivity      = Column(String(10),  nullable=True)

    def __repr__(self):
        return f"<Client {self.client_id} | {self.client_name} | {self.sector}>"
