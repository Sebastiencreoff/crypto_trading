from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
import datetime

Base = declarative_base()

class TradingTransaction(Base):
    __tablename__ = 'trading_transactions'

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String, index=True, nullable=True) # To associate with a specific task run

    buy_date_time = Column(DateTime, default=datetime.datetime.utcnow)
    buy_value_eur = Column(Float, comment="Total value of the buy in quote currency, e.g., EUR or USD") # Assuming quote is EUR/USD
    buy_fee_eur = Column(Float, default=0.0)

    currency_pair = Column(String, index=True) # e.g., "BTC/USD"
    base_currency_bought_amount = Column(Float)
    base_currency_buy_price = Column(Float, comment="Price per unit of base currency at time of buy")

    sell_date_time = Column(DateTime, nullable=True)
    base_currency_sell_price = Column(Float, nullable=True, comment="Price per unit of base currency at time of sell")
    sell_fee_eur = Column(Float, default=0.0, nullable=True)

    profit_eur = Column(Float, nullable=True)

    # Relationship to MaxLost (one-to-one if a transaction has one MaxLost setting)
    max_lost_setting = relationship("MaxLost", back_populates="trading_transaction", uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<TradingTransaction(id={self.id}, pair='{self.currency_pair}', buy_time='{self.buy_date_time}', sell_time='{self.sell_date_time}')>"

class PriceTick(Base):
    __tablename__ = 'price_ticks'

    id = Column(Integer, primary_key=True, index=True)
    currency_pair = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    price = Column(Float)

    def __repr__(self):
        return f"<PriceTick(id={self.id}, pair='{self.currency_pair}', time='{self.timestamp}', price={self.price})>"

class RollingMeanPricing(Base):
    __tablename__ = 'rolling_mean_pricings'

    id = Column(Integer, primary_key=True, index=True)
    currency_pair = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    frequency = Column(Integer, comment="Period of the rolling mean, e.g., 20 for a 20-period SMA")
    value = Column(Float, comment="The calculated mean value")

    def __repr__(self):
        return f"<RollingMeanPricing(id={self.id}, pair='{self.currency_pair}', freq={self.frequency}, value={self.value})>"

class MaxLost(Base):
    __tablename__ = 'max_lost_settings'

    id = Column(Integer, primary_key=True, index=True)
    # Foreign key to link to a specific trade
    transaction_id = Column(Integer, ForeignKey('trading_transactions.id'), unique=True, nullable=False, index=True)

    initial_base_currency_buy_price = Column(Float, comment="Buy price of the base currency for this transaction")
    # Current dynamic minimum gain percentage target (e.g., starts at 5%, might increase to 6% if price moves favorably)
    current_min_gain_percentage = Column(Float)
    # Current absolute price threshold below which a sale is triggered for max loss
    current_min_value_threshold = Column(Float)

    # Percentage of gain that triggers an update to min_value_threshold (e.g., if price gains X% more than current_min_gain_percentage)
    # This was __PERCENTAGE__ in SQLObject model, which was a class var. Making it instance specific if needed, or global config.
    # For now, let's assume this logic is handled in code, not stored per record, unless it needs to be dynamic per transaction.
    # The original __PERCENTAGE__ and __UPDATE__ were class-level constants in the SQLObject model.
    # These are better handled as application logic or configuration constants rather than DB fields per row,
    # unless they need to vary for each MaxLost record independently.

    # Relationship back to TradingTransaction
    trading_transaction = relationship("TradingTransaction", back_populates="max_lost_setting")

    def __repr__(self):
        return f"<MaxLost(id={self.id}, tx_id={self.transaction_id}, threshold={self.current_min_value_threshold})>"

# Example of how to create an engine and tables (for testing or initial setup)
# This part would typically be in a main script or managed by Alembic.
if __name__ == '__main__':
    # Using SQLite in-memory for example, replace with actual DB URL
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    print("SQLAlchemy models defined and tables created in memory (for example).")
