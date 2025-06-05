import datetime
import logging
from sqlalchemy.orm import Session
from sqlalchemy import desc # For ordering

# Assuming models are in .models relative to this file if this file is in crypto_trading/database/
from .models import PriceTick, RollingMeanPricing, MaxLost, TradingTransaction

logger = logging.getLogger(__name__)

# --- PriceTick related operations (formerly from algo/model/pricing.py) ---

def get_price_ticks_in_range(session: Session, currency_pair: str,
                             start_date_time: datetime.datetime,
                             stop_date_time: datetime.datetime) -> list[PriceTick]:
    """Retrieves PriceTick records for a currency_pair within a given datetime range."""
    try:
        return session.query(PriceTick).filter(
            PriceTick.currency_pair == currency_pair,
            PriceTick.timestamp >= start_date_time,
            PriceTick.timestamp <= stop_date_time
        ).order_by(PriceTick.timestamp).all()
    except Exception as e:
        logger.error(f"Error getting price ticks in range for {currency_pair}: {e}", exc_info=True)
        raise

def get_last_price_ticks(session: Session, currency_pair: str, count: int = None) -> list[PriceTick]:
    """Retrieves the last 'count' PriceTick records for a currency_pair, ordered by timestamp descending."""
    try:
        query = session.query(PriceTick).filter(
            PriceTick.currency_pair == currency_pair
        ).order_by(desc(PriceTick.timestamp))

        if count is not None and count > 0:
            query = query.limit(count)

        # Results are in descending order, reverse to get chronological for typical use
        return query.all()[::-1]
    except Exception as e:
        logger.error(f"Error getting last price ticks for {currency_pair}: {e}", exc_info=True)
        raise

def reset_price_ticks(session: Session, currency_pair: str = None):
    """Deletes PriceTick records. Can be filtered by currency_pair."""
    try:
        query = session.query(PriceTick)
        if currency_pair:
            query = query.filter(PriceTick.currency_pair == currency_pair)

        num_deleted = query.delete(synchronize_session=False)
        session.commit() # Or commit can be handled by the caller context
        logger.warning(f"Deleted {num_deleted} price tick(s) for currency {currency_pair if currency_pair else 'all'}.")
        return num_deleted
    except Exception as e:
        session.rollback()
        logger.error(f"Error resetting price ticks: {e}", exc_info=True)
        raise

# --- RollingMeanPricing related operations (formerly from algo/model/rolling_mean_pricing.py) ---

def save_rolling_mean(session: Session, currency_pair: str, frequency: int, value: float,
                      timestamp: datetime.datetime = None) -> RollingMeanPricing:
    """Saves a new rolling mean value to the database."""
    if timestamp is None:
        timestamp = datetime.datetime.utcnow()
    try:
        mean_entry = RollingMeanPricing(
            currency_pair=currency_pair,
            frequency=frequency,
            value=value,
            timestamp=timestamp
        )
        session.add(mean_entry)
        session.commit() # Or commit can be handled by the caller context
        logger.debug(f"Saved rolling mean for {currency_pair}, freq {frequency}: {value} at {timestamp}")
        return mean_entry
    except Exception as e:
        session.rollback()
        logger.error(f"Error saving rolling mean for {currency_pair}, freq {frequency}: {e}", exc_info=True)
        raise

def get_last_rolling_means(session: Session, currency_pair: str, frequency: int, count: int = None) -> list[RollingMeanPricing]:
    """Retrieves the last 'count' RollingMeanPricing records for a currency_pair and frequency."""
    try:
        query = session.query(RollingMeanPricing).filter(
            RollingMeanPricing.currency_pair == currency_pair,
            RollingMeanPricing.frequency == frequency
        ).order_by(desc(RollingMeanPricing.timestamp))

        if count is not None and count > 0:
            query = query.limit(count)

        return query.all()[::-1] # Results are in descending order, reverse for chronological
    except Exception as e:
        logger.error(f"Error getting last rolling means for {currency_pair}, freq {frequency}: {e}", exc_info=True)
        raise

def reset_rolling_means(session: Session, currency_pair: str = None, frequency: int = None):
    """Deletes RollingMeanPricing records. Can be filtered by currency_pair and/or frequency."""
    try:
        query = session.query(RollingMeanPricing)
        if currency_pair:
            query = query.filter(RollingMeanPricing.currency_pair == currency_pair)
        if frequency is not None:
            query = query.filter(RollingMeanPricing.frequency == frequency)

        num_deleted = query.delete(synchronize_session=False)
        session.commit() # Or commit can be handled by the caller context
        logger.warning(f"Deleted {num_deleted} rolling mean(s) for {currency_pair if currency_pair else 'all'}, freq {frequency if frequency is not None else 'all'}.")
        return num_deleted
    except Exception as e:
        session.rollback()
        logger.error(f"Error resetting rolling means: {e}", exc_info=True)
        raise

# --- MaxLost related operations (related to former algo/model/security.py/MaxLost) ---

def get_or_create_max_lost_setting(session: Session, transaction: TradingTransaction,
                                initial_buy_price: float,
                                default_min_gain_percentage: float,
                                default_min_value_threshold_factor: float # e.g., 0.95 for 5% loss from buy_price
                                ) -> MaxLost:
    """
    Retrieves an existing MaxLost setting for a transaction or creates a new one
    if it doesn't exist.
    `default_min_value_threshold_factor` is multiplied by initial_buy_price.
    """
    try:
        max_lost_setting = session.query(MaxLost).filter(MaxLost.transaction_id == transaction.id).first()

        if not max_lost_setting:
            min_value = initial_buy_price * default_min_value_threshold_factor
            max_lost_setting = MaxLost(
                transaction_id=transaction.id, # or trading_transaction=transaction
                initial_base_currency_buy_price=initial_buy_price,
                current_min_gain_percentage=default_min_gain_percentage,
                current_min_value_threshold=min_value
            )
            session.add(max_lost_setting)
            # No commit here, assume caller (e.g., Security algo) will commit as part of its unit of work.
            # session.commit()
            logger.info(f"Created new MaxLost setting for transaction {transaction.id}")
        return max_lost_setting
    except Exception as e:
        # session.rollback() # Rollback might be handled by caller if part of larger transaction
        logger.error(f"Error getting or creating MaxLost setting for transaction {transaction.id}: {e}", exc_info=True)
        raise

# Note: Updating MaxLost instances (the logic from the old MaxLost.process) will typically be:
# 1. Fetch MaxLost instance using get_or_create_max_lost_setting.
# 2. Modify its attributes directly (e.g., max_lost.current_min_gain_percentage = new_value).
# 3. The calling code (e.g., Security algo) then calls session.commit() to persist changes.
