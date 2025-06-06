import datetime
import logging
from sqlalchemy.orm import Session
from sqlalchemy import and_
from crypto_trading.database.models import PriceTick


def get_values(session: Session, currency_pair: str, start_date_time: datetime.datetime, stop_date_time: datetime.datetime):
    """Get values."""
    logging.debug(f"Fetching prices for {currency_pair} between {start_date_time} and {stop_date_time}")
    try:
        return session.query(PriceTick).filter(
            and_(
                PriceTick.currency_pair == currency_pair,
                PriceTick.timestamp.between(start_date_time, stop_date_time)
            )
        ).order_by(PriceTick.timestamp).all()
    except Exception as e:
        logging.error(f"Error fetching values for {currency_pair}: {e}", exc_info=True)
        return []


def get_last_values(session: Session, currency_pair: str, count: int = None):
    """Get last values."""
    logging.debug(f'Fetching last {count if count else "all"} prices for {currency_pair}')
    try:
        query = session.query(PriceTick.price).filter(
            PriceTick.currency_pair == currency_pair
        ).order_by(PriceTick.timestamp.desc())

        if count:
            if not isinstance(count, int) or count <= 0:
                logging.warning(f"Invalid count '{count}' for get_last_values, fetching all for {currency_pair}.")
            else:
                query = query.limit(count)

        # Fetch price values and reverse to maintain ascending order of time
        price_values = [result[0] for result in query.all()]
        return price_values[::-1]
    except Exception as e:
        logging.error(f"Error fetching last values for {currency_pair}: {e}", exc_info=True)
        return []


def reset(session: Session, currency_pair: str):
    """Reset pricing database for a given currency."""
    logging.info(f'Resetting pricing data for currency: {currency_pair}')
    try:
        num_deleted = session.query(PriceTick).filter(PriceTick.currency_pair == currency_pair).delete(synchronize_session=False)
        # session.commit() # Important: Commit should be handled by the caller of this function
        logging.info(f'Successfully deleted {num_deleted} PriceTick entries for {currency_pair}.')
    except Exception as e:
        # session.rollback() # Rollback should be handled by the caller
        logging.error(f"Error resetting PriceTick data for {currency_pair}: {e}", exc_info=True)
        raise # Re-raise the exception to allow caller to handle transaction
