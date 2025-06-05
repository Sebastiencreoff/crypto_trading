import datetime
import logging

import sqlobject


def get_values(db_conn, currency, start_date_time, stop_date_time): # Added db_conn
    """Get values."""
    logging.debug(f"Fetching prices for {currency} between {start_date_time} and {stop_date_time}")
    try:
        return Pricing.select(
            sqlobject.AND(
                Pricing.q.currency == currency,
                Pricing.q.date_time.between(start_date_time, stop_date_time)
            ),
            connection=db_conn # Use db_conn
        )
    except Exception as e:
        logging.error(f"Error fetching values for {currency}: {e}", exc_info=True)
        return []


def get_last_values(db_conn, currency, count=None): # Added db_conn
    """Get last values."""
    logging.debug(f'Fetching last {count if count else "all"} prices for {currency}')
    try:
        pricing_query = Pricing.select(
            Pricing.q.currency == currency,
            connection=db_conn # Use db_conn
        ).orderBy(Pricing.q.date_time)

        if count:
            # Ensure count is positive integer if specified
            if not isinstance(count, int) or count <= 0:
                logging.warning(f"Invalid count '{count}' for get_last_values, fetching all for {currency}.")
                pass # Fetch all if count is invalid
            else:
                pricing_query = pricing_query[-count:]

        values = [x.value for x in pricing_query]
        logging.debug(f'Found {len(values)} price(s) for {currency}.')
        return values
    except Exception as e:
        logging.error(f"Error fetching last values for {currency}: {e}", exc_info=True)
        return []


def reset(db_conn, currency): # Changed self to db_conn, added currency argument
    """Reset pricing database for a given currency."""
    logging.info(f'Resetting pricing data for currency: {currency}')
    try:
        Pricing.deleteMany(Pricing.q.currency == currency, connection=db_conn) # Use db_conn
        logging.info(f'Successfully reset pricing data for {currency}.')
    except Exception as e:
        logging.error(f"Error resetting pricing data for {currency}: {e}", exc_info=True)


class Pricing(sqlobject.SQLObject):
    """Db used to store data."""

    DATE_TIME_FORMAT = '%Y-%m-%dT%H:%M:%S'

    date_time = sqlobject.col.DateTimeCol(
        default=sqlobject.col.DateTimeCol.now())
    value = sqlobject.col.FloatCol(default=None)
    currency = sqlobject.col.StringCol()

    # Optional: Add index for faster queries
    # class sqlmeta:
    #     indices = [
    #         sqlobject.dbIndex('currency', 'date_time'),
    #     ]
