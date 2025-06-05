import datetime
import logging

import pandas
import sqlobject

from .pricing import Pricing # This import might be fine if Pricing is just used as a base class type

# It's generally better if model files define their own SQLObject classes
# rather than inheriting from one in another model file if the tables are distinct.
# However, if RollingMeanPricing is meant to extend the Pricing table or share its structure,
# this inheritance is okay. For now, assume the structure is intentional.

class RollingMeanPricing(Pricing): # Inherits from Pricing SQLObject class
    frequency = sqlobject.col.IntCol()
    # No need to redefine date_time, value, currency as they are inherited from Pricing.

    # Optional: Index for faster queries
    # class sqlmeta:
    #     indices = [
    #         sqlobject.dbIndex('currency', 'frequency', 'date_time'),
    #     ]

def insert_value(db_conn, currency, frequency, values): # Added db_conn
    """Calculates mean from provided values and inserts a RollingMeanPricing record."""

    if not values or len(values) < frequency:
        logging.warning(f"Not enough values (provided {len(values)}, need {frequency}) for rolling mean for {currency}, freq {frequency}. No value inserted.")
        return None

    try:
        # Calculate mean of the last 'frequency' number of values
        mean_value = pandas.Series(values[-frequency:]).mean().item()

        kwargs = {
            'currency': currency,
            'date_time': datetime.datetime.now(), # Timestamp of calculation
            'frequency': frequency,
            'value': mean_value, # This is the calculated mean, not a raw price
            'connection': db_conn # Add connection for SQLObject instantiation
        }

        newInstance = RollingMeanPricing(**kwargs)
        logging.debug(f"Inserted rolling mean for {currency}, freq {frequency}: {mean_value}")
        return newInstance
    except Exception as e:
        logging.error(f"Error inserting rolling mean for {currency}, freq {frequency}: {e}", exc_info=True)
        return None


def get_last_values(db_conn, currency, frequency, count=None): # Added db_conn
    """Get last rolling mean values with date_time increasing order."""

    logging.debug(f'Fetching last {count if count else "all"} rolling means for {currency}, freq {frequency}')
    try:
        query = RollingMeanPricing.select(
            sqlobject.AND(
                RollingMeanPricing.q.currency == currency,
                RollingMeanPricing.q.frequency == frequency
            ),
            connection=db_conn # Use db_conn
        ).orderBy(RollingMeanPricing.q.date_time)

        if count:
            if not isinstance(count, int) or count <= 0:
                logging.warning(f"Invalid count '{count}' for get_last_values (rolling mean), fetching all for {currency}, freq {frequency}.")
            else:
                query = query[-count:]

        values = [x.value for x in query] # x.value here is the mean
        logging.debug(f'Found {len(values)} rolling mean(s) for {currency}, freq {frequency}.')
        return values
    except Exception as e:
        logging.error(f"Error fetching last rolling means for {currency}, freq {frequency}: {e}", exc_info=True)
        return []


def reset(db_conn, currency=None): # Added db_conn, currency=None for consistency
    """Reset rolling mean pricing database for a given currency or all if None."""
    try:
        if currency:
            logging.info(f'Resetting rolling mean pricing data for currency: {currency}')
            RollingMeanPricing.deleteMany(RollingMeanPricing.q.currency == currency, connection=db_conn)
            logging.info(f'Successfully reset rolling mean pricing data for {currency}.')
        else:
            logging.warning('Resetting all rolling mean pricing data.')
            RollingMeanPricing.deleteMany(True, connection=db_conn) # Delete all records
            logging.info('Successfully reset all rolling mean pricing data.')
    except Exception as e:
        logging.error(f"Error resetting rolling mean pricing data (currency: {currency if currency else 'all'}): {e}", exc_info=True)

# The original RollingMeanPricing class definition was at the end, moving it to the top
# for better readability before functions that use it, if it's not already defined
# by an import that sqlobject handles specially.
# SQLObject classes are typically defined globally in their module.
# The provided snippet had it at the end, which is fine for Python but less conventional.
# For this overwrite, I've placed the class definition at the top.
