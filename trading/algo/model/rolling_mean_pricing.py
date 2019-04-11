import datetime
import logging

import sqlobject

from .pricing import Pricing


def insert_value(currency, frequency, values):
    value = None
    if values is not None and len(values) >= frequency:
        value = (sum(values[-frequency:])
                 / len(values[-frequency:]))

    return RollingMeanPricing(currency=currency,
                              date_time=datetime.datetime.now(),
                              frequency=frequency,
                              value=value)


def get_last_values(currency, frequency, count=None):
    """Get last values."""

    logging.debug('nb_values: %d', count)
    pricing = RollingMeanPricing.select(
        RollingMeanPricing.q.currency == currency
        and RollingMeanPricing.q.frequency == frequency
    ).orderBy(RollingMeanPricing.q.date_time)

    if count:
        pricing = pricing[0:count]
    return [x.value for x in pricing]


def reset(currency=None):
    """Reset database."""
    RollingMeanPricing.deleteMany(RollingMeanPricing.q.currency == currency)


class RollingMeanPricing(Pricing):
    frequency = sqlobject.col.IntCol()

