import datetime
import logging

import pandas
import sqlobject

from .pricing import Pricing


def insert_value(currency, frequency, values):

    kwargs = {'currency': currency,
              'date_time': datetime.datetime.now(),
              'frequency': frequency}

    if values is not None and len(values) >= frequency:
        kwargs['value'] = pandas.Series(values[-frequency:]).mean().item()
    return RollingMeanPricing(**kwargs)


def get_last_values(currency, frequency, count=None):
    """Get last values with date_time increasing order."""

    logging.debug('nb_values: %d', count)
    pricing = RollingMeanPricing.select(
        RollingMeanPricing.q.currency == currency
        and RollingMeanPricing.q.frequency == frequency
    ).orderBy(RollingMeanPricing.q.date_time)

    if count:
        pricing = pricing[-count:]
    return [x.value for x in pricing]


def reset(currency=None):
    """Reset database."""
    RollingMeanPricing.deleteMany(RollingMeanPricing.q.currency == currency)


class RollingMeanPricing(Pricing):
    frequency = sqlobject.col.IntCol()

