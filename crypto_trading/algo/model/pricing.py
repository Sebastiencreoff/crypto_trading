import datetime
import logging

import sqlobject


def get_values(currency, start_date_time, stop_date_time):
    """Get values."""

    return Pricing.select(
        Pricing.q.currency == currency
        and Pricing.q.date_time.between(start_date_time, stop_date_time))


def get_last_values(currency, count=None):
    """Get last values."""

    logging.debug('nb_values: %d', count)
    pricing = Pricing.select(
        Pricing.q.currency == currency
    ).orderBy(Pricing.q.date_time)

    if count:
        pricing = pricing[-count:]
    return [x.value for x in pricing]


def reset(self, currency):
    """Reset database."""
    logging.info('%s', self.name)

    Pricing.deleteMany(Pricing.q.currency == currency)


class Pricing(sqlobject.SQLObject):
    """Db used to store data."""

    DATE_TIME_FORMAT = '%Y-%m-%dT%H:%M:%S'

    date_time = sqlobject.col.DateTimeCol(
        default=sqlobject.col.DateTimeCol.now())
    value = sqlobject.col.FloatCol()
    currency = sqlobject.col.StringCol()



