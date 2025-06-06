import datetime
import logging

import pandas
import sqlobject


def get_last_values(currency, frecuency, count=None):
    """Get last values."""

    logging.debug('nb_values: %d', count)
    result = Bollinger.select(
        Bollinger.q.currency == currency
    ).orderBy(Bollinger.q.date_time)

    if count:
        result = result[-count:]
    return result


def insert_value(currency, frequency, values):

    kwargs = {'currency': currency,
              'date_time': datetime.datetime.now(),
              'frequency': frequency}

    if values is not None and len(values) >= frequency:
        mean = pandas.Series(values[-frequency:]).mean().item()
        std = pandas.Series(values[-frequency:]).std().item()

        kwargs['lower_limit'] = mean - 2 * std
        kwargs['upper_limit'] = mean + 2 * std

    return Bollinger(**kwargs)


def reset(self, currency):
    """Reset database."""
    logging.info('%s', self.name)

    Bollinger.deleteMany(Bollinger.q.currency == currency)


class Bollinger(sqlobject.SQLObject):
    """Db used to store data."""

    DATE_TIME_FORMAT = '%Y-%m-%dT%H:%M:%S'

    date_time = sqlobject.col.DateTimeCol(
        default=sqlobject.col.DateTimeCol.now())
    upper_limit = sqlobject.col.FloatCol(default=None)
    lower_limit = sqlobject.col.FloatCol(default=None)
    frequency = sqlobject.col.IntCol()
    currency = sqlobject.col.StringCol()