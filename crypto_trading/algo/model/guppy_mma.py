import datetime
import logging

import sqlobject


def get_last_values(currency, count=None):
    """Get last values."""

    logging.debug('nb_values: %d', count)
    guppy = Guppy.select(
        Guppy.q.currency == currency
    ).orderBy(Guppy.q.date_time)

    if count:
        guppy = guppy[-count:]
    return guppy


class Guppy(sqlobject.SQLObject):
    """Db used to store data."""

    DATE_TIME_FORMAT = '%Y-%m-%dT%H:%M:%S'

    count = sqlobject.col.IntCol()
    currency = sqlobject.col.StringCol()
    date_time = sqlobject.col.DateTimeCol(
        default=sqlobject.col.DateTimeCol.now())

