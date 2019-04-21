import datetime
import logging

import sqlobject


class Security(sqlobject.SQLObject):
    """Db used to store data."""
    transaction_id = sqlobject.col.IntCol()
    buy_value = sqlobject.col.FloatCol()
    min_gain = sqlobject.col.FloatCol()
    min_value = sqlobject.col.FloatCol()
