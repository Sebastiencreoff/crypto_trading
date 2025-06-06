import datetime
import logging

import sqlobject


class MaxLost(sqlobject.SQLObject):
    """Db used to store data."""
    transaction_id = sqlobject.col.IntCol()
    buy_value = sqlobject.col.FloatCol()
    min_gain = sqlobject.col.FloatCol()
    min_value = sqlobject.col.FloatCol()

    __PERCENTAGE__ = 5
    __UPDATE__ = 1

    def process(self, current_value):

        # Security Selling.
        if self.min_value >= current_value:
            logging.error('SELL: value {} < Security min value {}'.format(
                current_value, self.min_value))
            return True

        # Update Percentage.
        percentage = current_value / self.buy_value * 100
        if percentage > self.min_gain + self.__PERCENTAGE__:
            self.min_gain += self.__UPDATE__
            self.min_value = (self.buy_value * self.min_gain / 100)
            logging.warning(
                'Update Security Lost: min value {}, '
                'min gain: {}, current gain:{}'.format(
                    self.min_value, self.min_gain, percentage))
