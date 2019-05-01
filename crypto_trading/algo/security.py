import json
import logging

from . import model


class Security(object):
    """Class to thread hold to sell when the lost in a transaction is
    too high."""

    def __init__(self, config_dict):
        """Class Initialisation."""
        logging.debug('')
        self.__dict__ = json.load(open(config_dict, mode='r'))

        self.mean = None
        self.percentage = None
        self.value = None
        self.security = None
        if self.__dict__.get('maxLost') is not None:
            self.percentage = self.__dict__.get('maxLost').get('percentage')
            self.percentage_update = self.__dict__.get('maxLost').get(
                'percentage_update', 1)
            self.mean = self.__dict__.get('maxLost').get('mean', 200)

        if self.mean:
            model.create()

    def process(self, current_value, transaction):
        """Process data, it returned 1 to buy and -1 to sell."""
        logging.debug('')

        # Get last security
        if not self.security or self.security.transaction_id != transaction.id:
            result = model.Security.select(
                model.Security.q.transaction_id == transaction.id)
            self.security = result[0] if result.count() else None
        if not self.security:
            self.security = model.Security(
                buy_value=transaction.buy_value,
                min_gain=100 - self.percentage,
                min_value=(transaction.currency_buy_value
                           * (100 -  self.percentage)/100),
                transaction_id=transaction.id)

        if self.security.min_value >= current_value:
            logging.error('SELL: value {} < Security min value {}'.format(
                current_value, self.security.min_value))
            return True

        percentage = current_value / transaction.currency_buy_value * 100
        if percentage > self.security.min_gain + self.percentage:
            self.security.min_gain += self.percentage_update
            self.security.min_value = (self.security.buy_value
                                       * self.security.min_gain/100)
            logging.warning('Update Security Lost: value {}, gain:{}'.format(
                self.security.min_gain, self.security.min_value))

        # Sell if value goes below this mean.
        if self.mean:
            real_values = model.pricing.get_last_values(
                count=self.mean,
                currency=transaction.currency)

            avg = model.rolling_mean_pricing.insert_value(
                currency=transaction.currency,
                frequency=self.mean,
                values=real_values)

            if avg.value and current_value <= avg.value:
                logging.error('SELL: Current Value: {} lower than mean {} '
                              'at frequency: {}'.format(
                                current_value, avg.value, self.mean))
                return True

        return False



