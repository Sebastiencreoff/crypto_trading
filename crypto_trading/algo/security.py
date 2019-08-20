import json
import logging

from . import model
from . import utils

class Security(object):
    """Class to thread hold to sell when the lost in a transaction is
    too high."""

    def __init__(self, config_dict):
        """Class Initialisation."""
        logging.debug('')
        config = json.load(open(config_dict, mode='r'))

        self.mean = None
        self.maxLost = None
        self.take_profit = None
        if config.get('maxLost') is not None:

            model.MaxLost.__PERCENTAGE__ = config.get('maxLost').get(
                'percentage')
            model.MaxLost.__UPDATE__ = config.get('maxLost').get(
                'percentage_update', 1)
            self.mean = config.get('maxLost').get('mean')

        if config.get('takeProfit') is not None:
            self.take_profit = config.get('takeProfit').get('percentage')

        if self.mean:
            model.create()

    def process(self, current_value, currency):
        logging.debug('')

        if self.mean:
            real_values = model.pricing.get_last_values(
                count=self.mean,
                currency=currency)

            model.rolling_mean_pricing.insert_value(
                currency=currency,
                frequency=self.mean,
                values=real_values)

    def sell(self, current_value, transaction):
        """Process data, it returned 1 to buy and -1 to sell."""
        logging.debug('')

        # Get last security
        if not self.maxLost or self.maxLost.transaction_id != transaction.id:
            result = model.MaxLost.select(
                model.MaxLost.q.transaction_id == transaction.id)
            self.maxLost = result[0] if result.count() else None
        if not self.maxLost:

            self.maxLost = model.MaxLost(
                buy_value=transaction.currency_buy_value,
                min_gain=100 - model.MaxLost.__PERCENTAGE__,
                min_value=(transaction.currency_buy_value
                           * (100 - model.MaxLost.__PERCENTAGE__)/100),
                transaction_id=transaction.id)

        if self.maxLost.process(current_value):
            return True

        percentage = current_value/transaction.currency_buy_value * 100
        if self.take_profit and percentage >= 100 + self.take_profit:
            logging.warning('Take Profit: value {}, gain:{}'.format(
                current_value, percentage))
            return True

        # Sell if value goes below this mean.
        if self.mean:
            avg = model.rolling_mean_pricing.get_last_values(
                transaction.currency,
                frequency=self.mean,
                count=1)[0]

            if avg and current_value <= avg:
                logging.error('SELL: Current Value: {} lower than mean {} '
                              'at frequency: {}'.format(
                                current_value, avg, self.mean))
                return True

        return False

    def buy(self, current_value, currency):

        # Buy if current_value is upper than mean and mean is increasing.
        if self.mean:
            values = model.rolling_mean_pricing.get_last_values(
                currency,
                frequency=self.mean,
                count=10)
            if all(x is not None for x in values):
                return utils.is_increasing(values)
        return False


