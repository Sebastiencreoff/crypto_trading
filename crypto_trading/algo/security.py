import json
import logging


class Security(object):
    """Class to thread hold to sell when the lost in a transaction is
    too high."""

    def __init__(self, config_dict):
        """Class Initialisation."""
        logging.debug('')
        self.__dict__ = json.load(open(config_dict, mode='r'))

        self.percentage = None
        self.value = None
        if self.__dict__.get('maxLost') is not None:
            self.percentage = self.__dict__.get('maxLost').get('percentage')
            self.value = self.__dict__.get('maxLost').get('value')

    def process(self, buy_value, current_value):
        """Process data, it returned 1 to buy and -1 to sell."""
        logging.debug('')

        if current_value < buy_value:
            if (self.percentage
                    and (100 - current_value / buy_value * 100) > self.percentage):

                logging.error('SELL: Current Percentage {} upper than '
                              'max allowed {}'.format(
                                current_value, self.percentage))
                return True

            if self.value and buy_value - current_value > self.value:
                logging.error('SELL: Current Value Lost: {} '
                              'Max Value Lost allowed: {}'.format(
                                buy_value - current_value, self.value))
                return True
        return False



