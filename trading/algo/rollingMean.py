import datetime
import json
import logging

import trading.algo.model as model
import trading.config as cfg


class RollingMean(object):
    """Class to analysis data with mean rolling."""

    def __init__(self, config_dict):
        """Class Initialisation."""
        logging.debug('')

        self.slow_freq = None
        self.fast_freq = None
        self.__dict__ = json.load(open(config_dict, mode='r'))

        if self.__dict__.get('rollingMean') is not None:
            self.fast_freq, self.slow_freq = [
                i
                for i in sorted(self.__dict__.get(
                    'rollingMean').get('frequencies'))]

        logging.info('mean with a frequencies %s',
                     {self.slow_freq, self.fast_freq})

    def process(self, data_value):
        """Process data, it returned 1 to buy and -1 to sell."""
        logging.debug('')

        currency = cfg.conf.currency
        real_values = model.pricing.get_last_values(count=self.slow_freq,
                                                    currency=currency)

        model.rolling_mean_pricing.insert_value(currency=currency,
                                                frequency=self.slow_freq,
                                                values=real_values)
        model.rolling_mean_pricing.insert_value(currency=currency,
                                                frequency=self.fast_freq,
                                                values=real_values)

        # Check values
        fast_values = model.rolling_mean_pricing.get_last_values(
            currency=currency,
            count=2,
            frequency=self.fast_freq)
        slow_values = model.rolling_mean_pricing.get_last_values(
            currency=currency,
            count=2,
            frequency=self.slow_freq)

        if len(fast_values) == 2 and len(slow_values) == 2:
            # Test decreasing values => SELL

            if None in fast_values or None in slow_values:
                logging.debug('Not enough data')
                return 0

            elif (fast_values[0].value > slow_values[0].value
                  and fast_values[1].value < slow_values[1].value):
                logging.warning('Fast values becomes LOWER than Slow Values')
                return -1
            # Test increasing values => BUY
            elif (fast_values[0].value < slow_values[0].value
                  and fast_values[1].value > slow_values[1].value):
                logging.warning('Fast values becomes UPPER than Slow Values')
                return 1

            else:
                logging.debug('Nothing to do')
                return 0

        logging.error('Error by reading values')
        return 0




