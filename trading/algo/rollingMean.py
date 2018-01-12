import datetime
import json
import logging

import trading.algo.dbValue
import trading.config


class Frequencies(trading.algo.dbValue.DbValue):
    def __init__(self, freq):
        self.freq = freq
        super().__init__('rollingMean_{}'.format(self.freq))
        self.reset()

        logging.info('mean with a frequencies %s', self.freq)

    def insert_value(self, real_values):
        value = None
        if real_values is not None and len(real_values) >= self.freq:
            value = sum(real_values[-self.freq:]) / len(real_values[-self.freq:])
        super().insert_value(datetime.datetime.now(), value)


class RollingMean(object):
    """Class to analysis data with mean rolling."""

    def __init__(self, config_dict, db_pricing):
        """Class Initialisation."""
        logging.debug('')

        self.slow_freq = None
        self.fast_freq = None
        self.__dict__ = json.load(open(config_dict, mode='r'))
        
        self.db_value = db_pricing

        if self.__dict__.get('rollingMean') is not None:
            self.fast_freq, self.slow_freq = [
                Frequencies(i)
                for i in sorted(self.__dict__.get(
                    'rollingMean').get('frequencies'))]

        logging.info('mean with a frequencies %s',
                     {self.slow_freq.freq, self.fast_freq.freq})

    def process(self, data_value):
        """Process data, it returned 1 to buy and -1 to sell."""
        logging.debug('')

        real_values = self.db_value.get_last_values(self.slow_freq.freq)

        self.slow_freq.insert_value(real_values)
        self.fast_freq.insert_value(real_values)

        # Check values
        fast_values = self.fast_freq.get_last_values(2)
        slow_values = self.slow_freq.get_last_values(2)

        if len(fast_values) == 2 and len(slow_values) == 2:
            # Test decreasing values => SELL

            if None in fast_values or None in slow_values:
                logging.debug('Not enough data')
                return 0

            elif (fast_values[0] > slow_values[0]
                  and fast_values[1] < slow_values[1]):
                logging.warning('Fast values becomes LOWER than Slow Values')
                return -1
            # Test increasing values => BUY
            elif (fast_values[0] < slow_values[0]
                  and fast_values[1] > slow_values[1]):
                logging.warning('Fast values becomes UPPER than Slow Values')
                return 1

            else:
                logging.debug('Nothing to do')
                return 0

        logging.error('Error by reading values')
        return 0




