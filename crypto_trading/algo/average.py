import json
import logging

from . import model
import crypto_trading.config as cfg


class GuppyMMA(object):
    """Class to analysis data with mean rolling."""

    SHORT_TERM_DFT = [3, 5, 8, 10, 12, 15]
    LONG_TERM_DFT = [30, 35, 40, 45, 50, 60]

    def __init__(self, config_dict):
        """Class Initialisation."""
        logging.debug('')

        self.frequencies = None
        self.__dict__ = json.load(open(config_dict, mode='r'))

        if self.__dict__.get('GuppyMMA') is not None:
            self.short_terms = [i
                                for i in sorted(self.__dict__.get(
                                    'GuppyMMA').get('short_term',
                                                    self.SHORT_TERM_DFT))]
            self.long_terms = [i
                               for i in sorted(self.__dict__.get(
                                    'GuppyMMA').get('long_term',
                                                    self.LONG_TERM_DFT))]
            self.max_frequencies = max(self.long_terms)

            self.buy = self.__dict__.get('GuppyMMA').get(
                'buy', len(self.short_terms))
            self.sell = self.__dict__.get('GuppyMMA').get(
                'sell', len(self.short_terms))

        logging.info('mean with a frequencies %s',
                     self.short_terms  + self.long_terms)

    def process_freq(self, frequencies, currency):
        real_values = model.pricing.get_last_values(count=self.max_frequencies,
                                                    currency=currency)

        min_val = None
        max_val = None
        values = {}
        for freq in frequencies:
            avg = model.rolling_mean_pricing.insert_value(currency=currency,
                                                          frequency=freq,
                                                          values=real_values)
            logging.debug('MMA: {}, value:{}'.format(avg.frequency, avg.value))
            if not min_val:
                min_val = avg.value
            if not max_val:
                max_val = avg.value

            if avg.value:
                min_val = min(min_val, avg.value)
                max_val = min(max_val, avg.value)
                values[avg.frequency] = avg.value

        return min_val, max_val, values

    def process(self, data_value, currency):
        """Process data, it returned 1 to buy and -1 to sell."""
        logging.debug('')

        short_min, short_max, short_values = self.process_freq(
            self.short_terms, currency)
        long_min, long_max, long_values = self.process_freq(
            self.long_terms, currency)

        count = 0
        if long_max:
            for k, v in short_values.items():
                if v and v >= long_max:
                    count += 1
                    logging.info('Short Frequency: {} >= {}'.format(
                        {k: v}, long_max))

        previous_guppy = model.guppy_mma.get_last_values(
            count=1,
            currency=currency)

        guppy = model.guppy_mma.Guppy(count=count, currency=currency)
        logging.debug('Guppy count: {}'.format(guppy.count))

        if not len(previous_guppy):
            logging.debug('Nothing to do')
            return 0

        previous_guppy = previous_guppy[0]

        # Test increasing values => BUY
        if (guppy.count > previous_guppy.count
            and guppy.count == self.buy):
            logging.warning('Guppy buy limit reached')
            return 1

        # Test decreasing values => SELL
        elif (guppy.count < previous_guppy.count
            and guppy.count <= self.sell):
            logging.warning('Guppy sell limit reached')
            return -1

        else:
            logging.debug('Nothing to do')
            return 0

