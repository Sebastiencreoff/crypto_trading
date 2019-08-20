import datetime
import json
import logging

from . import model


class GuppyMMA(object):
    """Class to analysis data with mean rolling."""

    SHORT_TERM_DFT = [3, 5, 8, 10, 12, 15]
    LONG_TERM_DFT = [30, 35, 40, 45, 50, 60]

    def __init__(self, config_dict):
        """Class Initialisation."""
        logging.debug('')

        self.active = False
        self.buy = None
        self.long_terms = None
        self.short_terms = None
        self.sell = None

        cfg = json.load(open(config_dict, mode='r'))
        if cfg.get('GuppyMMA') is not None:
            self.active = True
            self.short_terms = [i
                                for i in sorted(cfg.get(
                                    'GuppyMMA').get('short_term',
                                                    self.SHORT_TERM_DFT))]
            self.long_terms = [i
                               for i in sorted(cfg.get(
                                    'GuppyMMA').get('long_term',
                                                    self.LONG_TERM_DFT))]

            self.buy = cfg.get('GuppyMMA').get(
                'buy', len(self.short_terms))
            self.sell = cfg.get('GuppyMMA').get(
                'sell', len(self.short_terms))

            logging.info('mean with a frequencies %s',
                         self.short_terms + self.long_terms)

    def max_frequencies(self):
        return max(self.long_terms) if self.long_terms else None

    def process_freq(self, frequencies, currency, values):
        min_val = None
        max_val = None
        values = {}
        for freq in frequencies:
            avg = model.rolling_mean_pricing.insert_value(currency=currency,
                                                          frequency=freq,
                                                          values=values)
            logging.debug('MMA: {}, value:{}'.format(avg.frequency, avg.value))
            if not min_val:
                min_val = avg.value
            if not max_val:
                max_val = avg.value

            if avg.value:
                min_val = min(min_val, avg.value)
                max_val = max(max_val, avg.value)
                values[avg.frequency] = avg.value

        return min_val, max_val, values

    def process(self, current_value, values, currency):
        """Process data, it returned 1 to buy and -1 to sell."""

        if not self.active:
            return 0

        logging.debug('')
        short_min, short_max, short_values = self.process_freq(
            self.short_terms, currency, values)
        long_min, long_max, long_values = self.process_freq(
            self.long_terms, currency, values)

        count = 0
        if len(long_values) == len(self.long_terms):
            for k, v in short_values.items():
                if v and v >= long_max:
                    count += 1
                    logging.debug('Short Frequency: {} >= {}'.format(
                        {k: v}, long_max))

            model.guppy_mma.Guppy(count=count,
                                  currency=currency,
                                  date_time=datetime.datetime.now())

            previous_guppy = model.guppy_mma.get_last_values(count=3,
                                                             currency=currency)

            if not len(previous_guppy):
                logging.debug('Nothing to do')
                return 0

            # Test increasing values => BUY
            if (short_values[18] >= short_values[30] >= short_values[48]
                    and long_values[180] >= long_values[210] >= long_values[240]
                    and all(x.count == self.buy for x in previous_guppy[1:])
                    and previous_guppy[0].count != self.buy):
                logging.info('Guppy buy limit reached')
                return 1

            # Test decreasing values => SELL
            if (all(x.count <= self.sell for x in previous_guppy)
                    and long_values[180] <= long_values[210] <= long_values[240]):
                logging.info('Guppy sell limit reached')
                return -1

        logging.debug('Nothing to do')
        return 0

