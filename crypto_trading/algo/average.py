import datetime
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

        algo_specific_config = config_dict.get('GuppyMMA')
        if algo_specific_config is not None:
            self.active = True
            self.short_terms = [i
                                for i in sorted(algo_specific_config.get(
                                    'short_term',
                                    self.SHORT_TERM_DFT))]
            self.long_terms = [i
                               for i in sorted(algo_specific_config.get(
                                    'long_term',
                                    self.LONG_TERM_DFT))]

            self.buy = algo_specific_config.get(
                'buy', len(self.short_terms))
            self.sell = algo_specific_config.get(
                'sell', len(self.short_terms))

            logging.info('mean with a frequencies %s',
                         self.short_terms + self.long_terms)

    def update_config(self, guppy_params_dict):
        """Updates the GuppyMMA configuration."""
        logging.debug(f"Updating GuppyMMA configuration with: {guppy_params_dict}")
        self.short_terms = sorted(guppy_params_dict.get('short_term', self.short_terms))
        self.long_terms = sorted(guppy_params_dict.get('long_term', self.long_terms))
        self.buy = guppy_params_dict.get('buy', self.buy)
        self.sell = guppy_params_dict.get('sell', self.sell)
        logging.info(f"GuppyMMA configuration updated: short_terms={self.short_terms}, long_terms={self.long_terms}, buy={self.buy}, sell={self.sell}")

    def max_frequencies(self):
        return max(self.long_terms) if self.long_terms else None

    def process_freq(self, frequencies, currency, values):
        min_val = None
        max_val = None
        return_values = {}
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
                return_values[avg.frequency] = avg.value

        return min_val, max_val, return_values

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
            if (short_values[self.short_terms[0]] >= short_values[self.short_terms[1]] >= short_values[self.short_terms[2]]
                    and long_values[self.long_terms[0]] >= long_values[self.long_terms[1]] >= long_values[self.long_terms[2]]
                    and all(x.count == self.buy for x in previous_guppy[1:])
                    and previous_guppy[0].count != self.buy):
                logging.info('Guppy buy limit reached')
                return 1

            # Test decreasing values => SELL
            if (all(x.count <= self.sell for x in previous_guppy)
                    and long_values[self.long_terms[0]] <= long_values[self.long_terms[1]] <= long_values[self.long_terms[2]]):
                logging.info('Guppy sell limit reached')
                return -1

        logging.debug('Nothing to do')
        return 0
