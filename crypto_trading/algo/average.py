import datetime
# import json # Removed as no longer needed by GuppyMMA directly
import logging

from . import model
from .algoIf import AlgoIf # Added import


class GuppyMMA(AlgoIf): # Changed inheritance
    """Class to analysis data with mean rolling."""

    SHORT_TERM_DFT = [3, 5, 8, 10, 12, 15]
    LONG_TERM_DFT = [30, 35, 40, 45, 50, 60]

    def __init__(self, guppy_config): # Parameter is now the GuppyMMA specific config dict
        """Class Initialisation."""
        logging.debug(f"Initializing GuppyMMA with config: {guppy_config}")

        self.active = False
        self.buy = None
        self.long_terms = None
        self.short_terms = None
        self.sell = None

        # Check if guppy_config is not empty, indicating the 'GuppyMMA' section was present and passed
        if guppy_config:
            # Consider adding an 'enabled' flag within the guppy_config itself for explicit control
            self.active = guppy_config.get('enabled', True) # Default to True if section exists but no 'enabled' key

            if self.active:
                self.short_terms = [i
                                    for i in sorted(guppy_config.get('short_term',
                                                                     self.SHORT_TERM_DFT))]
                self.long_terms = [i
                                   for i in sorted(guppy_config.get('long_term',
                                                                    self.LONG_TERM_DFT))]

                # Default 'buy'/'sell' trigger points to the number of short-term MAs if not specified in config.
                # This means all short-term MAs must align for a potential signal by default.
                default_buy_sell_trigger = len(self.short_terms) if self.short_terms else 0
                self.buy = guppy_config.get('buy', default_buy_sell_trigger)
                self.sell = guppy_config.get('sell', default_buy_sell_trigger)

                logging.info('GuppyMMA active with frequencies %s', self.short_terms + self.long_terms)
            else:
                logging.info("GuppyMMA is configured but not active (enabled: false).")
        else:
            logging.info("GuppyMMA configuration section not found or is empty. GuppyMMA will be inactive.")
            self.active = False # Ensure inactive if no config

    def max_frequencies(self):
        if not self.active or not self.long_terms: # Check if active and long_terms is initialized
            return 0 # Return a sensible default like 0 or None
        return max(self.long_terms)

    def process_freq(self, frequencies, currency, historical_values): # Renamed 'values'
        min_val = None
        max_val = None
        return_values = {}
        if not frequencies: # Handle case where frequencies might be empty
            return min_val, max_val, return_values

        for freq in frequencies:
            avg = model.rolling_mean_pricing.insert_value(currency=currency,
                                                          frequency=freq,
                                                          values=historical_values) # Renamed 'values'
            logging.debug('MMA: {}, value:{}'.format(avg.frequency, avg.value))
            if avg.value is not None: # Ensure avg.value is not None before comparison
                if min_val is None or avg.value < min_val:
                    min_val = avg.value
                if max_val is None or avg.value > max_val:
                    max_val = avg.value
                return_values[avg.frequency] = avg.value

        return min_val, max_val, return_values

    # Signature changed to match AlgoIf.process
    def process(self, current_value, historical_values, currency, **kwargs): # Renamed 'values', added **kwargs
        """Process data, it returned 1 to buy and -1 to sell."""

        if not self.active:
            return 0

        # Ensure terms are initialized before processing
        if not self.short_terms or not self.long_terms:
            logging.warning("GuppyMMA short_terms or long_terms not initialized. Skipping process.")
            return 0

        logging.debug('Processing GuppyMMA')
        short_min, short_max, short_values = self.process_freq(
            self.short_terms, currency, historical_values) # Renamed 'values'
        long_min, long_max, long_values = self.process_freq(
            self.long_terms, currency, historical_values) # Renamed 'values'

        # Ensure we have all necessary data points to avoid errors
        if not short_values or not long_values or \
           len(long_values) != len(self.long_terms) or \
           any(st not in short_values for st in self.short_terms[:3]) or \
           any(lt not in long_values for lt in self.long_terms[:3]):
            logging.debug('GuppyMMA: Not enough data points from process_freq. Skipping analysis.')
            return 0

        if short_max is None or long_max is None: # Check if max values were set
             logging.debug('GuppyMMA: short_max or long_max is None. Skipping analysis.')
             return 0

        count = 0
        for k, v in short_values.items():
            if v and v >= long_max: # long_max should not be None here
                count += 1
                logging.debug('Short Frequency: {} >= {}'.format(
                    {k: v}, long_max))

        model.guppy_mma.Guppy(count=count,
                              currency=currency,
                              date_time=datetime.datetime.now())

        previous_guppy = model.guppy_mma.get_last_values(count=3,
                                                         currency=currency)

        if not len(previous_guppy): # or len(previous_guppy) < 3 for some checks
            logging.debug('Not enough previous Guppy data')
            return 0

        # Test increasing values => BUY
        # Ensure all required keys are present before accessing
        if (short_values[self.short_terms[0]] >= short_values[self.short_terms[1]] >= short_values[self.short_terms[2]]
                and long_values[self.long_terms[0]] >= long_values[self.long_terms[1]] >= long_values[self.long_terms[2]]
                and all(x.count == self.buy for x in previous_guppy[1:]) # previous_guppy must have at least 2 elements for previous_guppy[1:]
                and (len(previous_guppy) > 0 and previous_guppy[0].count != self.buy)): # Check len before accessing previous_guppy[0]
            logging.info('Guppy buy limit reached')
            return 1

        # Test decreasing values => SELL
        if (all(x.count <= self.sell for x in previous_guppy)
                and long_values[self.long_terms[0]] <= long_values[self.long_terms[1]] <= long_values[self.long_terms[2]]):
            logging.info('Guppy sell limit reached')
            return -1

        logging.debug('GuppyMMA: Nothing to do')
        return 0
