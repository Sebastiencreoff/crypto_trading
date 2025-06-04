import datetime
import json
import logging

from . import model
from .algoIf import AlgoIf


class GuppyMMA(AlgoIf):
    """Class to analysis data with mean rolling."""

    SHORT_TERM_DFT = [3, 5, 8, 10, 12, 15]
    LONG_TERM_DFT = [30, 35, 40, 45, 50, 60]

    def __init__(self, config_section_or_path):
        """Class Initialisation."""
        logging.debug("Initializing GuppyMMA")

        self.active = False
        self.buy_decision = None
        self.long_terms = [] # Initialize as empty list
        self.short_terms = [] # Initialize as empty list
        self.sell_decision = None
        self._max_freq = 0  # Default to 0
        self.config_section = {}

        if isinstance(config_section_or_path, str):
            logging.info(f"GuppyMMA loading config from path: {config_section_or_path}")
            try:
                with open(config_section_or_path, mode='r') as f:
                    full_config = json.load(f)
                self.config_section = full_config.get("GuppyMMA", {})
            except FileNotFoundError:
                logging.error(f"Configuration file not found: {config_section_or_path}. GuppyMMA will use defaults.")
                self.config_section = {}
            except json.JSONDecodeError:
                logging.error(f"Error decoding JSON from {config_section_or_path}. GuppyMMA will use defaults.")
                self.config_section = {}
        elif isinstance(config_section_or_path, dict):
            logging.info("GuppyMMA loading config from provided dictionary section.")
            self.config_section = config_section_or_path
        else:
            logging.error("GuppyMMA config is neither a path nor a dictionary. Using defaults.")
            self.config_section = {}

        self.update_config(self.config_section)

    def update_config(self, config_section: dict):
        """Updates the algorithm's parameters from the given config section."""
        logging.debug(f"GuppyMMA updating config with section: {config_section}")
        self.config_section = config_section # Store new section

        if not self.config_section: # Handles empty dict or if it was reset due to errors
            logging.warning("GuppyMMA configuration section is empty or invalid. Using default values and activating.")
            self.active = True
            self.short_terms = self.SHORT_TERM_DFT[:] # Use a copy
            self.long_terms = self.LONG_TERM_DFT[:]  # Use a copy
            self.buy_decision = len(self.short_terms)
            self.sell_decision = len(self.short_terms)
        else:
            self.active = True
            self.short_terms = sorted(self.config_section.get('short_term', self.SHORT_TERM_DFT[:]))
            self.long_terms = sorted(self.config_section.get('long_term', self.LONG_TERM_DFT[:]))
            self.buy_decision = self.config_section.get('buy', len(self.short_terms))
            self.sell_decision = self.config_section.get('sell', len(self.short_terms))

        if self.long_terms:
            self._max_freq = max(self.long_terms)
        else:
            self._max_freq = 0 # Consistent default

        logging.info(f"GuppyMMA configured: active={self.active}, short_terms: {self.short_terms}, long_terms: {self.long_terms}, _max_freq: {self._max_freq}")
        logging.info(f"GuppyMMA buy_decision: {self.buy_decision}, sell_decision: {self.sell_decision}")

    def max_frequencies(self):
        return self._max_freq

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

            if avg.value: # Ensure avg.value is not None
                min_val = min(min_val, avg.value) if min_val is not None else avg.value
                max_val = max(max_val, avg.value) if max_val is not None else avg.value
                return_values[avg.frequency] = avg.value

        return min_val, max_val, return_values

    def process(self, current_value, values, currency):
        """Process data, it returned 1 to buy and -1 to sell."""

        if not self.active:
            logging.debug("GuppyMMA is not active. Skipping process.")
            return 0

        if not self.short_terms or not self.long_terms:
            logging.warning("GuppyMMA short_terms or long_terms are not configured. Skipping process.")
            return 0

        logging.debug(f"GuppyMMA processing for {currency} with current_value {current_value}")
        short_min, short_max, short_values = self.process_freq(
            self.short_terms, currency, values)
        long_min, long_max, long_values = self.process_freq(
            self.long_terms, currency, values)

        # Ensure all values were successfully retrieved for MAs
        if len(short_values) != len(self.short_terms) or len(long_values) != len(self.long_terms):
            logging.info("GuppyMMA: Not enough data to calculate all moving averages. Skipping decision.")
            return 0

        # Ensure short_max is not None before comparison (it can be if all short_term MAs are None)
        if short_max is None or long_max is None:
            logging.info("GuppyMMA: short_max or long_max is None. Skipping decision.")
            return 0

        count = 0
        # Original logic: if len(long_values) == len(self.long_terms): # This check is now implicitly handled above
        for k, v in short_values.items():
            if v and v >= long_max: # Ensure v is not None
                count += 1
                logging.debug('Short Frequency: {} >= {}'.format(
                    {k: v}, long_max))

        model.guppy_mma.Guppy(count=count,
                                currency=currency,
                                date_time=datetime.datetime.now())

        previous_guppy = model.guppy_mma.get_last_values(count=3,
                                                            currency=currency)

        if not len(previous_guppy):
            logging.debug('GuppyMMA: No previous Guppy data. Nothing to do.')
            return 0

        # Check if terms are available for indexing (robustness)
        if len(self.short_terms) < 3 or len(self.long_terms) < 3:
            logging.warning("GuppyMMA: Not enough short/long term definitions for trend check. Skipping some rules.")
            return 0 # Or handle differently

        # Test increasing values => BUY
        # Ensure all required MA values exist before trying to access them
        required_short_keys = self.short_terms[:3]
        required_long_keys = self.long_terms[:3]
        if not all(k in short_values for k in required_short_keys) or \
           not all(k in long_values for k in required_long_keys):
            logging.debug("GuppyMMA: Not all MA values present for buy signal trend check.")
            return 0 # Cannot determine trend

        if (short_values[self.short_terms[0]] >= short_values[self.short_terms[1]] >= short_values[self.short_terms[2]]
                and long_values[self.long_terms[0]] >= long_values[self.long_terms[1]] >= long_values[self.long_terms[2]]
                and all(x.count == self.buy_decision for x in previous_guppy[1:])
                and previous_guppy[0].count != self.buy_decision):
            logging.warning(f'GuppyMMA buy signal for {currency}')
            return 1

        # Test decreasing values => SELL
        if (all(x.count <= self.sell_decision for x in previous_guppy)
                and long_values[self.long_terms[0]] <= long_values[self.long_terms[1]] <= long_values[self.long_terms[2]]):
            logging.warning(f'GuppyMMA sell signal for {currency}')
            return -1

        logging.debug('GuppyMMA: Nothing to do')
        return 0
