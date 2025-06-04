#! /usr/bin/env python
# -*- coding:utf-8 -*-

# import json # Removed as no longer needed
import logging

from . import model
from .algoIf import AlgoIf # Added import


class Bollinger(AlgoIf): # Changed inheritance

    __COUNT__ = 2
    __FREQUENCY_DEFAULT__ = 20 # Renamed for clarity

    def __init__(self, bollinger_config): # Parameter is now the Bollinger specific config dict
        """Class Initialisation."""
        logging.debug(f"Initializing Bollinger with config: {bollinger_config}")

        self.active = False # Default to inactive
        self.frequency = self.__FREQUENCY_DEFAULT__ # Default frequency

        if bollinger_config:
            self.active = bollinger_config.get('enabled', True) # Default to True if section exists
            if self.active:
                self.frequency = bollinger_config.get('frequency', self.__FREQUENCY_DEFAULT__)
                logging.info('Bollinger active with frequency at %s', self.frequency)
            else:
                logging.info('Bollinger is configured but not active (enabled: false). Using default frequency %s.', self.frequency)
        else:
            logging.info("Bollinger configuration section not found or is empty. Bollinger will be inactive.")
            # self.active is already False, self.frequency is already default

    def max_frequencies(self):
        return self.frequency

    # Signature changed to match AlgoIf.process
    def process(self, current_value, historical_values, currency, **kwargs): # Renamed 'values', added **kwargs
        """Process data, it returned 1 to buy and -1 to sell."""
        if not self.active: # Added active check, consistent with other algos
            return 0

        logging.debug(f'Processing Bollinger for {currency} with current_value: {current_value}')

        # Assuming model.bollinger.insert_value might use current_value along with historical 'historical_values'
        # or that 'historical_values' should already reflect the history needed for the current period.
        model.bollinger.insert_value(currency, self.frequency, historical_values) # Renamed 'values'
        results = model.bollinger.get_last_values(
            currency, self.frequency,
            count=self.__COUNT__)

        # len(historical_values) should be at least self.__COUNT__ (which is 2) for historical_values[-1] and historical_values[-2] to be valid.
        # The original code had len(values) > self.__COUNT__ which means at least 3 values.
        # For direct comparison with results from DB based on these values, 2 should be enough.
        if (len(historical_values) >= self.__COUNT__
                and len(results) == self.__COUNT__
                and all(r.lower_limit is not None and r.upper_limit is not None for r in results)): # handle None from DB

            prev_bol, current_bol = results

            latest_historical_value = historical_values[-1]
            second_latest_historical_value = historical_values[-2]

            if (prev_bol.lower_limit > second_latest_historical_value
                    and current_bol.lower_limit <= latest_historical_value):
                logging.warning('Bollinger buy limit reached based on historical values')
                return 1

            if (prev_bol.upper_limit < second_latest_historical_value
                    and current_bol.upper_limit >= latest_historical_value):
                logging.warning('Bollinger sell limit reached based on historical values')
                return -1
        elif len(historical_values) < self.__COUNT__:
            logging.debug(f"Bollinger: Not enough historical values provided (need at least {self.__COUNT__}, got {len(historical_values)}).")
        elif len(results) != self.__COUNT__:
            logging.debug(f"Bollinger: Did not retrieve {self.__COUNT__} Bollinger records from DB (got {len(results)}).")
        else:
            logging.debug("Bollinger: Conditions not met for signal generation (e.g., None limits in results).")

        return 0