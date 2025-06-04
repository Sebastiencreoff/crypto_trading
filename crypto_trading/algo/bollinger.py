#! /usr/bin/env python
# -*- coding:utf-8 -*-

import json
import logging

from . import model
from .algoIf import AlgoIf


class Bollinger(AlgoIf):

    DEFAULT_COUNT = 2 # Renamed from __COUNT_DFT__
    DEFAULT_FREQUENCY = 20 # Renamed from __FREQUENCY_DFT__

    def __init__(self, config_section_or_path):
        """Class Initialisation."""
        logging.debug("Initializing Bollinger")

        self.frequency = self.DEFAULT_FREQUENCY
        self.count = self.DEFAULT_COUNT
        self.config_section = {}

        if isinstance(config_section_or_path, str):
            logging.info(f"Bollinger loading config from path: {config_section_or_path}")
            try:
                with open(config_section_or_path, mode='r') as f:
                    full_config = json.load(f)
                self.config_section = full_config.get("Bollinger", {})
            except FileNotFoundError:
                logging.error(f"Configuration file not found: {config_section_or_path}. Bollinger will use defaults.")
                self.config_section = {}
            except json.JSONDecodeError:
                logging.error(f"Error decoding JSON from {config_section_or_path}. Bollinger will use defaults.")
                self.config_section = {}
        elif isinstance(config_section_or_path, dict):
            logging.info("Bollinger loading config from provided dictionary section.")
            self.config_section = config_section_or_path
        else:
            logging.error("Bollinger config is neither a path nor a dictionary. Using defaults.")
            self.config_section = {}

        self.update_config(self.config_section)

    def update_config(self, config_section: dict):
        """Updates the algorithm's parameters from the given config section."""
        logging.debug(f"Bollinger updating config with section: {config_section}")
        self.config_section = config_section

        if not self.config_section:
            logging.warning("Bollinger configuration section is empty or invalid. Using default frequency.")
            self.frequency = self.DEFAULT_FREQUENCY
        else:
            self.frequency = self.config_section.get('frequency', self.DEFAULT_FREQUENCY)
            # self.count = self.config_section.get('count', self.DEFAULT_COUNT) # If count were configurable

        logging.info(f"Bollinger configured with frequency: {self.frequency}")

    def max_frequencies(self):
        return self.frequency

    def process(self, current_value, values, currency):
        """Process data, it returned 1 to buy and -1 to sell."""
        logging.debug(f"Bollinger processing for {currency} with current_value {current_value}")

        if len(values) < self.frequency or len(values) < self.count + 1:
             logging.info(f"Bollinger: Not enough historical data. Need > {max(self.frequency, self.count +1)}, got {len(values)}. Skipping.")
             return 0

        model.bollinger.insert_value(currency, self.frequency, values)
        # Results are typically ordered oldest to newest by the model's get_last_values
        # For current and previous bollinger values, we need the last two.
        results = model.bollinger.get_last_values(
            currency, self.frequency,
            count=self.count)

        if len(results) < self.count:
            logging.info(f"Bollinger: Not enough Bollinger results obtained. Need {self.count}, got {len(results)}. Skipping.")
            return 0

        if not all(hasattr(x, 'lower_limit') and hasattr(x, 'upper_limit') and \
                     x.lower_limit is not None and x.upper_limit is not None for x in results):
            logging.info("Bollinger: One or more results missing lower/upper limit. Skipping.")
            return 0

        if len(values) < 2: # Need at least two prices (current and previous)
            logging.info("Bollinger: Not enough values in 'values' list for price comparison. Skipping.")
            return 0

        # Assuming results are [oldest_bol, ..., newest_bol]
        # If count is 2, results[0] is prev_bol, results[1] is current_bol
        current_bol = results[-1]
        prev_bol = results[-2]

        current_price_obj = values[-1]
        previous_price_obj = values[-2]

        try:
            current_price = float(current_price_obj.value if hasattr(current_price_obj, 'value') else current_price_obj)
            previous_price = float(previous_price_obj.value if hasattr(previous_price_obj, 'value') else previous_price_obj)
        except (ValueError, TypeError) as e:
            logging.error(f"Bollinger: Error converting prices to float: {e}. Values were: {current_price_obj}, {previous_price_obj}. Skipping.")
            return 0

        # Buy signal: price crosses lower band from below
        if previous_price < prev_bol.lower_limit and current_price > current_bol.lower_limit:
            logging.warning(f"Bollinger buy signal (crossed lower band from below) for {currency}")
            return 1

        # Original logic for buy:
        # if (prev_bol.lower_limit > previous_price and current_bol.lower_limit <= current_price):
        # This means: previous price was below previous lower band, current price is above or at current lower band.
        # This is a valid interpretation of crossing lower band from below. Let's stick to the original interpretation.
        if prev_bol.lower_limit > previous_price and current_bol.lower_limit <= current_price : # This is the original condition
            logging.warning(f"Bollinger buy signal for {currency}") # Original message
            return 1

        # Sell signal: price crosses upper band from above
        if previous_price > prev_bol.upper_limit and current_price < current_bol.upper_limit:
            logging.warning(f"Bollinger sell signal (crossed upper band from above) for {currency}")
            return -1

        # Original logic for sell:
        # if (prev_bol.upper_limit < previous_price and current_bol.upper_limit >= current_price):
        # This means: previous price was above previous upper band, current price is below or at current upper band.
        if prev_bol.upper_limit < previous_price and current_bol.upper_limit >= current_price: # This is the original condition
            logging.warning(f"Bollinger sell signal for {currency}") # Original message
            return -1

        return 0
