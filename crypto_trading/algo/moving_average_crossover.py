import logging
import json
import numpy as np
from .algoIf import AlgoIf

class MovingAverageCrossover(AlgoIf):
    """
    Trading algorithm based on moving average crossovers.
    """
    DEFAULT_SHORT_WINDOW = 20
    DEFAULT_LONG_WINDOW = 50

    def __init__(self, config_section_or_path):
        """
        Initializes the MovingAverageCrossover algorithm.
        Args:
            config_section_or_path (dict or str): The configuration dictionary section
                                                  for this algo, or path to the main JSON config file.
        """
        logging.debug("Initializing MovingAverageCrossover")
        self.short_window = self.DEFAULT_SHORT_WINDOW
        self.long_window = self.DEFAULT_LONG_WINDOW
        self.config_section = {}

        if isinstance(config_section_or_path, str):
            logging.info(f"MovingAverageCrossover loading config from path: {config_section_or_path}")
            try:
                with open(config_section_or_path, mode='r') as f:
                    full_config = json.load(f)
                self.config_section = full_config.get("MovingAverageCrossover", {})
            except FileNotFoundError:
                logging.error(f"Configuration file not found: {config_section_or_path}. MAC will use defaults.")
                self.config_section = {}
            except json.JSONDecodeError:
                logging.error(f"Error decoding JSON from {config_section_or_path}. MAC will use defaults.")
                self.config_section = {}
        elif isinstance(config_section_or_path, dict):
            logging.info("MovingAverageCrossover loading config from provided dictionary section.")
            self.config_section = config_section_or_path
        else:
            logging.error("MovingAverageCrossover config is neither a path nor a dictionary. Using defaults.")
            self.config_section = {}

        self.update_config(self.config_section)

    def update_config(self, config_section: dict):
        """Updates the algorithm's parameters from the given config section."""
        logging.debug(f"MovingAverageCrossover updating config with section: {config_section}")
        self.config_section = config_section

        if not self.config_section:
            logging.warning("MovingAverageCrossover config section is empty. Using default windows.")
            self.short_window = self.DEFAULT_SHORT_WINDOW
            self.long_window = self.DEFAULT_LONG_WINDOW
        else:
            self.short_window = self.config_section.get("short_window", self.DEFAULT_SHORT_WINDOW)
            self.long_window = self.config_section.get("long_window", self.DEFAULT_LONG_WINDOW)

        if self.short_window >= self.long_window:
            logging.error(f"Short window ({self.short_window}) must be less than long window ({self.long_window}) for MovingAverageCrossover. Reverting to defaults.")
            self.short_window = self.DEFAULT_SHORT_WINDOW
            self.long_window = self.DEFAULT_LONG_WINDOW

        logging.info(
            f"MovingAverageCrossover configured with short_window: {self.short_window}, long_window: {self.long_window}"
        )

    def max_frequencies(self):
        """
        Returns the maximum number of historical data points needed.
        """
        return self.long_window

    def _calculate_sma(self, data, window):
        """
        Calculates the Simple Moving Average (SMA).
        """
        if not data or len(data) < window or window <= 0: # Added window <= 0 check
            return None
        return np.mean(data[-window:])

    def process(self, current_value, values, currency):
        """
        Process data to generate trading signals.
        Args:
            current_value (float): The current price of the asset. (Currently unused in MA calc)
            values (list[float]): A list of historical closing prices.
                                   The list is expected to be ordered from oldest to newest.
            currency (str): The currency pair being traded (e.g., "BTC-USD").
        Returns:
            int: 1 to buy, -1 to sell, 0 for no action.
        """
        logging.debug(f"Processing MovingAverageCrossover for {currency} with current value {current_value}")

        if len(values) < self.long_window: # Ensure enough data for the longest window
            logging.info(f"Not enough data for MAC. Need {self.long_window}, got {len(values)}.")
            return 0

        numeric_values = []
        try:
            # Attempt direct float conversion first
            numeric_values = [float(v) for v in values]
        except (ValueError, TypeError):
            try:
                # If direct fails, try accessing a .value attribute
                numeric_values = [float(v.value) for v in values]
            except (AttributeError, TypeError, ValueError) as e: # Catch more specific errors
                logging.error(f"Could not convert values to float for MAC: {values}. Error: {e}")
                return 0

        if not numeric_values: # Should not happen if initial checks are okay, but as safeguard
            logging.error("Numeric values list is empty after conversion attempts.")
            return 0

        # Short MA for current and previous period
        # For previous period, we use values up to index -2 (excluding the last one)
        sma_short_current = self._calculate_sma(numeric_values, self.short_window)
        sma_short_previous = self._calculate_sma(numeric_values[:-1], self.short_window)

        # Long MA for current and previous period
        sma_long_current = self._calculate_sma(numeric_values, self.long_window)
        sma_long_previous = self._calculate_sma(numeric_values[:-1], self.long_window)

        if sma_short_current is None or sma_short_previous is None or \
           sma_long_current is None or sma_long_previous is None:
            logging.info("Not enough data for one or more moving averages after attempting calculation (possibly due to short history for previous MAs).")
            return 0

        logging.debug(f"SMA Short Previous: {sma_short_previous}, Current: {sma_short_current}")
        logging.debug(f"SMA Long Previous: {sma_long_previous}, Current: {sma_long_current}")

        # Buy signal: Short MA crosses above Long MA
        if sma_short_previous < sma_long_previous and sma_short_current > sma_long_current:
            logging.warning(f"MovingAverageCrossover buy signal for {currency}")
            return 1

        # Sell signal: Short MA crosses below Long MA
        if sma_short_previous > sma_long_previous and sma_short_current < sma_long_current:
            logging.warning(f"MovingAverageCrossover sell signal for {currency}")
            return -1

        return 0
