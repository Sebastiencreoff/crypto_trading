import logging
# import json # Removed as no longer needed
import numpy as np
from .algoIf import AlgoIf

class MovingAverageCrossover(AlgoIf):
    """
    Trading algorithm based on moving average crossovers.
    """
    # Default values that can be overridden by config
    DEFAULT_SHORT_WINDOW = 20
    DEFAULT_LONG_WINDOW = 50

    def __init__(self, mac_config): # Parameter is now the MovingAverageCrossover specific config dict
        """
        Initializes the MovingAverageCrossover algorithm.

        Args:
            mac_config (dict): Configuration dictionary for this algorithm.
        """
        # No super().__init__() call as AlgoIf has no __init__
        logging.debug(f"Initializing MovingAverageCrossover with config: {mac_config}")

        self.active = False # Default to inactive
        # Initialize with defaults, then try to update from config
        self.short_window = self.DEFAULT_SHORT_WINDOW
        self.long_window = self.DEFAULT_LONG_WINDOW

        if mac_config:
            self.active = mac_config.get('enabled', True) # Default to True if section exists
            if self.active:
                self.short_window = mac_config.get("short_window", self.DEFAULT_SHORT_WINDOW)
                self.long_window = mac_config.get("long_window", self.DEFAULT_LONG_WINDOW)

                if self.short_window >= self.long_window:
                    logging.error(
                        f"MovingAverageCrossover config error: short_window ({self.short_window}) "
                        f"must be less than long_window ({self.long_window}). "
                        f"Using defaults: short={self.DEFAULT_SHORT_WINDOW}, long={self.DEFAULT_LONG_WINDOW}."
                    )
                    self.short_window = self.DEFAULT_SHORT_WINDOW
                    self.long_window = self.DEFAULT_LONG_WINDOW
                logging.info(
                    f"MovingAverageCrossover active with short_window: {self.short_window}, long_window: {self.long_window}"
                )
            else:
                logging.info("MovingAverageCrossover is configured but not active (enabled: false).")
        else:
            logging.info("MovingAverageCrossover configuration section not found or is empty. MA Crossover will be inactive.")
            # self.active is already False, windows are defaults

    def max_frequencies(self):
        """
        Returns the maximum number of historical data points needed.
        """
        return self.long_window

    def _calculate_sma(self, data, window):
        """
        Calculates the Simple Moving Average (SMA).
        """
        if len(data) < window:
            return None
        return np.mean(data[-window:])

    # Signature changed to match AlgoIf.process
    def process(self, current_value, historical_values, currency, **kwargs): # Renamed 'values', added **kwargs
        """
        Process data to generate trading signals.

        Args:
            current_value (float): The current price of the asset.
            historical_values (list[float]): A list of historical closing prices.
                                   The list is expected to be ordered from oldest to newest.
            currency (str): The currency pair being traded (e.g., "BTC-USD").
            **kwargs: Additional keyword arguments (not used by this algorithm).

        Returns:
            int: 1 to buy, -1 to sell, 0 for no action.
        """
        if not self.active: # Added active check
            return 0

        logging.debug(f"Processing MovingAverageCrossover for {currency} with current value {current_value}")

        if len(historical_values) < self.long_window: # Renamed 'values'
            logging.info("Not enough data to calculate long moving average.")
            return 0  # Not enough data

        # Ensure historical_values are floats for numpy calculations
        try:
            numeric_values = [float(v) for v in historical_values] # Renamed 'values'
        except (ValueError, TypeError):
            try:
                numeric_values = [float(v.value) for v in historical_values] # Renamed 'values'
            except (TypeError, AttributeError, ValueError) as e:
                logging.error(f"Could not convert historical_values to float: {historical_values}. Error: {e}") # Renamed 'values'
                return 0


        # Short MA for current and previous period
        sma_short_current = self._calculate_sma(numeric_values, self.short_window)
        # For previous period, we exclude the most recent value and take the next `short_window` values
        sma_short_previous = self._calculate_sma(numeric_values[:-1], self.short_window)

        # Long MA for current and previous period
        sma_long_current = self._calculate_sma(numeric_values, self.long_window)
        # For previous period, we exclude the most recent value and take the next `long_window` values
        sma_long_previous = self._calculate_sma(numeric_values[:-1], self.long_window)

        if sma_short_current is None or sma_short_previous is None or            sma_long_current is None or sma_long_previous is None:
            logging.info("Not enough data for one or more moving averages after attempting calculation.")
            return 0 # Not enough data to make a decision

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
