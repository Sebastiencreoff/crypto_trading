import logging
import numpy as np
from .algoIf import AlgoIf

class MovingAverageCrossover(AlgoIf):
    """
    Trading algorithm based on moving average crossovers.
    """

    def __init__(self, config_dict):
        """
        Initializes the MovingAverageCrossover algorithm.

        Args:
            config_dict (dict): Python dictionary containing the configuration.
        """
        logging.debug("Initializing MovingAverageCrossover")
        try:
            algo_config = config_dict.get("MovingAverageCrossover", {})
            self.short_window = algo_config.get("short_window", 20)
            self.long_window = algo_config.get("long_window", 50)

            if self.short_window >= self.long_window:
                logging.error("Short window must be less than long window for MovingAverageCrossover.")
                # Potentially raise an error or handle this case appropriately
                # For now, defaulting to safe values if config is problematic
                self.short_window = 20
                self.long_window = 50

        except Exception as e:
            logging.error(f"An unexpected error occurred during MovingAverageCrossover initialization: {e}")
            self.short_window = 20
            self.long_window = 50

        logging.info(
            f"MovingAverageCrossover initialized with short_window: {self.short_window}, long_window: {self.long_window}"
        )

    def update_config(self, mac_params_dict):
        """Updates the MovingAverageCrossover configuration."""
        logging.debug(f"Updating MovingAverageCrossover configuration with: {mac_params_dict}")
        new_short_window = mac_params_dict.get('short_window', self.short_window)
        new_long_window = mac_params_dict.get('long_window', self.long_window)

        if new_short_window < new_long_window:
            self.short_window = new_short_window
            self.long_window = new_long_window
            logging.info(f"MovingAverageCrossover configuration updated: short_window={self.short_window}, long_window={self.long_window}")
        else:
            logging.warning(f"Invalid window sizes in update_config for MovingAverageCrossover: short={new_short_window}, long={new_long_window}. Keeping existing values: short_window={self.short_window}, long_window={self.long_window}")

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

    def process(self, current_value, values, currency):
        """
        Process data to generate trading signals.

        Args:
            current_value (float): The current price of the asset.
            values (list[float]): A list of historical closing prices.
                                   The list is expected to be ordered from oldest to newest.
            currency (str): The currency pair being traded (e.g., "BTC-USD").

        Returns:
            int: 1 to buy, -1 to sell, 0 for no action.
        """
        logging.debug(f"Processing MovingAverageCrossover for {currency} with current value {current_value}")

        if len(values) < self.long_window:
            logging.info("Not enough data to calculate long moving average.")
            return 0  # Not enough data

        # Ensure values are floats for numpy calculations
        try:
            numeric_values = [float(v) for v in values]
        except (ValueError, TypeError): # Catch TypeError here as well
            try: # If values are Pydantic objects from model.pricing.Pricing.value
                numeric_values = [float(v.value) for v in values]
            except (TypeError, AttributeError, ValueError) as e:
                logging.error(f"Could not convert values to float: {values}. Error: {e}")
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
