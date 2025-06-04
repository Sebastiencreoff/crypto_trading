#!/usr/bin/env python


class AlgoIf:
    """Super class to manage data."""

    def process(self, current_value, historical_values, currency, **kwargs):
        """
        Process data to generate a trading signal.

        Args:
            current_value (float): The current price of the asset.
            historical_values (list): A list of historical data points (e.g., prices, candles).
                                      The exact nature depends on the algorithm's needs.
            currency (str): The currency pair being traded (e.g., "BTC-USD").
            **kwargs: Additional keyword arguments that specific algorithms might use.
                      For example, AIAlgo might expect 'indicator_signals' here.

        Returns:
            int: 1 to buy, -1 to sell, 0 for no action.
        """
        raise NotImplementedError("Each algorithm must implement its own process method.")
