#!/usr/bin/env python

import datetime
import json
import logging

from . import model
from . import average
from . import bollinger
from . import moving_average_crossover
from crypto_trading.utils.volatility import calculate_log_return_volatility
# Make sure AlgoIf is available if needed for type hinting, though not directly used here.
# from .algoIf import AlgoIf


class AlgoMain:
    """Class which manage all algorithm to deal with data."""

    def __init__(self, config_path: str): # Changed param name to config_path
        """Class Initialisation."""
        logging.info(f"Initializing AlgoMain with config path: {config_path}")

        self.full_config_data = {}
        try:
            with open(config_path, mode='r') as f:
                self.full_config_data = json.load(f)
        except FileNotFoundError:
            logging.error(f"AlgoMain: Configuration file not found: {config_path}. Algorithms may use defaults or fail.")
            # self.full_config_data remains {}
        except json.JSONDecodeError:
            logging.error(f"AlgoMain: Error decoding JSON from {config_path}. Algorithms may use defaults or fail.")
            # self.full_config_data remains {}
        except Exception as e:
            logging.error(f"AlgoMain: Unexpected error loading config {config_path}: {e}")
            # self.full_config_data remains {}

        self.algo_ifs = []
        # Pass the specific dictionary section to each algorithm.
        # Use .get(key, {}) to provide an empty dict if the key is missing,
        # allowing algos to initialize with their defaults.

        guppy_config = self.full_config_data.get("GuppyMMA", {})
        logging.info(f"Instantiating GuppyMMA with config: {guppy_config if guppy_config else 'Defaults'}")
        self.algo_ifs.append(average.GuppyMMA(guppy_config))

        bollinger_config = self.full_config_data.get("Bollinger", {})
        logging.info(f"Instantiating Bollinger with config: {bollinger_config if bollinger_config else 'Defaults'}")
        self.algo_ifs.append(bollinger.Bollinger(bollinger_config))

        mac_config = self.full_config_data.get("MovingAverageCrossover", {})
        logging.info(f"Instantiating MovingAverageCrossover with config: {mac_config if mac_config else 'Defaults'}")
        self.algo_ifs.append(moving_average_crossover.MovingAverageCrossover(mac_config))

        self.max_frequencies = 0 # Initialize before calculation
        self.recalculate_max_frequencies() # Calculate based on initialized algos

        # model.create() # This was in the original, assuming it's still needed.
                       # It might be better placed elsewhere if it's a global DB setup.
                       # For now, keeping it to match original functionality as closely as possible.
        try:
            model.create()
            logging.info("AlgoMain: model.create() called successfully.")
        except Exception as e:
            logging.error(f"AlgoMain: Error calling model.create(): {e}")


    def recalculate_max_frequencies(self):
        """Recalculates the overall max_frequencies needed by any algorithm."""
        self.max_frequencies = 0 # Default if no algos or none have frequencies
        if self.algo_ifs:
            valid_freqs = []
            for x in self.algo_ifs:
                try:
                    freq = x.max_frequencies()
                    if freq is not None: # Ensure freq is not None before adding
                        valid_freqs.append(freq)
                except Exception as e: # Catch potential errors from individual max_frequencies() calls
                    logging.error(f"Error getting max_frequencies from {x.__class__.__name__}: {e}")

            if valid_freqs:
                self.max_frequencies = max(valid_freqs)
        logging.info(f"AlgoMain recalculated max_frequencies to: {self.max_frequencies}")

    def update_all_algorithm_configs(self, new_full_config: dict):
        """
        Updates configurations for all managed algorithms and recalculates max_frequencies.
        Args:
            new_full_config (dict): The new full configuration data, similar to initial algo.json.
        """
        logging.info("AlgoMain: Updating all algorithm configurations dynamically.")
        self.full_config_data = new_full_config # Store the new full config

        for algo_instance in self.algo_ifs:
            algo_name = algo_instance.__class__.__name__
            # Determine the config key based on the algorithm's class name
            config_key = ""
            if isinstance(algo_instance, average.GuppyMMA): # More robust check
                config_key = "GuppyMMA"
            elif isinstance(algo_instance, bollinger.Bollinger):
                config_key = "Bollinger"
            elif isinstance(algo_instance, moving_average_crossover.MovingAverageCrossover):
                config_key = "MovingAverageCrossover"
            # Add other mappings if more algorithms exist:
            # elif isinstance(algo_instance, some_other_algo.SomeOtherAlgoClass):
            #     config_key = "SomeOtherAlgoKey"
            else:
                logging.warning(f"AlgoMain: No config mapping for algorithm class: {algo_name}")
                continue

            if config_key in new_full_config:
                logging.info(f"Updating config for {algo_name} with section for {config_key}.")
                algo_instance.update_config(new_full_config[config_key])
            else:
                logging.warning(f"AlgoMain: No configuration found for {config_key} in new_full_config. {algo_name} may use defaults or retain old config.")
                # Optionally, call update_config with an empty dict to reset to defaults:
                # algo_instance.update_config({})

        self.recalculate_max_frequencies() # Recalculate max frequencies as they might have changed
        logging.info("AlgoMain: Finished updating all algorithm configurations.")

    def process(self, current_value, currency):
        """Process data, it returned 1 to buy and -1 to sell."""
        logging.debug(f"AlgoMain processing for {currency}, current_value: {current_value}, max_freq: {self.max_frequencies}")

        # Price data
        try:
            model.pricing.Pricing(currency=currency,
                                  date_time=datetime.datetime.now(),
                                  value=current_value)
        except Exception as e:
            logging.error(f"AlgoMain: Error saving pricing data for {currency}: {e}")
            return 0 # Cannot proceed if pricing data cannot be saved

        if self.max_frequencies <= 0:
            logging.info(f"AlgoMain: max_frequencies is {self.max_frequencies}. Skipping value retrieval and processing for {currency}.")
            return 0

        values = []
        try:
            values = model.pricing.get_last_values(
                count=self.max_frequencies,
                currency=currency)
        except Exception as e:
            logging.error(f"AlgoMain: Error retrieving last values for {currency}: {e}")
            return 0 # Cannot proceed if historical values cannot be retrieved

        # Calculate and log volatility
        if values: # Ensure 'values' is not empty or None
            # The 'values' from model.pricing.get_last_values are Pydantic objects
            # Extract the numeric price from each object
            try:
                # Assuming 'values' is a list of objects with a 'value' attribute
                numeric_prices = [item.value for item in values]
            except AttributeError:
                # If 'values' is already a list of numbers (e.g. floats/ints)
                numeric_prices = values
            except Exception as e:
                logging.error(f"AlgoMain: Could not extract numeric prices from 'values' for volatility calculation. Error: {e}")
                numeric_prices = []

            if numeric_prices:
                # Define a window for volatility, e.g., 20 periods.
                # This could eventually be made configurable.
                volatility_window = 20
                current_volatility = calculate_log_return_volatility(numeric_prices, window=volatility_window)
                if current_volatility is not None:
                    logging.info(f"AlgoMain: Calculated volatility for {currency} (window {volatility_window}): {current_volatility:.6f}")
                else:
                    logging.info(f"AlgoMain: Volatility calculation for {currency} returned None (e.g., insufficient data).")
            else:
                logging.info(f"AlgoMain: No numeric prices available to calculate volatility for {currency}.")
        else:
            logging.info(f"AlgoMain: No historical values found to calculate volatility for {currency}.")


        result = 0
        for algo in self.algo_ifs:
            try:
                # Pass only the necessary part of values if algos are sensitive to too much data
                # For now, passing all retrieved values up to max_frequencies
                result += algo.process(current_value, values, currency)
            except Exception as e:
                logging.error(f"AlgoMain: Error during processing by {algo.__class__.__name__} for {currency}: {e}")
                # Decide if one algo error should stop others or just be skipped

        # Ensure result is within expected bounds if necessary, e.g. if multiple algos vote
        # For now, direct sum as per original.
        # result = max(-1, min(1, result)) # Example: clamp result to -1, 0, 1

        logging.info(f'AlgoMain result for {currency}: {result}')
        return result

    def reset(self):
        """Resets the underlying data model."""
        logging.info("AlgoMain: Resetting model.")
        try:
            model.reset()
            logging.info("AlgoMain: Model reset successfully.")
        except Exception as e:
            logging.error(f"AlgoMain: Error during model reset: {e}")
