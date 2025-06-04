#!/usr/bin/env python

import datetime
import json
import logging

from . import model
from . import average # Required for isinstance check
from . import bollinger # Required for isinstance check
from . import moving_average_crossover # Required for isinstance check
from .ai_algo import AIAlgo


class AlgoMain:
    """Class which manage all algorithm to deal with data."""

    def __init__(self, config_filepath): # Changed variable name for clarity
        """Class Initialisation."""

        with open(config_filepath, mode='r') as f:
            self.config_data = json.load(f)

        self.algo_ifs = []

        # GuppyMMA
        guppy_config = self.config_data.get('GuppyMMA', {})
        if guppy_config.get('enabled', True): # Default to True if 'enabled' key missing but section exists
            self.algo_ifs.append(average.GuppyMMA(guppy_config))
            logging.info("GuppyMMA algorithm enabled and initialized.")
        elif 'enabled' in guppy_config: # Only log if 'enabled' is explicitly false
            logging.info("GuppyMMA algorithm is disabled by configuration (enabled: false).")
        # If section 'GuppyMMA' is missing, guppy_config is {}, .get('enabled', True) is True,
        # GuppyMMA will be instantiated (and likely run with defaults or be inactive based on its own logic).
        # This matches the behavior if 'enabled' key is missing.

        # Bollinger
        bollinger_config = self.config_data.get('Bollinger', {})
        if bollinger_config.get('enabled', True):
            self.algo_ifs.append(bollinger.Bollinger(bollinger_config))
            logging.info("Bollinger algorithm enabled and initialized.")
        elif 'enabled' in bollinger_config:
            logging.info("Bollinger algorithm is disabled by configuration (enabled: false).")

        # MovingAverageCrossover
        mac_config = self.config_data.get('MovingAverageCrossover', {})
        if mac_config.get('enabled', True):
            self.algo_ifs.append(moving_average_crossover.MovingAverageCrossover(mac_config))
            logging.info("MovingAverageCrossover algorithm enabled and initialized.")
        elif 'enabled' in mac_config:
            logging.info("MovingAverageCrossover algorithm is disabled by configuration (enabled: false).")

        # AIAlgo
        ai_algo_config = self.config_data.get('AIAlgo', {}) # Get the AIAlgo specific config
        if ai_algo_config.get('enabled', False): # Default to False for AIAlgo if 'enabled' key is missing
            self.algo_ifs.append(AIAlgo(ai_algo_config)) # Pass only the AIAlgo specific config
            logging.info("AIAlgo algorithm enabled and initialized.")
        elif 'enabled' in ai_algo_config: # Log if 'enabled' is present and false
             logging.info("AIAlgo algorithm is disabled by configuration (enabled: false).")
        # If AIAlgo section is missing, or 'enabled' is missing, it defaults to not loading.

        self.max_frequencies = 0
        if self.algo_ifs:
            try:
                valid_freqs = [x.max_frequencies() for x in self.algo_ifs if hasattr(x, 'max_frequencies') and x.max_frequencies() is not None]
                if valid_freqs:
                    self.max_frequencies = max(valid_freqs)
            except AttributeError as e:
                logging.warning(f"An algorithm without max_frequencies method might be present or max_frequencies returned None: {e}")
        model.create()

    def process(self, current_value, currency):
        """Process data, it returned 1 to buy and -1 to sell."""

        # Price data
        model.pricing.Pricing(currency=currency,
                              date_time=datetime.datetime.now(),
                              value=current_value)

        historical_values = [] # Renamed from 'values'
        if self.max_frequencies > 0:
            historical_values = model.pricing.get_last_values( # Renamed from 'values'
                count=self.max_frequencies,
                currency=currency)
        else:
            logging.warning("max_frequencies is 0, not fetching historical values.")

        total_result = 0
        indicator_signals_dict = {} # Renamed for clarity from indicator_signals

        # Process non-AI algorithms first and collect their signals
        for algo_instance in self.algo_ifs:
            if not isinstance(algo_instance, AIAlgo):
                # Pass historical_values, other args as per new signature (kwargs not used here)
                signal = algo_instance.process(current_value, historical_values, currency)
                indicator_signals_dict[algo_instance.__class__.__name__] = signal # Renamed
                total_result += signal

        # Process AIAlgo, passing in the collected signals
        for algo_instance in self.algo_ifs:
            if isinstance(algo_instance, AIAlgo):
                # Pass historical_values and indicator_signals_dict as a keyword argument
                ai_signal = algo_instance.process(current_value, historical_values, currency,
                                                  indicator_signals=indicator_signals_dict) # Pass as kwarg
                total_result += ai_signal

        logging.info('Total result after all algos: %d', total_result)
        return total_result

    def reset(self):
        model.reset()
