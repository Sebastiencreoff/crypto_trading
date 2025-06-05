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

    def __init__(self, config_dict):
        """Class Initialisation."""

        self.__dict__ = json.load(open(config_dict, mode='r'))
        self.algo_ifs = []
        ai_algo_instance = None
        ai_target_configs = {}

        if self.__dict__.get("AIAlgo", {}).get("enabled", False):
            # Pass the main config dict to AIAlgo constructor
            ai_algo_instance = AIAlgo(self.__dict__)
            ai_target_configs = ai_algo_instance.get_target_algo_configs()
            self.algo_ifs.append(ai_algo_instance)

        # Instantiate GuppyMMA
        guppy_config = self.__dict__.copy()
        if ai_algo_instance and 'GuppyMMA' in ai_target_configs:
            # Update the 'GuppyMMA' key in our copy of the main config
            guppy_config['GuppyMMA'] = ai_target_configs['GuppyMMA']
        self.algo_ifs.append(average.GuppyMMA(guppy_config))

        # Instantiate Bollinger
        bollinger_config = self.__dict__.copy()
        if ai_algo_instance and 'Bollinger' in ai_target_configs:
            # Update the 'Bollinger' key in our copy of the main config
            bollinger_config['Bollinger'] = ai_target_configs['Bollinger']
        self.algo_ifs.append(bollinger.Bollinger(bollinger_config))

        # Instantiate MovingAverageCrossover
        mac_config = self.__dict__.copy()
        if ai_algo_instance and 'MovingAverageCrossover' in ai_target_configs:
            # Update the 'MovingAverageCrossover' key in our copy of the main config
            mac_config['MovingAverageCrossover'] = ai_target_configs['MovingAverageCrossover']
        self.algo_ifs.append(moving_average_crossover.MovingAverageCrossover(mac_config))

        self.max_frequencies = 0
        if self.algo_ifs:
            try:
                valid_freqs = [x.max_frequencies() for x in self.algo_ifs if hasattr(x, 'max_frequencies') and x.max_frequencies() is not None]
                if valid_freqs:
                    self.max_frequencies = max(valid_freqs)
            except AttributeError as e: # Should be more specific or check hasattr more carefully
                logging.warning(f"An algorithm without max_frequencies method might be present or max_frequencies returned None: {e}")
        model.create()

    def process(self, current_value, currency):
        """Process data, it returned 1 to buy and -1 to sell."""

        # Price data
        model.pricing.Pricing(currency=currency,
                              date_time=datetime.datetime.now(),
                              value=current_value)

        values = []
        if self.max_frequencies > 0:
            values = model.pricing.get_last_values(
                count=self.max_frequencies,
                currency=currency)
        else:
            logging.warning("max_frequencies is 0, not fetching historical values.")

        total_result = 0
        indicator_signals = {} # To collect signals from non-AI algos for AIAlgo
        new_algo_configs = {}  # To store configs from AIAlgo if it runs

        # First pass: Process non-AI algorithms and collect their signals
        # These signals can be used by AIAlgo
        for algo_instance in self.algo_ifs:
            if not isinstance(algo_instance, AIAlgo):
                try:
                    signal = algo_instance.process(current_value, values, currency)
                    indicator_signals[algo_instance.__class__.__name__] = signal
                    total_result += signal
                    logging.debug(f"Signal from {algo_instance.__class__.__name__}: {signal}. Current total: {total_result}")
                except Exception as e:
                    logging.error(f"Error processing {algo_instance.__class__.__name__}: {e}", exc_info=True)


        # Second pass: Process AIAlgo if it exists
        # AIAlgo uses the indicator_signals from other algorithms
        ai_algo_processed_successfully = False
        for algo_instance in self.algo_ifs:
            if isinstance(algo_instance, AIAlgo):
                try:
                    # AIAlgo's process method now returns its signal and a dictionary of new configurations
                    ai_signal, received_configs = algo_instance.process(current_value, values, currency, indicator_signals)
                    total_result += ai_signal
                    new_algo_configs = received_configs # Store the new configs from AIAlgo
                    ai_algo_processed_successfully = True # Mark that AIAlgo has been processed
                    logging.debug(f"Signal from AIAlgo: {ai_signal}. Current total: {total_result}")
                    if new_algo_configs:
                        logging.info(f"AIAlgo produced new configurations: {new_algo_configs}")
                    else:
                        logging.debug("AIAlgo did not produce new configurations this cycle.")
                    break # Assuming only one AIAlgo instance needs to be processed
                except Exception as e:
                    logging.error(f"Error processing AIAlgo: {e}", exc_info=True)
                    # In case of an error in AIAlgo, new_algo_configs remains empty or its last state

        # Third pass: If AIAlgo was processed successfully and returned new configs, update other algos
        if ai_algo_processed_successfully and new_algo_configs:
            for algo_to_update in self.algo_ifs:
                # Do not try to update AIAlgo with its own generated configs in this loop
                if not isinstance(algo_to_update, AIAlgo):
                    algo_name = algo_to_update.__class__.__name__
                    if algo_name in new_algo_configs:
                        try:
                            logging.info(f"AlgoMain: Attempting to update {algo_name} with new config from AIAlgo: {new_algo_configs[algo_name]}")
                            algo_to_update.update_config(new_algo_configs[algo_name])
                            # Individual update_config methods should have their own success/failure logging
                        except Exception as e:
                            logging.error(f"AlgoMain: Error updating {algo_name} with new config: {e}", exc_info=True)
                    else:
                        logging.debug(f"AlgoMain: No new config from AIAlgo for {algo_name} in this cycle.")
        elif ai_algo_processed_successfully: # AIAlgo ran but new_algo_configs is empty
            logging.debug("AlgoMain: AIAlgo ran but provided no new configurations to apply.")


        logging.info('Total result after all algos: %d', total_result)
        return total_result

    def reset(self):
        model.reset()
