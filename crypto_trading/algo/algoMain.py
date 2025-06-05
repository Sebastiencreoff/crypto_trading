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
            except AttributeError as e:
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
        indicator_signals = {}

        # Process non-AI algorithms first and collect their signals
        for algo_instance in self.algo_ifs:
            if not isinstance(algo_instance, AIAlgo):
                signal = algo_instance.process(current_value, values, currency)
                indicator_signals[algo_instance.__class__.__name__] = signal
                total_result += signal

        # Process AIAlgo, passing in the collected signals
        for algo_instance in self.algo_ifs:
            if isinstance(algo_instance, AIAlgo):
                # AIAlgo's process method will need to be updated to accept indicator_signals dict.
                ai_signal = algo_instance.process(current_value, values, currency, indicator_signals)
                total_result += ai_signal

        logging.info('Total result after all algos: %d', total_result)
        return total_result

    def reset(self):
        model.reset()
