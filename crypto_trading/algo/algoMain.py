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

    def __init__(self, config_obj): # Changed signature to accept config_obj
        """Class Initialisation."""

        # Use algo_config_dict from config_obj (assumption)
        # If algo_config_dict is nested, adjust accordingly e.g., config_obj.algo_specific_settings.algo_config_dict
        algo_config_data = config_obj.algo_config_dict if hasattr(config_obj, 'algo_config_dict') else {}
        if not algo_config_data:
            logging.warning("AlgoMain initialized with empty or missing algo_config_dict from config_obj.")

        self.algo_ifs = []
        ai_algo_instance = None
        ai_target_configs = {}

        # Use algo_config_data for configuration
        if algo_config_data.get("AIAlgo", {}).get("enabled", False):
            ai_algo_instance = AIAlgo(config_obj) # Pass the main config_obj or specific part
            ai_target_configs = ai_algo_instance.get_target_algo_configs()
            self.algo_ifs.append(ai_algo_instance)

        # Instantiate GuppyMMA
        # Each sub-algo should also be updated to accept config_obj or relevant part
        self.algo_ifs.append(average.GuppyMMA(config_obj)) # Pass config_obj

        # Instantiate Bollinger
        self.algo_ifs.append(bollinger.Bollinger(config_obj)) # Pass config_obj

        # Instantiate MovingAverageCrossover
        self.algo_ifs.append(moving_average_crossover.MovingAverageCrossover(config_obj)) # Pass config_obj

        self.max_frequencies = 0
        if self.algo_ifs:
            try:
                valid_freqs = [x.max_frequencies() for x in self.algo_ifs if hasattr(x, 'max_frequencies') and x.max_frequencies() is not None]
                if valid_freqs:
                    self.max_frequencies = max(valid_freqs)
            except AttributeError as e: # Should be less likely if sub-algos conform
                logging.warning(f"Error accessing max_frequencies from a sub-algorithm: {e}")
        # model.create() removed, should be handled by main application setup

    def process(self, db_conn, current_value, currency): # Added db_conn
        """Process data, it returned 1 to buy and -1 to sell."""

        # Price data - model.pricing.Pricing is likely a SQLObject.
        # If so, it needs connection=db_conn. This implies model.pricing.Pricing needs refactoring.
        # For now, assuming model.save_price is the correct way to persist price ticks as per model.py refactor.
        # model.pricing.Pricing(currency=currency, # This line is problematic if Pricing is SQLObject based
        #                       date_time=datetime.datetime.now(),
        #                       value=current_value,
        #                       connection=db_conn) # Assuming Pricing is SQLObject and needs connection
        # The above line is removed as model.save_price in trading.py already handles this.
        # This process method is for calculating signals, not saving the current price tick again.

        values = []
        if self.max_frequencies > 0:
            # model.pricing.get_last_values needs db_conn if it queries DB.
            # This implies model.pricing needs refactoring.
            values = model.pricing.get_last_values(
                db_conn, # Pass db_conn
                count=self.max_frequencies,
                currency=currency)
        else:
            logging.warning("max_frequencies is 0, not fetching historical values for algo processing.")

        total_result = 0
        indicator_signals = {}

        # Process non-AI algorithms first and collect their signals
        for algo_instance in self.algo_ifs:
            if not isinstance(algo_instance, AIAlgo):
                # Sub-algo process methods also need db_conn if they access DB
                signal = algo_instance.process(db_conn, current_value, values, currency)
                indicator_signals[algo_instance.__class__.__name__] = signal
                total_result += signal

        # Process AIAlgo, passing in the collected signals
        for algo_instance in self.algo_ifs:
            if isinstance(algo_instance, AIAlgo):
                # AIAlgo's process method will need to be updated to accept db_conn and indicator_signals dict.
                ai_signal = algo_instance.process(db_conn, current_value, values, currency, indicator_signals)
                total_result += ai_signal

        logging.info('AlgoMain: Total result for %s after all algos: %d', currency, total_result)
        return total_result

    def reset(self, db_conn, currency): # Added db_conn and currency
        logging.info(f"AlgoMain: Resetting model data for currency {currency}.")
        model.reset(db_conn, currency) # Pass db_conn and currency
        # Optionally, reset internal states of sub-algorithms if they have reset methods
        for algo_instance in self.algo_ifs:
            if hasattr(algo_instance, 'reset'):
                # Sub-algo reset methods might also need db_conn and currency
                try:
                    algo_instance.reset(db_conn, currency)
                except TypeError: # Handle if their reset doesn't take args yet
                    logging.warning(f"Could not call reset on {algo_instance.__class__.__name__} with db_conn/currency. Attempting without.")
                    try:
                        algo_instance.reset()
                    except Exception as e:
                         logging.error(f"Failed to call reset on {algo_instance.__class__.__name__}: {e}")

        logging.info(f"AlgoMain: Reset complete for currency {currency}.")
