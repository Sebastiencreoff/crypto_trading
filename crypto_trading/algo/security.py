import json
import logging

from . import model
from . import utils

class Security(object):
    """Class to thread hold to sell when the lost in a transaction is
    too high."""

    # Default values, can be overridden by config
    DEFAULT_MAXLOST_PERCENTAGE = 10 # Example default, adjust as needed
    DEFAULT_MAXLOST_PERCENTAGE_UPDATE = 1
    DEFAULT_TAKEPROFIT_PERCENTAGE = 20 # Example default

    def __init__(self, security_specific_config): # Parameter is now the specific config dictionary
        """Class Initialisation."""
        logging.debug(f"Initializing Security with config: {security_specific_config}")

        self.mean = None # This seems to be related to a rolling mean for a sell condition
        self.maxLost = None # This is an instance variable, maybe for state tracking, not the config value directly
        self.take_profit_percentage = None # Store the configuration value

        max_lost_config = security_specific_config.get('maxLost', {}) # Default to empty dict if 'maxLost' key is missing
        if max_lost_config:
            # These appear to set class-level attributes on the model.MaxLost class.
            # This is unusual and might be better if MaxLost instances used these values.
            # For now, replicating existing behavior but with specific config.
            model.MaxLost.__PERCENTAGE__ = max_lost_config.get('percentage', self.DEFAULT_MAXLOST_PERCENTAGE)
            model.MaxLost.__UPDATE__ = max_lost_config.get('percentage_update', self.DEFAULT_MAXLOST_PERCENTAGE_UPDATE)
            self.mean = max_lost_config.get('mean') # This is the 'frequency' for a rolling mean
            logging.info(
                f"Security: maxLost configured with percentage={model.MaxLost.__PERCENTAGE__}, "
                f"update_percentage={model.MaxLost.__UPDATE__}, mean_frequency={self.mean}"
            )
        else:
            logging.info("Security: 'maxLost' configuration not found or empty. Using defaults for model.MaxLost.")
            # Set defaults for model.MaxLost class attributes if no config
            model.MaxLost.__PERCENTAGE__ = self.DEFAULT_MAXLOST_PERCENTAGE
            model.MaxLost.__UPDATE__ = self.DEFAULT_MAXLOST_PERCENTAGE_UPDATE
            # self.mean remains None if not in config

        take_profit_config = security_specific_config.get('takeProfit', {}) # Default to empty dict
        if take_profit_config:
            self.take_profit_percentage = take_profit_config.get('percentage', self.DEFAULT_TAKEPROFIT_PERCENTAGE)
            logging.info(f"Security: takeProfit configured with percentage={self.take_profit_percentage}")
        else:
            logging.info("Security: 'takeProfit' configuration not found or empty. Using default for take_profit_percentage.")
            self.take_profit_percentage = self.DEFAULT_TAKEPROFIT_PERCENTAGE


        # model.create() is called if self.mean is set.
        # This implies self.mean (rolling mean frequency for a sell condition)
        # is a primary part of this class's setup.
        if self.mean:
            # This should ideally be called only once globally, not per instance.
            # However, replicating existing logic.
            model.create()
            logging.info(f"Security: model.create() called due to self.mean being set to {self.mean}")
        else:
            logging.info("Security: self.mean not configured under maxLost, model.create() not called by Security init.")

    # Renamed self.take_profit to self.take_profit_percentage to avoid confusion in the `sell` method.
    # The `sell` method will need to be updated to use `self.take_profit_percentage`.
    # This change is outside the scope of just __init__ but noted for consistency.
    # For now, let's only change __init__ and what it directly sets.
    # The original code set `self.take_profit`. I will stick to that for minimal diff in this step.
    # Reverting self.take_profit_percentage to self.take_profit for now.
        if take_profit_config:
            self.take_profit = take_profit_config.get('percentage', self.DEFAULT_TAKEPROFIT_PERCENTAGE) # Changed back
            logging.info(f"Security: takeProfit configured with percentage={self.take_profit}")
        else:
            logging.info("Security: 'takeProfit' configuration not found or empty. Using default for take_profit.")
            self.take_profit = self.DEFAULT_TAKEPROFIT_PERCENTAGE # Changed back

    def process(self, current_value, currency):
        logging.debug('')

        if self.mean:
            real_values = model.pricing.get_last_values(
                count=self.mean,
                currency=currency)

            model.rolling_mean_pricing.insert_value(
                currency=currency,
                frequency=self.mean,
                values=real_values)

    def sell(self, current_value, transaction):
        """Process data, it returned 1 to buy and -1 to sell."""
        logging.debug('')

        # Get last security
        if not self.maxLost or self.maxLost.transaction_id != transaction.id:
            result = model.MaxLost.select(
                model.MaxLost.q.transaction_id == transaction.id)
            self.maxLost = result[0] if result.count() else None
        if not self.maxLost:

            self.maxLost = model.MaxLost(
                buy_value=transaction.currency_buy_value,
                min_gain=100 - model.MaxLost.__PERCENTAGE__,
                min_value=(transaction.currency_buy_value
                           * (100 - model.MaxLost.__PERCENTAGE__)/100),
                transaction_id=transaction.id)

        if self.maxLost.process(current_value):
            return True

        percentage = current_value/transaction.currency_buy_value * 100
        if self.take_profit and percentage >= 100 + self.take_profit:
            logging.warning('Take Profit: value {}, gain:{}'.format(
                current_value, percentage))
            return True

        # Sell if value goes below this mean.
        if self.mean:
            avg = model.rolling_mean_pricing.get_last_values(
                transaction.currency,
                frequency=self.mean,
                count=1)[0]

            if avg and current_value <= avg:
                logging.error('SELL: Current Value: {} lower than mean {} '
                              'at frequency: {}'.format(
                                current_value, avg, self.mean))
                return True

        return False

    def buy(self, current_value, currency):

        # Buy if current_value is upper than mean and mean is increasing.
        if self.mean:
            values = model.rolling_mean_pricing.get_last_values(
                currency,
                frequency=self.mean,
                count=10)
            if all(x is not None for x in values):
                return utils.is_increasing(values)
        return False


