import json # Keep for now, might be removed if config_obj fully replaces direct dict use
import logging

from . import model # model.MaxLost, model.pricing, model.rolling_mean_pricing will need db_conn
from . import utils

class Security(object):
    """Class to threshold sell when the loss in a transaction is
    too high or take profit."""

    def __init__(self, config_obj): # Changed signature
        """Class Initialisation."""
        logging.debug('Security: Initializing with config_obj')

        # Use algo_config_dict from config_obj (assumption)
        algo_config_data = config_obj.algo_config_dict if hasattr(config_obj, 'algo_config_dict') else {}
        if not algo_config_data:
            logging.warning("Security initialized with empty or missing algo_config_dict from config_obj.")

        self.mean = None
        self.maxLost_instance_cache = None # Renamed self.maxLost to avoid confusion with model.MaxLost class
        self.take_profit_percentage = None # Renamed self.take_profit for clarity

        max_lost_config = algo_config_data.get('maxLost')
        if max_lost_config is not None:
            # These should ideally be instance variables or passed to MaxLost methods,
            # not class variables of model.MaxLost, if they can vary per Security instance or task.
            # For now, follow existing pattern but acknowledge it's problematic for concurrent tasks with different configs.
            model.MaxLost.__PERCENTAGE__ = max_lost_config.get('percentage')
            model.MaxLost.__UPDATE__ = max_lost_config.get('percentage_update', 1)
            self.mean = max_lost_config.get('mean') # Used for rolling mean check

        take_profit_config = algo_config_data.get('takeProfit')
        if take_profit_config is not None:
            self.take_profit_percentage = take_profit_config.get('percentage')

        # model.create() removed, handled by main application setup
        logging.info(f"Security initialized: mean={self.mean}, take_profit={self.take_profit_percentage}, MaxLost %={model.MaxLost.__PERCENTAGE__}")


    def process(self, db_conn, current_value, currency): # Added db_conn
        """Processes current value to update rolling mean if configured."""
        # This method's original purpose was to update rolling_mean_pricing.
        # model.rolling_mean_pricing.insert_value implies DB write and needs db_conn.
        # model.pricing.get_last_values implies DB read and needs db_conn.
        # These sub-models (pricing, rolling_mean_pricing) need refactoring.
        logging.debug(f'Security process for {currency}: current_value={current_value}')

        if self.mean and self.mean > 0 :
            try:
                real_values = model.pricing.get_last_values( # Needs db_conn
                    db_conn,
                    count=self.mean,
                    currency=currency)

                if real_values and len(real_values) == self.mean:
                     # model.rolling_mean_pricing.insert_value needs db_conn if it writes to DB.
                    model.rolling_mean_pricing.insert_value(
                        db_conn,
                        currency=currency,
                        frequency=self.mean,
                        values=real_values) # Assuming insert_value calculates mean and saves
                    logging.debug(f"Security: Updated rolling mean for {currency} with {self.mean} values.")
                else:
                    logging.debug(f"Security: Not enough values ({len(real_values) if real_values else 0}/{self.mean}) to update rolling mean for {currency}.")
            except Exception as e:
                logging.error(f"Security: Error processing rolling mean for {currency}: {e}", exc_info=True)
        else:
            logging.debug(f"Security: Rolling mean calculation not configured (mean={self.mean}).")


    def sell(self, db_conn, current_value, transaction, currency): # Added db_conn and currency
        """Determines if a sell is advised based on security rules."""
        logging.debug(f'Security sell check for {currency}: current_value={current_value}, transaction_id={transaction.id}')

        # MaxLost logic
        if model.MaxLost.__PERCENTAGE__ is not None: # Check if MaxLost is configured
            # Get last security instance for this transaction
            if not self.maxLost_instance_cache or self.maxLost_instance_cache.transaction_id != transaction.id:
                result = model.MaxLost.select( # Needs connection
                    model.MaxLost.q.transaction_id == transaction.id,
                    connection=db_conn
                )
                self.maxLost_instance_cache = result[0] if result.count() else None

            if not self.maxLost_instance_cache:
                # MaxLost is a SQLObject, needs connection.
                self.maxLost_instance_cache = model.MaxLost(
                    buy_value=transaction.currency_buy_value, # Price per unit of crypto
                    min_gain=100 - model.MaxLost.__PERCENTAGE__,
                    min_value=(transaction.currency_buy_value
                               * (100 - model.MaxLost.__PERCENTAGE__)/100),
                    transaction_id=transaction.id,
                    connection=db_conn) # Pass connection
                logging.info(f"Security: Created new MaxLost instance for tx {transaction.id} for {currency}, min_value={self.maxLost_instance_cache.min_value:.2f}")

            # maxLost_instance_cache.process needs db_conn if it modifies the DB record (e.g. updates itself)
            if self.maxLost_instance_cache.process(db_conn, current_value): # Pass db_conn
                logging.warning(f"Security: SELL signal from MaxLost for {currency}. Current: {current_value}, Min allowed: {self.maxLost_instance_cache.min_value:.2f}")
                return True

        # Take Profit logic
        if self.take_profit_percentage and transaction.currency_buy_value > 0: # Avoid division by zero
            percentage_gain = (current_value / transaction.currency_buy_value * 100) - 100
            if percentage_gain >= self.take_profit_percentage:
                logging.warning(f'Security: SELL signal from TakeProfit for {currency}. Current: {current_value}, Gain: {percentage_gain:.2f}%, Target: {self.take_profit_percentage:.2f}%')
                return True

        # Rolling Mean based sell logic
        if self.mean and self.mean > 0:
            try:
                # model.rolling_mean_pricing.get_last_values needs db_conn
                avg_values = model.rolling_mean_pricing.get_last_values(
                    db_conn, # Pass db_conn
                    currency, # currency was transaction.currency
                    frequency=self.mean,
                    count=1)

                if avg_values and avg_values[0] is not None:
                    avg = avg_values[0]
                    if current_value <= avg:
                        logging.warning(f'Security: SELL signal from RollingMean for {currency}. Current: {current_value:.2f} <= Mean: {avg:.2f} (freq {self.mean})')
                        return True
                else:
                    logging.debug(f"Security: Could not get rolling mean for {currency} (freq {self.mean}) to evaluate sell decision.")
            except Exception as e:
                logging.error(f"Security: Error checking rolling mean for {currency} sell decision: {e}", exc_info=True)

        logging.debug(f"Security: No sell signal for {currency} at {current_value}.")
        return False

    def buy(self, db_conn, current_value, currency): # Added db_conn
        """Determines if a buy is advised based on security rules (e.g., increasing rolling mean)."""
        logging.debug(f'Security buy check for {currency}: current_value={current_value}')

        # Rolling Mean based buy logic
        if self.mean and self.mean > 0:
            try:
                # model.rolling_mean_pricing.get_last_values needs db_conn
                values = model.rolling_mean_pricing.get_last_values(
                    db_conn, # Pass db_conn
                    currency,
                    frequency=self.mean,
                    count=10) # Fetch more values to check if mean is increasing

                if values and all(x is not None for x in values) and len(values) >= 2: # Need at least 2 points to check trend
                    # Check if current value is above the most recent mean and if the mean itself is increasing
                    # This is a simplified check. A more robust check might involve utils.is_increasing on the mean values.
                    # For now, let's assume 'values' are recent mean values.
                    # The original logic was: return utils.is_increasing(values)
                    # This implies 'values' are means, not raw prices.
                    # This requires rolling_mean_pricing.get_last_values to return mean values.
                    if current_value > values[-1] and utils.is_increasing(values): # Assuming values are means
                        logging.info(f"Security: BUY signal from RollingMean for {currency}. Current: {current_value:.2f} > Mean: {values[-1]:.2f}, and mean is increasing.")
                        return True
                else:
                    logging.debug(f"Security: Not enough data or valid mean values for {currency} (freq {self.mean}) to evaluate buy decision.")
            except Exception as e:
                logging.error(f"Security: Error checking rolling mean for {currency} buy decision: {e}", exc_info=True)

        logging.debug(f"Security: No buy signal for {currency} at {current_value}.")
        return False


