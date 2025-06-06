#!/usr/bin/env python

import logging
import time
import multiprocessing
import datetime # Added for datetime.datetime.utcnow()

# SQLAlchemy related imports
from crypto_trading.database import core_operations as db_core_ops
from crypto_trading.database import models as db_models
# from crypto_trading.database import algo_operations as db_algo_ops # Not directly used yet, but good to have if algos need it

from . import algo
from . import connection
# Removed: from .connection import coinBase (No longer exists)
from .connection import binance

# Removed: from . import model (SQLObject model)


class Trading:
    """Trading process, adapted to run as a managed task using SQLAlchemy."""

    def __init__(self, config_obj, task_id, stop_event, results_queue, session): # Added session
        """Initialisation of the trading task."""
        self.conf = config_obj
        self.task_id = task_id
        self.stop_event = stop_event
        self.results_queue = results_queue
        self.session = session # SQLAlchemy session

        # Removed: self.db_conn = self.conf.db_conn (SQLObject connection)

        self.logger = logging.getLogger(f"TradingTask-{self.task_id}")
        self.logger.info(f"Initializing trading task {self.task_id} for currency {self.conf.currency}")

        self.connect = None # Exchange connection, not DB connection
        # Removed Coinbase connection handling
        if self.conf.connection_type == 'simulation':
            self.connect = connection.simulation.SimulationConnect(
                self.conf.connection_config_path, self.conf.dir_path)
        elif self.conf.connection_type == 'binance':
            self.connect = connection.binance.BinanceConnect(
                self.conf.connection_config_path)
        else:
            error_msg = f"Task {self.task_id}: Unknown connection type: {self.conf.connection_type}"
            self.logger.error(error_msg)
            self.results_queue.put({"status": "error", "message": error_msg})
            raise ValueError(error_msg)

        self.algo_if = algo.AlgoMain(self.conf) # Already takes full config_obj
        self.security = algo.Security(self.conf) # Already takes full config_obj

        self.logger.info(f"Task {self.task_id}: Initialization complete.")

    # Removed _get_algo as AlgoMain is directly instantiated with self.conf

    def run(self):
        """Launch the trading process loop, checking for stop_event."""
        self.logger.info(f"Task {self.task_id}: Starting Trading run loop for {self.conf.currency}.")
        self.results_queue.put({"status": "info", "message": f"Task {self.task_id}: Started for {self.conf.currency}"})

        prev_currency_value = None
        # Fetch ongoing transaction using SQLAlchemy session and new db operation
        current_transaction = db_core_ops.get_open_transaction(
            self.session, currency_pair=self.conf.currency, task_id=str(self.task_id)
        )
        self.logger.info(f"Task {self.task_id}: Initial transaction status: {'Ongoing ID: ' + str(current_transaction.id) if current_transaction else 'None'}")

        try:
            while not self.stop_event.is_set():
                try:
                    currency_value = self.connect.get_value(self.conf.currency)
                    if currency_value is None and self.conf.connection_type == 'simulation':
                        self.logger.info(f"Task {self.task_id}: Simulation data source exhausted for {self.conf.currency}.")
                        self.results_queue.put({"status": "info", "message": f"Task {self.task_id}: Simulation ended for {self.conf.currency}."})
                        break
                except connection.EndOfProcess:
                    self.logger.info(f"Task {self.task_id}: EndOfProcess signal received for {self.conf.currency}.")
                    self.results_queue.put({"status": "info", "message": f"Task {self.task_id}: EndOfProcess for {self.conf.currency}."})
                    break
                except Exception as e:
                    self.logger.error(f"Task {self.task_id}: Error getting currency value: {e}", exc_info=True)
                    self.results_queue.put({"status": "error", "message": f"Task {self.task_id}: Error getting value: {e}"})
                    # No session.rollback() here as no DB changes made yet in this block
                    time.sleep(self.conf.delay_secs if hasattr(self.conf, 'delay_secs') else 5) # Use delay_secs
                    continue

                if prev_currency_value != currency_value:
                    self.logger.debug(f"Task {self.task_id}: Currency Value for {self.conf.currency}: {currency_value}")
                    prev_currency_value = currency_value

                    try:
                        # Save price tick using SQLAlchemy session
                        # save_price_tick commits internally, consider if batching is needed later.
                        db_core_ops.save_price_tick(
                            self.session,
                            currency_pair=self.conf.currency,
                            price=currency_value,
                            timestamp=datetime.datetime.utcnow()
                        )
                    except Exception as e:
                        self.logger.error(f"Task {self.task_id}: Failed to save price {currency_value} for {self.conf.currency}: {e}", exc_info=True)
                        # This is not a trade-critical error, so we might not rollback or continue, but log it.

                    # Process trading logic using SQLAlchemy session
                    # Algo methods now expect session as first arg
                    algo_signal = self.algo_if.process(self.session, currency_value, currency=self.conf.currency)
                    self.security.process(self.session, currency_value, currency=self.conf.currency) # Security process might update its own state/models

                    if current_transaction and current_transaction.buy_date_time: # Already bought, considering selling
                        should_sell_security = self.security.sell(
                            self.session,
                            current_value=currency_value,
                            transaction=current_transaction, # Pass SQLAlchemy model instance
                            currency=self.conf.currency
                        )
                        if algo_signal < 0 or should_sell_security:
                            try:
                                sell_price, _ = self.connect.sell( # Assuming sell returns (price_per_unit, amount_crypto_sold)
                                    current_transaction.base_currency_bought_amount,
                                    current_transaction.currency_pair, # Use currency_pair from transaction
                                    currency_value # Current market price for selling
                                )

                                # Update existing SQLAlchemy transaction object
                                current_transaction.sell_date_time = datetime.datetime.utcnow()
                                current_transaction.base_currency_sell_price = sell_price
                                current_transaction.sell_fee_eur = 0.0 # Placeholder for fee

                                # Calculate profit
                                profit = ( (sell_price * current_transaction.base_currency_bought_amount) -
                                           current_transaction.buy_value_eur -
                                           (current_transaction.buy_fee_eur if current_transaction.buy_fee_eur else 0.0) -
                                           (current_transaction.sell_fee_eur if current_transaction.sell_fee_eur else 0.0) )
                                current_transaction.profit_eur = profit

                                self.session.commit() # Commit sell transaction
                                self.logger.info(f"Task {self.task_id}: Sold {current_transaction.base_currency_bought_amount} of {current_transaction.currency_pair} at {sell_price}. Tx ID: {current_transaction.id}. Profit: {profit:.2f}")
                                self.results_queue.put({"status": "sold", "message": f"Task {self.task_id}: Sold {current_transaction.currency_pair} at {sell_price}. Profit: {profit:.2f}"})
                                current_transaction = None # Reset for next potential buy
                            except Exception as e:
                                self.session.rollback()
                                self.logger.error(f"Task {self.task_id}: Error during sell operation: {e}", exc_info=True)
                                continue

                    elif algo_signal > 0 and not current_transaction: # No open transaction, considering buying
                         should_buy_security = self.security.buy(
                             self.session,
                             current_value=currency_value,
                             currency=self.conf.currency
                         )
                         if should_buy_security:
                            try:
                                # connect.buy returns (price_per_crypto, amount_of_crypto_bought)
                                price_per_crypto, amount_of_crypto_bought = self.connect.buy(
                                    self.conf.transaction_amt, # This is buy_value_eur (total to spend)
                                    self.conf.currency,
                                    currency_value # Current market price for buying
                                )

                                # Create new SQLAlchemy transaction object
                                current_transaction = db_models.TradingTransaction(
                                    task_id=str(self.task_id), # Ensure task_id is string
                                    currency_pair=self.conf.currency,
                                    buy_value_eur=self.conf.transaction_amt,
                                    buy_fee_eur=0.0, # Placeholder for fee
                                    base_currency_bought_amount=amount_of_crypto_bought,
                                    base_currency_buy_price=price_per_crypto,
                                    buy_date_time=datetime.datetime.utcnow()
                                )
                                self.session.add(current_transaction)
                                self.session.commit() # Commit buy transaction
                                self.logger.info(f"Task {self.task_id}: Bought {amount_of_crypto_bought} of {self.conf.currency} at {price_per_crypto}. Tx ID: {current_transaction.id}")
                                self.results_queue.put({"status": "bought", "message": f"Task {self.task_id}: Bought {self.conf.currency} at {price_per_crypto}"})
                            except Exception as e:
                                self.session.rollback()
                                self.logger.error(f"Task {self.task_id}: Error during buy operation: {e}", exc_info=True)
                                continue

                # Removed SQLObject session.commit() from here; SQLAlchemy commits are per operation for now

                if self.conf.delay_secs > 0:
                    for _ in range(self.conf.delay_secs):
                        if self.stop_event.is_set():
                            self.logger.info(f"Task {self.task_id}: Stop event detected during delay for {self.conf.currency}.")
                            break
                        time.sleep(1)
                if self.stop_event.is_set():
                     self.logger.info(f"Task {self.task_id}: Stop event detected for {self.conf.currency}, exiting run loop.")
                     break
            
            if current_transaction and current_transaction.buy_date_time and not current_transaction.sell_date_time:
                self.logger.info(f"Task {self.task_id}: Loop ended with an active open trade for {self.conf.currency}. Tx ID: {current_transaction.id}")
                self.results_queue.put({"status": "info", "message": f"Task {self.task_id}: Ended with active trade for {self.conf.currency}."})

        except Exception as e:
            self.logger.error(f"Task {self.task_id}: Unhandled exception in run loop for {self.conf.currency}: {e}", exc_info=True)
            self.results_queue.put({"status": "error", "message": f"Task {self.task_id}: Failed with error: {e}"})
            try:
                self.session.rollback() # Rollback on unhandled error in main try-except
            except Exception as rb_e:
                self.logger.error(f"Task {self.task_id}: Critical error during session rollback: {rb_e}", exc_info=True)
        finally:
            self.logger.info(f"Task {self.task_id}: Trading run loop for {self.conf.currency} stopped.")
            self.results_queue.put({"status": "stopped", "message": f"Task {self.task_id}: Stopped for {self.conf.currency}."})
            # Session closing is handled by the Task class, not here.
            # Removed: if self.db_conn: self.db_conn.close()

    def stop(self):
        """Signals the trading task to stop."""
        self.logger.info(f"Task {self.task_id}: Stop method called for {self.conf.currency}. Setting stop_event.")
        self.stop_event.set()
        self.results_queue.put({"status": "stopping", "message": f"Task {self.task_id}: Stop signal sent for {self.conf.currency}."})

    def is_running(self):
        """Check if the trading loop is running."""
        return self.loop == 1

    def profits(self):
        return model.get_profits()

    def reset_trading_state(self):
        """Resets trading state for the currency of this task using SQLAlchemy session."""
        self.logger.info(f"Task {self.task_id}: Resetting trading state for {self.conf.currency}.")
        try:
            db_core_ops.reset_trading_transactions(self.session, currency_pair=self.conf.currency, task_id=str(self.task_id))
            # Algo reset might involve DB ops, so pass session
            self.algo_if.reset(self.session, currency=self.conf.currency)
            self.results_queue.put({"status": "info", "message": f"Task {self.task_id}: Trading state reset for {self.conf.currency}."})
            self.logger.info(f"Task {self.task_id}: Trading state reset complete for {self.conf.currency}.")
        except Exception as e:
            self.session.rollback() # Rollback if reset fails
            self.logger.error(f"Task {self.task_id}: Error resetting trading state: {e}", exc_info=True)
            self.results_queue.put({"status": "error", "message": f"Task {self.task_id}: Error resetting state for {self.conf.currency}."})
