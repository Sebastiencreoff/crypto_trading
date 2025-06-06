import logging
import time
import datetime
import threading
import queue
import httpx # Added for sending notifications
from typing import Optional # Added for task_id_for_log type hint.

# Assuming new config management is in a package accessible via PYTHONPATH
from config_management.schemas import AppConfig, ExchangeConfig, AlgoConfig, DatabaseConfig

# Database imports
from crypto_trading.database import core_operations as db_core_ops
from crypto_trading.database import models as db_models
from crypto_trading.database.core_operations import get_total_profit

# Algorithm and connection imports
from crypto_trading import algo
from crypto_trading import connection

logger_core = logging.getLogger(__name__) # Renamed to avoid conflict with Trading.logger

# --- Notification Helper ---
def _send_notification_sync(message: str, notification_url: Optional[str], task_id_for_log: Optional[str] = "N/A"):
    """
    Synchronously sends a notification message to the notification service.
    A task_id can be provided for more contextual logging.
    """
    if not notification_url:
        logger_core.warning(f"Task {task_id_for_log}: Notification service URL not configured. Cannot send notification: '{message[:50]}...'")
        return

    try:
        # NotificationRequest schema: {"message": str, "channel_id": Optional[str]}
        # Sending to default channel, so channel_id is not specified.
        payload = {"message": message}

        # Increased timeout, default is 5s, might be too short for service-to-service over k8s network sometimes
        with httpx.Client(timeout=10.0) as client:
            response = client.post(f"{notification_url}/notify", json=payload)

        if response.status_code == 200: # Assuming Notification Service returns 200 on success
            logger_core.info(f"Task {task_id_for_log}: Notification sent successfully: '{message[:50]}...'")
        else:
            logger_core.error(f"Task {task_id_for_log}: Failed to send notification. Status: {response.status_code}, Response: {response.text[:100]}")
    except httpx.RequestError as e:
        logger_core.error(f"Task {task_id_for_log}: HTTP request error sending notification to {notification_url}/notify: {e}", exc_info=True)
    except Exception as e:
        logger_core.error(f"Task {task_id_for_log}: Unexpected error sending notification: {e}", exc_info=True)


class Trading:
    """
    Trading process, adapted to run as a managed task, configured via AppConfig.
    The run method is intended to be executed in a background thread/task.
    """

    def __init__(self,
                 app_config: AppConfig,          # Full application config
                 exchange_config: ExchangeConfig,
                 algo_config: AlgoConfig,
                 task_params: dict,
                 session, # SQLAlchemy session
                 task_id: str,
                 stop_event: threading.Event,
                 results_queue: queue.Queue):

        self.app_config = app_config
        self.exchange_config = exchange_config
        self.algo_config = algo_config
        self.task_params = task_params
        self.session = session
        self.task_id = task_id
        self.stop_event = stop_event
        self.results_queue = results_queue

        # Extract notification URL from AppConfig
        self.notification_url = str(app_config.notification_service_url) if app_config.notification_service_url else None

        self.logger = logging.getLogger(f"TradingTask-{self.task_id}") # Instance-specific logger
        self.logger.info(f"Initializing trading task for currency {self.task_params['currency']}")
        if self.notification_url:
            self.logger.info(f"Notification service URL configured at: {self.notification_url}")
        else:
            self.logger.warning("Notification service URL not configured for this task.")


        self.connect = None
        if self.exchange_config.name == 'simulation':
            self.connect = connection.simulation.SimulationConnect(
                config_data=self.exchange_config.extra_settings,
                base_dir=str(self.app_config.base_config_path) if self.app_config.base_config_path else None
            )
        elif self.exchange_config.name == 'binance':
            self.connect = connection.binance.BinanceConnect(exchange_config=self.exchange_config)
        else:
            error_msg = f"Task {self.task_id}: Unknown exchange type: {self.exchange_config.name}"
            self.logger.error(error_msg)
            self._notify_sync(f"ERROR: {error_msg}") # Send notification on critical init error
            self.results_queue.put({"status": "error", "message": error_msg, "task_id": self.task_id})
            raise ValueError(error_msg)

        mock_algo_conf_for_main = {
            "algo_config_dict": self.algo_config.parameters,
            "currency": self.task_params['currency'],
            "transactionAmt": self.task_params['transaction_amount'],
            "delay": self.app_config.other_settings.get('delay_seconds', 60),
            "maxLost": self.algo_config.parameters.get("maxLost", {}),
            "takeProfit": self.algo_config.parameters.get("takeProfit", {}),
            "AIAlgo": self.algo_config.parameters.get("AIAlgo", {}),
        }
        self.algo_if = algo.AlgoMain(config_override=mock_algo_conf_for_main)
        self.security = algo.Security(config_override=mock_algo_conf_for_main)
        self.logger.info(f"Initialization complete for {self.task_params['currency']}.")

    def _notify_sync(self, message: str):
        """Internal helper to send notification using instance's notification_url and task_id."""
        _send_notification_sync(message, self.notification_url, self.task_id)


    def run(self):
        currency = self.task_params['currency']
        transaction_amount = self.task_params['transaction_amount']
        delay_seconds = self.app_config.other_settings.get('delay_seconds', 60)

        self.logger.info(f"Trading run loop starting for {currency}.")
        self._notify_sync(f"Task {self.task_id} for {currency} started.")
        self.results_queue.put({"status": "info", "message": f"Task {self.task_id}: Started for {currency}", "task_id": self.task_id})

        prev_currency_value = None
        current_transaction = db_core_ops.get_open_transaction(
            self.session, currency_pair=currency, task_id=self.task_id
        )
        self.logger.info(f"Initial transaction status: {'Ongoing ID: ' + str(current_transaction.id) if current_transaction else 'None'}")

        try:
            while not self.stop_event.is_set():
                try:
                    currency_value = self.connect.get_value(currency)
                    if currency_value is None and self.exchange_config.name == 'simulation':
                        self.logger.info(f"Simulation data source exhausted for {currency}.")
                        self._notify_sync(f"Task {self.task_id} for {currency}: Simulation ended.")
                        self.results_queue.put({"status": "info", "message": "Simulation ended.", "task_id": self.task_id})
                        break
                except connection.EndOfProcess:
                    self.logger.info(f"EndOfProcess signal received for {currency}.")
                    self._notify_sync(f"Task {self.task_id} for {currency}: EndOfProcess signal received.")
                    self.results_queue.put({"status": "info", "message": "EndOfProcess.", "task_id": self.task_id})
                    break
                except Exception as e:
                    self.logger.error(f"Error getting currency value: {e}", exc_info=True)
                    self._notify_sync(f"Task {self.task_id} for {currency}: CRITICAL ERROR getting currency value: {e}")
                    self.results_queue.put({"status": "error", "message": f"Error getting value: {e}", "task_id": self.task_id})
                    time.sleep(delay_seconds) # Consider a backoff strategy for repeated errors
                    continue

                if prev_currency_value != currency_value:
                    self.logger.debug(f"Currency Value for {currency}: {currency_value}")
                    prev_currency_value = currency_value

                    try:
                        db_core_ops.save_price_tick(
                            self.session, currency_pair=currency, price=currency_value,
                            timestamp=datetime.datetime.utcnow()
                        )
                    except Exception as e:
                        self.logger.error(f"Failed to save price {currency_value} for {currency}: {e}", exc_info=True)

                    algo_signal = self.algo_if.process(self.session, currency_value, currency=currency)
                    self.security.process(self.session, currency_value, currency=currency)

                    if current_transaction and current_transaction.buy_date_time:
                        should_sell_security = self.security.sell(
                            self.session, current_value=currency_value,
                            transaction=current_transaction, currency=currency
                        )
                        if algo_signal < 0 or should_sell_security:
                            try:
                                sell_price, _ = self.connect.sell(
                                    current_transaction.base_currency_bought_amount,
                                    current_transaction.currency_pair, currency_value
                                )
                                current_transaction.sell_date_time = datetime.datetime.utcnow()
                                current_transaction.base_currency_sell_price = sell_price
                                current_transaction.sell_fee_eur = 0.0
                                profit = ( (sell_price * current_transaction.base_currency_bought_amount) -
                                           current_transaction.buy_value_eur -
                                           (current_transaction.buy_fee_eur or 0.0) -
                                           (current_transaction.sell_fee_eur or 0.0) )
                                current_transaction.profit_eur = profit
                                self.session.commit()
                                success_msg = f"Sold {current_transaction.base_currency_bought_amount:.6f} of {currency} at {sell_price:.2f}. Profit: {profit:.2f} EUR. TxID: {current_transaction.id}"
                                self.logger.info(success_msg)
                                self._notify_sync(f"Task {self.task_id}: {success_msg}")
                                self.results_queue.put({"status": "sold", "message": success_msg, "task_id": self.task_id})
                                current_transaction = None
                            except Exception as e:
                                self.session.rollback()
                                error_msg = f"Error during sell operation: {e}"
                                self.logger.error(error_msg, exc_info=True)
                                self._notify_sync(f"Task {self.task_id}: ERROR during sell: {e}")
                                continue
                    elif algo_signal > 0 and not current_transaction:
                         should_buy_security = self.security.buy(
                             self.session, current_value=currency_value, currency=currency
                         )
                         if should_buy_security:
                            try:
                                price_per_crypto, amount_of_crypto_bought = self.connect.buy(
                                    transaction_amount, currency, currency_value
                                )
                                current_transaction = db_models.TradingTransaction(
                                    task_id=self.task_id, currency_pair=currency,
                                    buy_value_eur=transaction_amount, buy_fee_eur=0.0,
                                    base_currency_bought_amount=amount_of_crypto_bought,
                                    base_currency_buy_price=price_per_crypto,
                                    buy_date_time=datetime.datetime.utcnow()
                                )
                                self.session.add(current_transaction)
                                self.session.commit()
                                success_msg = f"Bought {amount_of_crypto_bought:.6f} of {currency} at {price_per_crypto:.2f}. TxID: {current_transaction.id}"
                                self.logger.info(success_msg)
                                self._notify_sync(f"Task {self.task_id}: {success_msg}")
                                self.results_queue.put({"status": "bought", "message": success_msg, "task_id": self.task_id})
                            except Exception as e:
                                self.session.rollback()
                                error_msg = f"Error during buy operation: {e}"
                                self.logger.error(error_msg, exc_info=True)
                                self._notify_sync(f"Task {self.task_id}: ERROR during buy: {e}")
                                continue

                if delay_seconds > 0:
                    for _ in range(delay_seconds):
                        if self.stop_event.is_set(): break
                        time.sleep(1)
                if self.stop_event.is_set():
                     self.logger.info(f"Stop event detected for {currency}, exiting run loop.")
                     break

            if current_transaction and current_transaction.buy_date_time and not current_transaction.sell_date_time:
                self.logger.info(f"Loop ended with an active open trade for {currency}. Tx ID: {current_transaction.id}")
                self.results_queue.put({"status": "info", "message": "Ended with active trade.", "task_id": self.task_id})

        except Exception as e:
            self.logger.error(f"Unhandled exception in run loop for {currency}: {e}", exc_info=True)
            self._notify_sync(f"Task {self.task_id} for {currency}: CRITICAL unhandled exception in run loop: {e}")
            self.results_queue.put({"status": "error", "message": f"Failed with error: {e}", "task_id": self.task_id})
            try:
                self.session.rollback()
            except Exception as rb_e:
                self.logger.error(f"Critical error during session rollback: {rb_e}", exc_info=True)
        finally:
            self.logger.info(f"Trading run loop for {currency} stopped.")
            self._notify_sync(f"Task {self.task_id} for {currency} stopped.")
            self.results_queue.put({"status": "stopped", "message": "Stopped.", "task_id": self.task_id})

    def stop(self):
        self.logger.info(f"Stop method called. Setting stop_event.")
        self.stop_event.set()
        self.results_queue.put({"status": "stopping", "message": "Stop signal sent.", "task_id": self.task_id})

    def profits(self):
        if not self.session:
            self.logger.error(f"Database session is not available for profits calculation.")
            return 0.0
        try:
            total_profits = get_total_profit(self.session, task_id=self.task_id)
            return total_profits if total_profits is not None else 0.0
        except Exception as e:
            self.logger.error(f"Error retrieving profits: {e}", exc_info=True)
            return 0.0

    def reset_trading_state(self):
        currency = self.task_params['currency']
        self.logger.info(f"Resetting trading state for {currency}.")
        try:
            db_core_ops.reset_trading_transactions(self.session, currency_pair=currency, task_id=self.task_id)
            self.algo_if.reset(self.session, currency=currency)
            self.results_queue.put({"status": "info", "message": "Trading state reset.", "task_id": self.task_id})
            self.logger.info(f"Trading state reset complete for {currency}.")
            self._notify_sync(f"Task {self.task_id} for {currency}: Trading state has been reset.")
        except Exception as e:
            self.session.rollback()
            self.logger.error(f"Error resetting trading state: {e}", exc_info=True)
            self._notify_sync(f"Task {self.task_id} for {currency}: ERROR resetting trading state: {e}")
            self.results_queue.put({"status": "error", "message": "Error resetting state.", "task_id": self.task_id})
