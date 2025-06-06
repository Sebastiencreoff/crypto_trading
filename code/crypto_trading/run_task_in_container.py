# code/crypto_trading/run_task_in_container.py
import os
import json
import logging
import sys
import multiprocessing # Required by Trading class, though stop_event might not be used initially from K8s Job
from datetime import datetime # For potential logging timestamps if needed

# Ensure the 'code' directory is in the Python path
# This is to allow imports like 'from crypto_trading.trading import Trading'
# The Dockerfile already sets WORKDIR /app and copies 'code' to /app/code
# It also sets PYTHONPATH="/app:${PYTHONPATH}"
# So, direct imports should work assuming the script is in /app/code/crypto_trading
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from crypto_trading.trading import Trading
from config_management.loader import Config # Assuming Config class can be used or adapted
from crypto_trading.database.core_operations import init_db, get_db_session # For db connection

# Configure basic logging to standard output
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    stream=sys.stdout)

def main():
    logging.info("Starting trading task container.")

    task_config_json = os.getenv("TASK_CONFIG_JSON")
    if not task_config_json:
        logging.error("TASK_CONFIG_JSON environment variable not set.")
        sys.exit(1)

    try:
        task_config_dict = json.loads(task_config_json)
        logging.info(f"Successfully parsed task configuration: {task_config_dict}")
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode TASK_CONFIG_JSON: {e}")
        sys.exit(1)

    # The Trading class expects a Config object, not just a dict.
    # We need to map the dictionary to a Config object or adapt the Trading class.
    # For now, let's assume we can construct a Config object or it can take a dictionary.
    # This part might need adjustment based on the actual Config class structure.

    # Option 1: If Config class can be instantiated from a dictionary
    # config_obj = Config(task_config_dict) # This depends on Config class implementation

    # Option 2: Manually create a Config-like object or adapt Trading.
    # For now, let's assume the task_config_dict contains all necessary fields
    # that the Trading class's constructor and run method will use.
    # We'll need to ensure the Trading class is flexible or create a proper Config instance.

    # --- Database Connection ---
    # The Trading class expects a db_conn in its config_obj.
    # The main application likely initializes the DB and passes the engine or session.
    # Here, running in a new container, we need to establish a DB connection.
    # This assumes the DB is accessible from the K8s pod.
    # The DB connection string should ideally come from the central_config.json or env variables.

    # Load central config to get database URI
    # This path assumes the Dockerfile copies config/central_config.json to /app/config/central_config.json
    central_config_path = "/app/config/central_config.json"
    db_uri = None
    try:
        with open(central_config_path, 'r') as f:
            central_config_data = json.load(f)
            db_uri = central_config_data.get("database", {}).get("uri")
            logging.info(f"Database URI found: {db_uri is not None}")
    except Exception as e:
        logging.error(f"Could not load database URI from {central_config_path}: {e}")
        sys.exit(1)

    if not db_uri:
        logging.error("Database URI not found in central configuration.")
        sys.exit(1)

    try:
        engine = init_db(db_uri) # Initialize DB engine
        SessionLocal = get_db_session(engine) # Get session factory
        db_session = SessionLocal() # Create a new session for this task
        logging.info("Database session created successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize database connection: {e}")
        sys.exit(1)


    # Construct a config object suitable for the Trading class.
    # This is a placeholder and needs to align with actual Config class structure.
    # For example, if Trading expects specific attributes on config_obj:
    class TaskSpecificConfig:
        def __init__(self, loaded_dict, db_connection):
            self.currency = loaded_dict.get("currency") # Example: "BTC/USD"
            self.exchange = loaded_dict.get("exchange") # Example: "binance"
            self.strategy = loaded_dict.get("strategy") # Example: "moving_average"
            self.interval = loaded_dict.get("interval") # Example: "1h"
            self.transaction_amt = loaded_dict.get("transaction_amt", 100) # Default if not provided
            self.delay_secs = loaded_dict.get("delay_secs", 60) # Default
            self.task_id = loaded_dict.get("task_id", "unknown_task") # For logging/tracking

            # These are often part of a more complex config structure (e.g. connection_config, algo_config)
            # The Trading class might expect these to be nested.
            # This requires careful mapping from task_config_dict to what Trading expects.
            self.connection_type = loaded_dict.get("connection_type", "live") # or "simulation"
            self.connection_config = loaded_dict.get("connection_config", {}) # API keys, etc. (use K8s secrets for live)
            self.algo_config = loaded_dict.get("algo_config", {}) # Strategy specific params
            self.dir_path = loaded_dict.get("dir_path") # For simulation data if applicable

            self.db_conn = db_connection # The active SQLAlchemy session
            self.logger = logging.getLogger(f"TradingTask-{self.task_id}")
            self.paper_trade = loaded_dict.get("paper_trade", True) # Default to paper trading

            # Add other necessary fields based on Trading class and Config requirements.
            # Example: specific API keys for the exchange if not in connection_config
            # self.api_key = loaded_dict.get("api_key")
            # self.api_secret = loaded_dict.get("api_secret")


    try:
        config_for_trading = TaskSpecificConfig(task_config_dict, db_session)
        logging.info(f"Trading configuration prepared for task_id: {config_for_trading.task_id}")
    except Exception as e:
        logging.error(f"Error preparing TaskSpecificConfig: {e}")
        db_session.close()
        sys.exit(1)


    # The Trading class uses a multiprocessing.Event for stopping.
    # In a Kubernetes Job, the typical way to stop is to delete the Job/Pod.
    # The stop_event might not be directly triggerable externally in the same way.
    # For now, we pass a dummy event. If graceful shutdown is needed from within
    # the Trading class based on external signals, this mechanism would need rethinking
    # (e.g. checking a file, a K8s signal, or a short-lived task).
    stop_event = multiprocessing.Event()
    results_queue = multiprocessing.Queue() # Trading class expects this. Results will be logged.

    try:
        logging.info(f"Initializing Trading instance for task: {config_for_trading.task_id}")
        trading_instance = Trading(
            config_obj=config_for_trading,
            task_id=config_for_trading.task_id, # Pass task_id if Trading class uses it
            stop_event=stop_event,
            results_queue=results_queue
        )
        logging.info(f"Starting trading task: {config_for_trading.task_id}")
        trading_instance.run() # This is the main blocking call

        logging.info(f"Trading task {config_for_trading.task_id} completed. Retrieving results from queue.")
        # Process results from the queue and log them
        # This loop ensures all messages from the Trading instance are processed.
        while not results_queue.empty():
            try:
                result = results_queue.get_nowait()
                logging.info(f"Task {config_for_trading.task_id} result: {result}")
            except multiprocessing.queues.Empty:
                break
        logging.info(f"Finished processing results for task {config_for_trading.task_id}.")

    except Exception as e:
        logging.error(f"Error during trading task execution for {config_for_trading.task_id}: {e}", exc_info=True)
        # Ensure results queue is drained even on error, in case some were put before failure
        while not results_queue.empty():
            try:
                result = results_queue.get_nowait()
                logging.error(f"Task {config_for_trading.task_id} partial result on error: {result}")
            except multiprocessing.queues.Empty:
                break
        db_session.close()
        sys.exit(1) # Exit with error code
    finally:
        logging.info(f"Closing database session for task {config_for_trading.task_id}.")
        db_session.close()

    logging.info(f"Trading task container for {config_for_trading.task_id} finished successfully.")
    sys.exit(0) # Ensure a 0 exit code on success

if __name__ == "__main__":
    main()
