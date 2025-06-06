# code/crypto_trading/run_task_in_container.py
import os
import json
import logging
import sys
import signal # For graceful shutdown
import threading # For stop_event
from datetime import datetime # For potential logging timestamps if needed

# Ensure the 'code' directory is in the Python path.
# Dockerfile sets WORKDIR /app and PYTHONPATH="/app:${PYTHONPATH}".
# Code is copied to /app/code/.
# This sys.path.append adds /app/code to the path if the script is in /app/code/crypto_trading
# This allows imports like 'from trading_service.core import Trading'
# and 'from config_management.schemas import ...'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) # Goes up two levels to /app

from trading_service.core import Trading # Changed import
from config_management.schemas import AppConfig, ExchangeConfig, AlgoConfig # For structured config
from crypto_trading.database.core_operations import init_db, get_db_session # For db connection

# Configure basic logging to standard output
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', # Added name to format
                    stream=sys.stdout)

# Get a logger for this specific script
logger = logging.getLogger("run_task_in_container")

def main():
    logger.info("Starting trading task container.")

    # --- Signal Handling Setup ---
    stop_event = threading.Event() # Use threading.Event

    def signal_handler(signum, frame):
        signal_name = signal.Signals(signum).name
        logger.info(f"Signal {signal_name} (ID: {signum}) received, setting stop event...")
        stop_event.set()
        # Potentially add a small delay or a second signal for forceful exit if needed
        # For now, just setting the event and letting the Trading class handle shutdown.

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    logger.info("Signal handlers for SIGTERM and SIGINT registered.")

    # --- Configuration Loading ---
    task_config_json = os.getenv("TASK_CONFIG_JSON")
    if not task_config_json:
        logger.error("TASK_CONFIG_JSON environment variable not set.")
        sys.exit(1)

    try:
        task_config_dict = json.loads(task_config_json)
        logger.info(f"Successfully parsed task configuration: {task_config_dict.get('task_id', 'N/A')}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode TASK_CONFIG_JSON: {e}")
        sys.exit(1)

    task_id_str = task_config_dict.get("task_id", "unknown_task_in_container")
    logger.name = f"run_task_in_container-{task_id_str}" # Update logger name with task_id

    # --- Database Connection ---
    # Database URI will be sourced exclusively from TASK_CONFIG_JSON via app_settings.database_url

    db_session = None # Initialize to None
    engine_to_use = None

    # --- Pydantic Config Object Creation ---
    try:
        app_conf_data = task_config_dict.get("app_settings", {})
        exchange_conf_data = task_config_dict.get("exchange_settings", {})
        algo_conf_data = task_config_dict.get("algo_settings", {})
        task_params_data = task_config_dict.get("task_parameters", {})

        # AppConfig is expected to contain database_url directly from TASK_CONFIG_JSON's app_settings
        app_config_obj = AppConfig(**app_conf_data)
        exchange_config_obj = ExchangeConfig(**exchange_conf_data)
        algo_config_obj = AlgoConfig(**algo_conf_data)

        logger.info(f"AppConfig loaded: {app_config_obj.model_dump_json(indent=2)}")
        logger.info(f"ExchangeConfig loaded: {exchange_config_obj.model_dump_json(indent=2)}")
        logger.info(f"AlgoConfig loaded: {algo_config_obj.model_dump_json(indent=2)}")
        logger.info(f"TaskParams for Trading: {task_params_data}")

        # Now establish DB connection using the definitive URI from AppConfig
        if not app_config_obj.database or not app_config_obj.database.database_url:
            logger.error("Database URL is not defined in AppConfig from TASK_CONFIG_JSON.")
            sys.exit(1)

        definitive_db_uri = str(app_config_obj.database.database_url)

        logger.info(f"Using database URI: {definitive_db_uri} for this task (from TASK_CONFIG_JSON).")
        engine_to_use = init_db(definitive_db_uri)
        SessionLocal = get_db_session(engine_to_use)
        db_session = SessionLocal()
        logger.info("Database session created successfully using URI from TASK_CONFIG_JSON.")

    except Exception as e:
        logger.error(f"Error creating Pydantic config objects or DB session: {e}", exc_info=True)
        if db_session: db_session.close()
        sys.exit(1)

    # --- Trading Instance Execution ---
    trading_instance = None
    try:
        logger.info(f"Initializing Trading instance for task: {task_id_str}")
        trading_instance = Trading(
            app_config=app_config_obj,
            exchange_config=exchange_config_obj,
            algo_config=algo_config_obj,
            task_params=task_params_data,
            session=db_session, # Pass the active SQLAlchemy session
            task_id=task_id_str,
            stop_event=stop_event,
            results_queue=None # Explicitly None, as per recent changes
        )
        logger.info(f"Starting trading task execution: {task_id_str}")
        trading_instance.run() # This is the main blocking call

        logger.info(f"Trading task {task_id_str} completed its run method.")

    except Exception as e:
        logger.error(f"Error during trading task execution for {task_id_str}: {e}", exc_info=True)
        if db_session: db_session.close()
        sys.exit(1) # Exit with error code
    finally:
        logger.info(f"Closing database session for task {task_id_str}.")
        if db_session:
            db_session.close()
        if engine_to_use: # Dispose of the engine if created
            engine_to_use.dispose()
            logger.info("Database engine disposed.")


    if stop_event.is_set():
        logger.info(f"Trading task container for {task_id_str} finished due to stop signal.")
    else:
        logger.info(f"Trading task container for {task_id_str} finished successfully.")

    sys.exit(0) # Ensure a 0 exit code on graceful completion or signaled stop

if __name__ == "__main__":
    main()
