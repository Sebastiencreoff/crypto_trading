import argparse
import logging
import signal
import sys
import os

# Assuming config_management is in thePYTHONPATH or discoverable
# For a typical project structure, if 'code' is the root for python modules:
# from config_management.loader import load_config
# from config_management.schemas import AppConfig
# However, if run_slack_handler.py is in crypto_trading, and config_management is a sibling to crypto_trading,
# this might require path adjustments or making 'code' a package.
# For now, let's assume they are importable.
# If this script is inside 'crypto_trading' and 'code' is the project root in PYTHONPATH:
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..')) # Add 'code' to path

from config_management.loader import load_config
from config_management.schemas import AppConfig, SlackConfig
from crypto_trading.task_manager import TaskManager
from crypto_trading.slack.slack_interface import SlackCommandHandler

# Basic logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variable to hold the handler instance for signal handling
global_slack_handler = None

def signal_handler(sig, frame):
    logger.info("Shutdown signal received. Stopping Slack command handler...")
    if global_slack_handler and hasattr(global_slack_handler, 'stop'):
        logger.info("Attempting to gracefully stop Slack Command Handler...")
        global_slack_handler.stop() # Requires SlackCommandHandler to have a stop() method
    else:
        logger.info("No explicit stop method or handler not set. Exiting.")
    sys.exit(0)

def main():
    global global_slack_handler

    # Adjust default config path relative to this script's location if needed.
    # Assuming this script is in code/crypto_trading/ and config is in code/config/
    # So, ../config/central_config.json from this script's perspective.
    # Or, if run from project root (e.g. 'code'), then 'config/central_config.json'
    # Let's use a path relative to the project structure assuming 'code' is root.
    # This means if the script is in /app/code/crypto_trading, config is /app/code/config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config_path = os.path.join(script_dir, "..", "..", "config", "central_config.json")
    # Normalize the path to handle ".." correctly
    default_config_path = os.path.normpath(default_config_path)


    parser = argparse.ArgumentParser(description='Slack Command Handler for Crypto Trading Bot')
    parser.add_argument(
        '-c', '--config',
        help=f'Path to the central configuration JSON file. Defaults to {default_config_path}',
        default=default_config_path,
        type=str
    )
    args = parser.parse_args()

    logger.info(f"Loading configuration from: {args.config}")

    try:
        # Type hint for clarity, assuming load_config returns an AppConfig instance or similar
        app_config: AppConfig = load_config(args.config, AppConfig)
        if not app_config:
            logger.error("Failed to load application configuration. load_config returned None.")
            sys.exit(1)

        if not app_config.slack or not isinstance(app_config.slack, SlackConfig):
            logger.error("Slack configuration (slack_config) is missing or invalid in the central config.")
            sys.exit(1)

        # Initialize TaskManager. It loads Kubernetes config internally.
        logger.info("Initializing TaskManager...")
        task_manager = TaskManager()
        logger.info("TaskManager initialized.")

        # SlackCommandHandler expects a 'conf' object. We pass app_config.slack (which is SlackConfig).
        # SlackCommandHandler needs to be compatible with SlackConfig attributes (e.g., bot_token).
        # The existing SlackCommandHandler __init__ uses os.environ.get("SLACK_BOT_TOKEN")
        # This needs to be reconciled. For now, we assume SlackCommandHandler will be updated
        # or that SLACK_BOT_TOKEN is the primary way it gets the token.
        # Let's design SlackCommandHandler to prioritize token from 'conf' if available.
        logger.info("Initializing SlackCommandHandler...")
        slack_handler_conf = app_config.slack # This is a SlackConfig object

        # We need to ensure SLACK_BOT_TOKEN is set for WebClient/RTMClient if SlackCommandHandler relies on it.
        # A better way is to pass the token from slack_handler_conf directly to WebClient/RTMClient.
        # This will be handled in the modification of SlackCommandHandler.
        if not os.environ.get("SLACK_BOT_TOKEN") and slack_handler_conf.bot_token:
             logger.info("SLACK_BOT_TOKEN not set in env, using from config for SlackCommandHandler.")
             # This is a temporary measure; SlackCommandHandler should use the token from its conf.
             os.environ["SLACK_BOT_TOKEN"] = slack_handler_conf.bot_token
        elif not os.environ.get("SLACK_BOT_TOKEN") and not slack_handler_conf.bot_token:
            logger.error("SLACK_BOT_TOKEN is not set in environment and not found in slack_config.bot_token.")
            sys.exit(1)


        slack_handler = SlackCommandHandler(conf=slack_handler_conf, task_manager=task_manager)
        global_slack_handler = slack_handler # For signal handling

        # Add an is_initialized method to SlackCommandHandler
        if hasattr(slack_handler, 'is_initialized') and not slack_handler.is_initialized():
            logger.error("SlackCommandHandler failed to initialize (e.g., Slack client auth failed). Exiting.")
            sys.exit(1)
        elif not hasattr(slack_handler, 'is_initialized'):
            logger.warning("SlackCommandHandler does not have an is_initialized method. Assuming successful initialization.")

        logger.info("Starting Slack command handler...")
        slack_handler.start_listening() # This is blocking

    except FileNotFoundError:
        logger.error(f"Configuration file not found at {args.config}. Please check the path.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during startup: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    main()
