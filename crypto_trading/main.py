#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import signal
import sys
import threading

import crypto_trading.config
# import crypto_trading.trading # Trading class has moved. This import might be invalid.
# from .slack.slack_interface import SlackInterface # SlackInterface has moved.

# Placeholder for the old Trading class path.
# Depending on the future of this main.py, it might interact with the new trading_service
# or be deprecated. For now, we'll assume it's trying to run some local logic,
# even if the Trading class itself is gone from its original location.
# This part of main.py is likely to be non-functional after current refactorings.
if hasattr(crypto_trading, 'trading') and hasattr(crypto_trading.trading, 'Trading'):
    OriginalTradingClass = crypto_trading.trading.Trading
else:
    OriginalTradingClass = None # Trading class is no longer here


def main():

    parser = argparse.ArgumentParser(description='Crypto Trading Main')
    parser.add_argument('-c', '--config',
                        help='Trading configuration file',
                        required=True,
                        type=str)
    parser.add_argument('-l', '--logging',
                        default='DEBUG',
                        help='logging level(DEBUG,INFO,WARNING,ERROR,CRITICAL)',
                        type=str)
    args = parser.parse_args()

    logging_level_int = getattr(logging, args.logging, logging.DEBUG)

    logging.basicConfig(
        level=logging_level_int,
        format='%(asctime)s  %(levelname)s %(module)s-%(funcName)s: %(message)s'
    )

    # The original Trading class instantiation is problematic as the class has moved
    # and its __init__ signature changed.
    # trading = crypto_trading.trading.Trading(args.config) # This line will likely fail.

    # For the purpose of this subtask, we are only removing Slack logic.
    # The non-functional trading part is a side effect of broader refactoring.
    logging.info("Attempting to initialize and run trading logic (if available).")
    logging.warning("Trading class has moved to trading_service.core. The old main.py entry point for direct trading is likely non-functional.")

    # --- Slack initialization and thread starting logic removed ---
    logging.info("Slack integration has been moved to the Notification Service.")


    # The rest of the trading logic startup will likely fail or needs complete rework
    # For example, if OriginalTradingClass is None:
    if OriginalTradingClass:
        try:
            # This instantiation will fail if args.config is a path to the old config type
            # and not the new AppConfig, or if Trading class expects different args.
            # The old Trading class expected a path, which then loaded a specific JSON structure.
            # The new one in trading_service.core expects AppConfig, ExchangeConfig, etc.
            # This main.py is not equipped to provide those.
            trading_instance = OriginalTradingClass(args.config) # Placeholder, likely problematic

            # Stopping properly.
            for sig_event in [signal.SIGTERM, signal.SIGQUIT, signal.SIGINT]: # Renamed variable to avoid conflict
                signal.signal(sig_event, lambda x, y: trading_instance.stop() if hasattr(trading_instance, 'stop') else sys.exit(0))

            if hasattr(trading_instance, 'run'):
                trading_instance.run()
            else:
                logging.error("Trading instance does not have a 'run' method. Cannot start trading.")
        except Exception as e:
            logging.error(f"Failed to initialize or run the (old) Trading instance: {e}", exc_info=True)
    else:
        logging.error("Original Trading class not found. Cannot start trading via main.py.")
        logging.info("Please use the new trading_service API to manage trading tasks.")

    # Keep the process alive if no trading loop starts, e.g. for other potential daemons (none here now)
    # Or simply exit if the primary purpose (trading) can't run.
    # For now, if trading doesn't run, it will just exit after this.


if __name__ == '__main__':
    main()

