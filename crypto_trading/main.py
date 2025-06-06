#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import signal
import sys
import threading

import crypto_trading.config
import crypto_trading.trading
from .slack.slack_interface import SlackInterface # Updated import path


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

    trading = crypto_trading.trading.Trading(args.config)

    # Initialize SlackInterface
    # Note: This assumes that trading.conf will have 'slack_token' and 'slack_channel_id'
    # These need to be added to the configuration file (e.g., trading_SIMU.json)
    try:
        slack_interface = SlackInterface(conf=trading.conf, trading_instance=trading)
        # Start Slack listening in a separate thread
        slack_thread = threading.Thread(target=slack_interface.start_listening, daemon=True)
        slack_thread.start()
    except AttributeError as e:
        logging.error(f"Failed to initialize SlackInterface. Missing configuration? {e}")
        # Depending on requirements, you might want to exit or continue without Slack.
        # For now, we'll log the error and continue.
        slack_interface = None # Ensure it's defined for potential later checks
    except Exception as e:
        logging.error(f"An unexpected error occurred during SlackInterface initialization or thread starting: {e}")
        slack_interface = None


    # Stopping properly.
    for stop in [signal.SIGTERM, signal.SIGQUIT, signal.SIGINT]:
        signal.signal(stop, lambda x, y: trading.stop())

    trading.run()


if __name__ == '__main__':
    main()

