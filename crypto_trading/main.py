#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import signal
import sys

import crypto_trading.config
import crypto_trading.trading


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

    # Stopping properly.
    for stop in [signal.SIGTERM, signal.SIGQUIT, signal.SIGINT]:
        signal.signal(stop, lambda x, y: trading.stop())

    trading.run()


if __name__ == '__main__':
    main()

