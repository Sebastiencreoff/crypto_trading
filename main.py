#!/usr/bin/env python
# coding: utf-8

import getopt
import logging
import signal
import sys

import trading.config
import trading.trading


def usage(argv):
    print(
        """Usage must be:
        {0} -c config_file.json  [-h] (with short attribute)
        {0} -confile_file=config_file.json --help
        
        where:
            -h / --help:    help command
            -c / --config:  configuration file
            -l / --logging:  logging level (DEBUG,INFO,WARNING,ERROR,CRITICAL)
    """.format(argv[0]))


def main(argv):

    try:
        opts, args = getopt.getopt(argv[1:], 'hc:l:',
                                   ['help', 'config_file=', 'logging='])
    except getopt.GetoptError as e:
        usage(argv)
        sys.exit(2)

    config_file = None
    logging_level = 'DEBUG'
    for opt, arg in opts:
        if opt in [ '-h', '--help']:
            usage(argv)
            sys.exit()
        elif opt in ['-c', '--config']:
            config_file = arg
        elif opt in ['-l', '--logging']:
            logging_level = arg

    logging_level_int = getattr(logging, logging_level, logging.DEBUG)

    if not config_file:
        usage(argv)
        sys.exit(2)

    logging.basicConfig(
        level=logging_level_int,
        format='%(asctime)s  %(levelname)s %(module)s-%(funcName)s: %(message)s'
    )

    # Load global configuration
    trading.config.init(config_file)

    # Create new threads
    thread1 = trading.trading.Trading(config_file)

    print("Starting Trading")
    try:
        thread1.start()

    except KeyboardInterrupt:
        thread1.join()

        print("\nTrading Finished!")

    print ("Exiting Main Thread")


if __name__ == '__main__':
    main(sys.argv)

