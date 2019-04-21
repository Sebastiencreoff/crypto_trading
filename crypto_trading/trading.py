#!/usr/bin/env python

import json
import logging
import time
import threading

from . import algo
from . import config as cfg
from . import connection
from . import model


class Trading(threading.Thread):
    """Trading process."""
    
    def __init__(self, config_file):
        """Initialisation of all configuration needed."""

        # Load global configuration
        self.conf = cfg.init(config_file)

        self.loop = 1

        # Connection
        self.connect = None 
        if self.conf.connection == 'coinbase':
            self.connect = connection.coinBase.CoinBaseConnect(
                self.conf.connection_config)
        else:
            self.connect = connection.simulation.SimulationConnect(
                self.conf.connection_config, self.conf.dir_path)

        # Algo.
        self.algo_if = algo.AlgoMain(
            self.conf.algo_config)

        self.security = algo.Security(self.conf.algo_config)

        threading.Thread.__init__(self)

    def run(self):
        """Launch the trading process.

         It will contain:
                - 1 thread for data acquisition
                - 1 thread by currency to deal with
        """

        model.create()
        prev_currency_value = None
        trans = model.get_current_trading()

        while self.loop == 1:
            currency_value = self.connect.get_value(self.conf.currency)
            if prev_currency_value != currency_value:
                logging.info('Currency Value: %s', currency_value)
                # Update previous currency
                prev_currency_value = currency_value

                if not trans:
                    trans = model.Trading(buy_value=self.conf.transaction_amt,
                                          currency=self.conf.currency)

                # Process trading
                result = self.algo_if.process(currency_value,
                                              currency=self.conf.currency)

                if trans.buy_date_time:
                    if (result < 0
                        or self.security.process(
                                current_value=currency_value,
                                transaction=trans)):
                        trans.save_sell(*self.connect.sell(
                            trans.currency_buy_amt, trans.currency,
                            currency_value))
                        trans = None

                elif result > 0:
                    trans = trans.save_buy(*self.connect.buy(
                        trans.buy_value, trans.currency, currency_value))

            time.sleep(self.conf.delay)
            
    def stop(self):
        """Stop trading."""

        logging.info('stop request received')
        self.loop = 0
