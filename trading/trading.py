#!/usr/bin/env python

import logging
import time
import threading

import trading.algo.algoMain
import trading.algo.maxLost
import trading.config as cfg
import trading.connection.simulation
import trading.connection.coinBase


class Trading(threading.Thread):
    """Trading process."""
    
    def __init__(self, config_file):
        """Initialisation of all configuration needed."""

        self.loop = 1

        # Connection
        self.connect = None 
        if cfg.conf.connection == 'coinbase':
            self.connect = trading.connection.coinBase.CoinBaseConnect(
                cfg.conf.currency,
                cfg.conf.connection_config)
        else:
            self.connect = trading.connection.simulation.SimulationConnect(
                cfg.conf.connection_config)

        self.algo_if = trading.algo.algoMain.AlgoMain(
            cfg.conf.algo_config)

        self.security = trading.algo.maxLost.MaxLost(cfg.conf.algo_config)

        threading.Thread.__init__(self)

    def run(self):
        """Launch the trading process.

         It will contain:
                - 1 thread for data acquisition
                - 1 thread by currency to deal with
        """

        prev_currency = None

        while self.loop == 1:
        
            currency = self.connect.get_currency()
            if prev_currency != currency:
                logging.warning('Currency Value: %s', currency)
                # Update previous currency
                prev_currency = currency
                result = self.algo_if.process(currency)
                trans = self.connect.current_transaction()
                # Process trading
                if trans:
                    if (result < 0
                        or self.security.process(trans.currency_buy_value,
                                                 currency)):
                        self.connect.sell_currency(trans, currency)
                elif result > 0:
                    self.connect.buy_currency(cfg.conf.transaction_amt,
                                              currency)

            time.sleep(cfg.conf.delay)
            
    def stop(self):
        """Stop trading."""

        logging.info('stop request received')
        self.loop = 0
