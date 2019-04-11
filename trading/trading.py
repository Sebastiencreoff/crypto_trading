#!/usr/bin/env python

import json
import logging
import time
import threading

import trading.algo.algoMain
import trading.algo.maxLost
import trading.config as cfg
import trading.connection.simulation
import trading.connection.coinBase
import trading.model


class Trading(threading.Thread):
    """Trading process."""
    
    def __init__(self):
        """Initialisation of all configuration needed."""

        self.loop = 1

        # Connection
        self.connect = None 
        if cfg.conf.connection == 'coinbase':
            self.connect = trading.connection.coinBase.CoinBaseConnect(
                cfg.conf.connection_config)
        else:
            self.connect = trading.connection.simulation.SimulationConnect(
                cfg.conf.connection_config)

        # Algo.
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

        trading.model.create()
        prev_currency_value = None
        trans = trading.model.get_current_trading()

        while self.loop == 1:
            currency_value = self.connect.get_value(cfg.conf.currency)
            if prev_currency_value != currency_value:
                logging.warning('Currency Value: %s', currency_value)
                # Update previous currency
                prev_currency_value = currency_value

                if not trans:
                    trans = trading.model.Trading(
                        buy_value=cfg.conf.transaction_amt,
                        currency=cfg.conf.currency)

                # Process trading
                result = self.algo_if.process(currency_value)
                if trans.buy_date_time:
                    if (result < 0
                        or self.security.process(
                                trans.buy_value,
                                currency_value * trans.currency_buy_amt)):
                        trans.save_sell(*self.connect.sell(
                            trans.currency_buy_amt, trans.currency,
                            currency_value))
                        trans = None

                elif result > 0:
                    trans = trans.save_buy(*self.connect.buy(
                        trans.buy_value, trans.currency, currency_value))

            time.sleep(cfg.conf.delay)
            
    def stop(self):
        """Stop trading."""

        logging.info('stop request received')
        self.loop = 0
