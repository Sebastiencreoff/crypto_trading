import csv
import json
import logging
import os
import random

from . import connection

DFT_FEE_PERCENT = 0.015

class EndOfProcess(Exception):
    pass


class SimulationConnect(connection.Connect):
    """Simulated API connection for trading."""
    
    def __init__(self, config_dict, dir_path=None):
        """Initialisation of all configuration needed.

            :param config_dict: configuration dictionary for connection

            configuration example:
            empty file
        """
        logging.info('SimulationConnect::building')
        super().__init__(config_dict)
        cfg = json.load(open(config_dict, mode='r'))

        self.fee_percent = cfg.get('fee_percent', DFT_FEE_PERCENT)
        self.value = 0

        if cfg.get('type', 'random') == 'file':
            file_path = dir_path + cfg.get('file')
            assert os.path.isfile(file_path)
            self.value_func = self.read_value
            self.index = 0
            with open(file_path, 'r') as csv_file:
                self.values = [float(x[0]) for x in csv.reader(csv_file)]
        else:
            self.value_func = self.get_random_value

    def get_value(self, currency=None):
        return self.value_func(currency)

    def read_value(self, currency=None):
        try:
            self.value = self.values[self.index]
            self.index += 1
            return self.value
        except IndexError:
            raise EndOfProcess

    def get_random_value(self, currency=None):
        """Get currencies from coinBase in refCurrency.

            :param currency: currency
            :return : value of the currency

            :raise NameError: if currency not found
            :example :
                >>> get_currency(currency='EUR')
                920
        
        """
        # price simulation
        self.value += random.randint(-100, 100)
            
        if self.value <= 0:
            self.value = random.randint(0, 100)

        logging.info('response: %s', self.value)
           
        return self.value

    def buy(self, amount, currency, currency_value):
        """Buy currency in EUR, currency is defined at class initialisation.

            :param amount: amount value
            :param currency: currency to buy
            :param currency_value: current currency value.
            :return : currency_value bought and fee amount.
            :example :
            >>> buy_currency(amount=10)
                0.2, 0.01
        """
        fee_amount = amount * self.fee_percent
        buy_amount = (amount - fee_amount) / self.value
        logging.warning('success currency: %s '
                        'amount: %s/%s (%s in EUR) '
                        'fee_amount: %s',
                        currency,
                        buy_amount,
                        amount,
                        self.value,
                        fee_amount)
        return buy_amount, fee_amount

    def sell(self, amount, currency, currency_value):
        """Sell currency, currency is defined at class initialisation.

            :param amount: amount value in currency
            :param currency: currency to sell
            :param currency_value: current currency value.
            :return : currency_value bought and fee amount or None, None

            :example :
                >>> sell(amount=0.1, currency='BTC')
                10.1, 0.1
        """
        fee_amount = amount * self.value * self.fee_percent
        logging.warning('success currency: %s '
                        'amount: %s/%s (%s in %s),'
                        'fee_amount: %s',
                        currency,
                        amount,
                        amount * self.value,
                        self.value,
                        currency,
                        fee_amount)
        return self.value, fee_amount

