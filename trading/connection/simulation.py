import logging
import random

import trading.connection.connection


class SimulationConnect(trading.connection.connection.Connect):
    """CoinBase API connection."""
    
    def __init__(self, config_dict):
        """Initialisation of all configuration needed.

            :param config_dict: configuration dictionary for connection

            configuration example:
            empty file
        """
        logging.info('SimulationConnect::building')
        super().__init__('SIMU', config_dict)

        self.currency = 0

    def get_currency(self, ref_currency='EUR'):
        """Get currencies from coinBase in refCurrency.

            :param ref_currency: reference value
            :return : value of the currency

            :raise NameError: if currency not found
            :example :
                >>> get_currency(ref_currency='EUR')
                920
        
        """
        # price simulation
        self.currency += random.randint(-100, 100)
            
        if self.currency <= 0:
            self.currency = random.randint(0, 100)

        logging.info('response: %s', self.currency)
           
        return self.currency

    def buy_currency(self, amount=0):
        """Buy currency.

            :param amount: amount value
            :return : boolean which indicate if it succeed,
                      feeAmt ( set to 0 if failed)
            :example :
                >>> buy_currency(amount=10)
                0.01 or None
        """
        if self.database.get_current_transaction() is None:
            fee_amount = amount * 0.01
            self.database.buy(amount, amount*self.currency,
                              fee_amount*self.currency)
            logging.warning('success currency: %s amount: %d, fee_amount: %d',
                         self.currency,
                         amount*self.currency,
                         fee_amount*self.currency)
            return fee_amount

        logging.error('another transaction is already processing')
        return None

    def sell_currency(self, amount=0):
        """Sell currency.

            :param amount: amount value
            :return : boolean which indicate if it succeed,

            :example :
                >>> sell_currency(amount=10)
                true
        """

        transaction = self.database.get_current_transaction()
        if transaction is not None:
            logging.warning('success currency: %s amount: %d, gain: %d',
                            self.currency,
                            amount*self.currency,
                            amount * self.currency
                            - transaction.buy_value
                            - transaction.buy_value_fee)
            
            return self.database.sell(transaction.buy_date_time,
                                      amount*self.currency)
        else:
            logging.error('no transaction is already processing')
            return None

