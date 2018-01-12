import json
import logging
import os.path

import coinbase.wallet.client

import trading.connection.connection

REF_CURRENCY = 'EUR'


class CoinBaseConnect(trading.connection.connection.Connect):
    """coinBase API connection."""
    
    def __init__(self,  currency, config_dict):
        """Initialisation of all configuration needed.

        :param currency:   currency to deal with ( BTC, etc...)
        :param config_dict: configuration dictionary for connection

        configuration example:

            'configDict' : 'config/coinBase.json'
            'simulation' : No

            and configDict is a jso0n file which contained:
            {'api_key' : xxxx , 'api_secret' : xxx, 'simulation' : true }

        """
        logging.info('')

        assert os.path.isfile(config_dict)
        self.__dict__ = json.load(open(config_dict, mode='r'))
        self.simulation = self.__dict__.get('simulation', True)

        self.client = coinbase.wallet.client.Client(
            self.__dict__.get('api_key', None),
            self.__dict__.get('api_secret', None))

        # Check payment method
        self.payment_method = None

        logging.debug('coinbase payment_method: %s',
                      self.client.get_payment_methods())

        for payment_method in self.client.get_payment_methods().data:
            if payment_method['type'] == 'fiat_account':
                self.payment_method = payment_method

        if not self.payment_method:
            logging.critical('fiat_account not found')
            raise NameError('Only fiat_account is accepted')

        logging.info('coinbase payment_method id: %s', self.payment_method.id)

        # Get account currency
        self.account = None

        logging.debug('coinbase client accounts: %s',
                      self.client.get_accounts())

        for account in self.client.get_accounts().data:
            if account['currency'] == currency:
                self.account = account

        if not self.account:
            logging.critical('account not found for currency: %s', currency)
            raise NameError('Invalid currency')

        logging.debug('coinbase account: %s', self.account)
        super().__init__(currency, config_dict)

    def get_currency(self, ref_currency='EUR'):
        """Get currencies from coinBase in refCurrency.

            :param ref_currency: value to get

            :return : value of the currency

            :raise NameError: if currency not found
            :example :
                >>> get_currency(ref_currency='EUR')
                920
        """

        rates = self.client.get_exchange_rates(currency=self.currency)

        if isinstance(rates, dict):
            logging.info('%s', rates['rates'][ref_currency])
            return float(rates['rates'][ref_currency])
        
        logging.error('error in response')
        return None

    def buy_currency(self, amount=0, currency_value=0):
        """Buy currency, currency is defined at class initialisation.

            :param amount: amount value
            :return : boolean which indicate if it succeed,
                      feeAmt ( set to 0 if failed)
            :example :
            >>> buy_currency(amount=10)
                true, 0.01
        """
 
        if self.database.get_current_transaction() is not None:
            logging.error('another transaction is already processing')
            return None  

        try:
            buy = self.client.buy(self.account.id,
                                  amount=amount,
                                  currency=REF_CURRENCY,
                                  payment_method=self.payment_method.id,
                                  quote=self.simulation)

        except coinbase.wallet.errors.CoinbaseError as e:
            logging.critical('Buy error: {}'.format(e))
        else:
            logging.debug('response: %s', buy)

            logging.warning('success currency: %s '
                            'amount: %s/%s (%s in %s) '
                            'fee_amount: %s',
                            self.currency,
                            buy.amount.amount,
                            buy.subtotal.amount,
                            currency_value,
                            REF_CURRENCY,
                            buy.fee.amount)

            self.database.buy(currency_value,
                              buy.amount.amount,
                              buy.fee.amount)
            return buy
        return None

    def sell_currency(self, transaction, currency_value=0):
        """Sell currency, currency is defined at class initialisation.

            :param transaction: transaction
            :return : boolean which indicate if it succeed,

            :example :
                >>> sell_currency(amount=10)
                true
        """

        try:
            sell = self.client.sell(self.account.id,
                                    amount=transaction.buy_value,
                                    currency=self.currency,
                                    payment_method=self.payment_method.id,
                                    quote=self.simulation)

        except coinbase.wallet.errors.CoinbaseError as e:
            logging.critical('Sell error: {}'.format(e))
        else:
            logging.debug('response: %s', sell)
            gain = self.database.sell(transaction,
                                      currency_value,
                                      sell.fee.amount)

            logging.warning('success currency: %s '
                            'amount: %s/%s (%s in %s),'
                            'fee_amount: %s, gain: %s',
                            self.currency,
                            sell.amount.amount,
                            sell.subtotal.amount,
                            currency_value,
                            REF_CURRENCY,
                            sell.fee.amount,
                            gain)
            return True
        return False
