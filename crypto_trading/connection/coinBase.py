import functools
import json
import logging
import os.path
import requests

import coinbase.wallet.client

from . import connection

REF_CURRENCY = 'EUR'
MAX_TRY = 10


def manage_exception(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs): # Added **kwargs
        count = 0
        while count < MAX_TRY:
            try:
                count += 1
                return func(*args, **kwargs) # Added **kwargs
            except coinbase.wallet.error.CoinbaseError as e:
                logging.critical('Coinbase error: {}'.format(e))
                raise e
            except requests.exceptions.RequestException as e:
                logging.warning('Exception error: {}'.format(e))

    return wrapper


class CoinBaseConnect(connection.Connect):
    """coinBase API connection."""

    def __init__(self, config_dict):
        """Initialisation of all configuration needed.

        :param config_dict: configuration dictionary for connection

        configuration example:

            'configDict' : 'config/coinBase.json'
            'simulation' : No

            and configDict is a jso0n file which contained:
            {'api_key' : xxxx , 'api_secret' : xxx, 'simulation' : true }

        """
        logging.info('')
        assert os.path.isfile(config_dict)
        cfg = json.load(open(config_dict, mode='r'))
        self.simulation = cfg.get('simulation', True)

        self.client = coinbase.wallet.client.Client(
            cfg.get('api_key', None),
            cfg.get('api_secret', None))

        # Check payment method
        self.payment_method = None
        try:
            payment_methods_response = self.client.get_payment_methods()
            logging.debug('coinbase payment_method response: %s', payment_methods_response)
            if payment_methods_response and payment_methods_response.data:
                for payment_method_data in payment_methods_response.data:
                    # Assuming payment_method_data is a dict-like object
                    if payment_method_data.get('type') == 'fiat_account':
                        self.payment_method = payment_method_data # Store the dict-like object
                        break # Found a fiat_account, no need to continue
        except requests.exceptions.RequestException as e:
            logging.warning("Failed to get payment methods during init, possibly due to invalid credentials or network issue: %s", e)
            # self.payment_method remains None, subsequent check will handle it

        if not self.payment_method:
            logging.critical('fiat_account not found or failed to retrieve payment methods.')
            # Consider if NameError is still appropriate or if a custom exception is better
            raise NameError('Fiat account not found or failed to retrieve payment methods. Check credentials and API access.')

        # If self.payment_method is set, it should be the dict-like object. Access its id via get.
        logging.info('coinbase payment_method id: %s', self.payment_method['id'] if self.payment_method else "N/A") # Changed .get('id') to ['id'] for consistency if it's a dict
        super().__init__(config_dict)

    def _get_account_id(self, currency):
        # Ensure this returns a dict, as used by buy/sell methods now expecting ['id']
        for account in self.client.get_accounts().data: # This data is a list of dicts from setUp
            if account['currency'] == currency:
                return account

        logging.critical('account not found for currency: %s', currency)
        raise NameError('Invalid currency')

    @manage_exception
    def get_value(self, currency=None):
        """Get currencies from coinBase in EUR.

            :param currency: currency value to get

            :return : value of the currency

            :raise NameError: if currency not found
            :example :
                >>> get_currency(currency='BTC')
                920
        """
        rates = self.client.get_exchange_rates(currency=currency)

        if isinstance(rates, dict):
            logging.info('%s', rates['rates'][REF_CURRENCY])
            return float(rates['rates'][REF_CURRENCY])

        logging.error('error in response')
        return None

    @manage_exception
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
        account_id = self._get_account_id(currency)['id']
        payment_method_id = self.payment_method['id']
        buy = self.client.buy(account_id,
                              amount=amount,
                              currency=REF_CURRENCY,
                              payment_method=payment_method_id,
                              quote=self.simulation)

        logging.debug('response: %s', buy)

        logging.warning('success currency: %s '
                        'amount: %s/%s (%s in %s) '
                        'fee_amount: %s',
                        currency,
                        buy.amount.amount,
                        buy.subtotal.amount,
                        currency_value,
                        REF_CURRENCY,
                        buy.fee.amount)
        return float(buy.amount.amount), float(buy.fee.amount)

    @manage_exception
    def sell(self, amount, currency, currency_value):
        """Sell currency, currency is defined at class initialisation.

            :param amount: amount value in currency
            :param currency: currency to sell
            :param currency_value: current currency value.
            :return : amount sell in Eur, fee amount in Eur

            :example :
                >>> sell(amount=0.1, currency='BTC')
                10.1, 0.1
        """
        account_id = self._get_account_id(currency)['id']
        payment_method_id = self.payment_method['id']
        sell = self.client.sell(account_id,
                                amount=amount,
                                currency=currency,
                                payment_method=payment_method_id,
                                quote=self.simulation)

        logging.debug('response: %s', sell)

        logging.warning('success currency: %s '
                        'amount: %s/%s (%s in %s),'
                        'fee_amount: %s',
                        currency,
                        sell.amount.amount,
                        sell.subtotal.amount,
                        currency_value,
                        REF_CURRENCY,
                        sell.fee.amount)
        return currency_value, float(sell.fee.amount)
