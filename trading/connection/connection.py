#! /usr/bin/env python


class Connect(object):
    """Connect API connection."""

    def __init__(self, config_dict):
        """Initialisation of all configuration needed.

            :param currency:   currency to deal with ( BTC, etc...)
            :param config_dict: configuration dictionary for connection
        """

    def allow_currency(self, currency='BTC'):
        """Check if currency is available.

            :param currency: currency value to check
            :raise NameError: if currency not found
            :example :
                >>> allow_currency(currency='BTC')
                920
        """
        if currency not in self.allow_currencies:
            raise NameError('Currency: {} not in {}'.format(
                currency, self.allow_currencies))

    def get_value(self, currency=None):
        """Get currencies from coinBase in refCurrency.

            :param ref_currency: reference value
            :return : value of the currency
            :raise NameError: if currency not found
            :example :
                >>> get_currency(ref_currency='EUR')
                920
        """
        pass

    def buy(self, amount, currency, currency_value):
        """Buy currency in EUR, currency is defined at class initialisation.

            :param amount: amount value
            :param currency: currency to buy
            :param currency_value: current currency value.
            :return : currency_value bought and fee amount.
            :example :
            >>> buy(amount=10)
                0.2, 0.01
        """
        pass

    def sell(self, amount, currency, currency_value):
        """Sell currency, currency is defined at class initialisation.

            :param amount: amount value in currency
            :param currency: currency to sell
            :param currency_value: current currency value.
            :return : amount sell in Eur, fee amount in Eur or None,None

            :example :
                >>> sell(amount=0.1, currency='BTC')
                10.1, 0.1
        """
        pass
