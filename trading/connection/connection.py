import abc

import trading.connection.dbtransaction


class Connect(object):
    """Connect API connection."""

    allow_currencies = ['BTC', 'BCH', 'ETH', 'LTC']

    def __init__(self, currency, config_dict):
        """Initialisation of all configuration needed.

            :param currency:   currency to deal with ( BTC, etc...)
            :param config_dict: configuration dictionary for connection
        """

        self.currency = currency

        self.database = trading.connection.dbtransaction.DbTransaction(
            currency=currency)

    def allow_currency(self, currency='BTC'):
        """Check if currency is available.

            :param currency: currency value to check
            :raise NameError: if currency not found
            :example :
                >>> allow_currency(currency='BTC')
                920
        """
        import pdb
        pdb.set_trace()
        if currency not in self.allow_currencies:
            raise NameError('Currency: {} not in {}'.format(
                currency, self.allow_currencies))

    def get_currency(self, ref_currency='EUR'):
        """Get currencies from coinBase in refCurrency.

            :param ref_currency: reference value
            :return : value of the currency
            :raise NameError: if currency not found
            :example :
                >>> get_currency(ref_currency='EUR')
                920
        """
        pass

    def buy_currency(self, amount=0):
        """Buy currency, currency is defined at class initialisation.

            :param amount: amount value in the currency

            :return : boolean which indicate if it succeed,
                      feeAmt ( set to None if failed)
            :example :
                >>> buy_currency(amount=10)
                true, 0.01
        """
        pass

    def sell_currency(self,  currency_amt=0):
        """Sell a currency amount, currency is defined at class initialisation.

            :param currency_amt: amount value
            :return : boolean which indicate if it succeed,

            :example :
                >>> sell_currency(currency_amt=10)
                true
        """
        pass

    def reset(self):
        """Reset database.

            :example :
                >>> reset()
        """
        self.database.reset()

    def in_progress(self):
        """Transaction in progress.

            :example :
                >>> in_progress()
                true
        """
        return self.database.get_current_transaction() is not None

    def current_transaction(self):
        return self.database.get_current_transaction()