import abc
import json
import os.path



class Connect(abc.ABC):
    """ coinBase API connection """
    
    def __init__(self, configFile = "config/coinbase.json"):
        """ initialisation of all configuration needed """
        
        assert os.path.isfile(configFile)

        self.__dict__ = json.loads(configFile)

        self.currency = {}
        self.allowCurrency = [ 'BTC', 'BCH', 'ETH', 'LTC']

    def allowCurrency(self, currency = 'BTC'):
        """ check if currency is available
            :param currency: currency value to check

            :raise NameError: if currency not found
            :exemple :
                >>> allowCurrency(currency='BTC', refCurrency = 'EUR')
                920
        
        """
        if currency not in self.allowCurrency:
            raise NameError (" currency : $currency not found" )


    def get_currency(self, currency='BTC', refCurrency = 'EUR'):
        """ get currencies from coinBase in refCurrency
            :param currency: value to get
            :param refCurrency: reference value

            :return : value of the currency

            :raise NameError: if currency not found
            :exemple :
                >>> get_currency(currency='BTC', refCurrency = 'EUR')
                920
        
        """
        pass




    def buy_currency(self, currency='BTC', amt = 0):
        """ buy currency
            :param currency: currency to buy
            :param amt: amount value

            :return : boolean which indicate if it succeed,
                      feeAmt ( set to 0 if failed)
            :exemple :
                >>> buy_currency(currency='BTC', amt = '10')
                true, 0.01
        
        """
        pass


    def sell_currency(self, currency='BTC', amt = 0):
        """ sell currency
            :param currency: currency to sell
            :param amt: amount value

            :return : boolean which indicate if it succeed,

            :exemple :
                >>> sell_currency(currency='BTC', amt = '10')
                true
        
        """
        pass



