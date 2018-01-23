#from coinbase.wallet.client import Client
import random

def coinBaseConnect(object):
    """ coinBase API connection """
    
    def __init__(self, configFile = "config/coinbase.json"):
        """ initialisation of all configuration needed """
        
        self.__dict__ = json.loads(configFile)
        self.user   = self.__dict__['api_key']
        self.key    = self.__dict__['api_secret']
        self.simulation   = self.__dict__['simulation']

        self.client = Client(self.user, self.key)

        self.currency = {}

    def get_currency(self, currency='BTC', refCurrency = 'EUR'):
        """ get currencies from coinBase in refCurrency
            :param currency: value to get
            :param refCurrency: reference value

            :return : value of the currency

            :raise ValueError: if currency not found
            :exemple :
                >>> get_currency(currency='BTC', refCurrency = 'EUR')
                920
        
        """
        
        if self.simulation == 1:

            # price simulation
            if currency in self.currency :
                self.currency[currency] += random.randInt(-100,100)
            
                if(self.currency[currency] == 0)
                    self.currency[currency] = random.randInt(0,100)

            else:
                self.currency[currency] = random.randInt(0,100)

            return self.currency[currency]

        else:

            raise ValueError

        return -1


    def buy_currency(self, currency='BTC', amt):
        """ buy currency
            :param currency: currency to buy
            :param amt: amount value

            :return : boolean which indicate if it succeed,
                      feeAmt ( set to 0 if failed)
            :exemple :
                >>> buy_currency(currency='BTC', amt = '10')
                true, 0.01
        
        """

        if self.simulation == 1:
            
            feeAmt = amt * 0.01

            return True, feeAmt

        else:

            raise ValueError

        return False, 0

    def sell_currency(self, currency='BTC', amount):
        """ sell currency
            :param currency: currency to sell
            :param amt: amount value

            :return : boolean which indicate if it succeed,

            :exemple :
                >>> sell_currency(currency='BTC', amt = '10')
                true
        
        """
        if self.simulation == 1:

            return True

        else:

            raise ValueError

        return False
