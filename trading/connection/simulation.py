import random

from trading.connection import connection

class SimulationConnect(connection.Connect):
    """ coinBase API connection """
    
    def __init__(self, configFile):
        """ initialisation of all configuration needed 
            :param configFile: configuration file
        """
        super().__init__(configFile)



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
        super().allowCurrency(currency)


        # price simulation
        if currency in self.currency :
            self.currency[currency] += random.randint(-100,100)
            
            if self.currency[currency] == 0 :
                self.currency[currency] = random.randint(0,100)

        else:
            self.currency[currency] = random.randint(0,100)

        return self.currency[currency]



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
        super().allowCurrency(currency)

        feeAmt = amt * 0.01

        return True, feeAmt


    def sell_currency(self, currency='BTC', amt = 0):
        """ sell currency
            :param currency: currency to sell
            :param amt: amount value

            :return : boolean which indicate if it succeed,

            :exemple :
                >>> sell_currency(currency='BTC', amt = '10')
                true
        
        """
        super().allowCurrency(currency)

        return True

