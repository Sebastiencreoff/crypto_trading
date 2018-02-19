from coinbase.wallet.client import Client

def Trading(object):
    """ Trading process """
    
    def __init__(self, configFile = "config/trading.json"):
        """ initialisation of all configuration needed """
        
        self.__dict__ = json.loads(configFile)
        self.user   = self.__dict__['user']
        self.key    = self.__dict__['key']
        self.transactionFeeAmt = self.__dict__['transactionFeeAmt']
        
        self.loop = 1
        
    def run(self):
        """ launch the trading process, which will contain:
                - 1 thread for data acquisition
                - 1 thread by currency to deal with
        """

        # client connection
        client = Client(self.user, self.key)
        
        while(self.loop == 1):
        
            currencies = client.get_currencies();
            
            
        
        
    def stop(self):
        """ stop trading """
        
        
        self.loop = 0
        
        pass
        