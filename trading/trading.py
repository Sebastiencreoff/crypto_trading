from threading import Thread
import time

def Trading(Thread):
    """ Trading process """
    
    def __init__(self, configFile):
        """ initialisation of all configuration needed """
        Thread.__init__(self)


        self.__dict__ = json.loads(configFile)
        self.tradingCurrency   = self.__dict__['tradingCurrency']
        self.transactionFeeAmt = self.__dict__['transactionFeeAmt']
        self.connectionConfig  = self.__dict__["connnectionConfig"]
        self.delay             = self.__dict__["delay"]
        self.loop = 1

         # connection
        if self.__dict__["connection"] eq "coinbase":
            connect = CoinBaseConnect(self.connectionConfig) 
        else:
            connect = SimulationConnect(self.connectionConfig)
        
    def run(self):
        """ launch the trading process, which will contain:
                - 1 thread for data acquisition
                - 1 thread by currency to deal with
        """
        while(self.loop == 1):
        
            currency = connect.get_currency(self.tradingCurrency)

            # save currency
            
            # process trading

            time.sleep(self.delay)
            
        
        
    def stop(self):
        """ stop trading """
        self.loop = 0
        

if __name__ == '__main__':
    
    trading = Trading()

    print("Starting Trading")
    try:
        trading.run()

    except KeyboardInterrupt:
        trading.stop()

        print("\nTrading Finished!")